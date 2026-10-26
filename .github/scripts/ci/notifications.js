const isActionsBot = item => item.user?.login === 'github-actions[bot]' && item.user?.type === 'Bot';
const failed = conclusion => ['failure', 'timed_out'].includes(conclusion);
const code = value => `<code>${String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code>`;

async function upsertComment(github, repo, number, key, body, legacyPrefix) {
  body = body.trim();
  if (!body) return;
  const marker = `<!-- recstore:${key} -->`;
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number: number, per_page: 100,
  });
  // Recognize the old unmarked bot messages as well as new notifications.
  const existing = comments.filter(comment => isActionsBot(comment) && (
    comment.body?.includes(marker) || (legacyPrefix && comment.body?.startsWith(legacyPrefix))
  )).pop();
  if (existing?.body.replace(marker, '').trim() === body) return;
  if (existing) {
    await github.rest.issues.updateComment({
      ...repo, comment_id: existing.id, body: `${marker}\n\n${body}`,
    });
  } else {
    await github.rest.issues.createComment({
      ...repo, issue_number: number, body: `${marker}\n\n${body}`,
    });
  }
}

async function notifySize(github, context) {
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo, pull_number: context.issue.number,
  });
  // Match the size labeler's existing XL boundary (additions + deletions).
  if (pr.additions + pr.deletions < 1000) return;
  const body = 'This PR exceeds the recommended size of 1000 lines. ' +
    'Please make sure you are NOT addressing multiple issues with one PR. ' +
    'Note this PR might be rejected due to its size.';
  await upsertComment(github, context.repo, pr.number, 'pr-size', body,
    'This PR exceeds the recommended size of 1000 lines.');
}

async function notifyFormat(github, context, run) {
  // Fork workflow_run payloads can omit pull_requests. Query open PRs in that case.
  let prs;
  if (run.pull_requests?.length) {
    prs = await Promise.all(run.pull_requests.map(async pr => (
      await github.rest.pulls.get({ ...context.repo, pull_number: pr.number })
    ).data));
  } else {
    prs = await github.paginate(github.rest.pulls.list, {
      ...context.repo, state: 'open', per_page: 100,
    });
  }
  for (const pr of prs) {
    // A slow failure from an older commit must not notify a newer PR revision.
    if (pr.state !== 'open' || pr.head.sha !== run.head_sha ||
        pr.head.ref !== run.head_branch || !pr.head.repo) continue;
    const url = `${pr.head.repo.html_url}/actions/workflows/format.yml`;
    const body = [
      '## Format check failed',
      '',
      'To apply the formatting fixes:',
      '',
      `1. Open [the Format workflow](${url}) in the PR source repository.`,
      '2. Click **Run workflow**.',
      `3. Select this PR's branch: ${code(pr.head.ref)}.`,
      '4. Set **mode** to **sweep**, then click **Run workflow** to start it.',
      '',
      'This formats the whole tree. If files change, **github-actions[bot] will create and push a new commit to this branch**.',
    ].join('\n');
    await upsertComment(github, context.repo, pr.number, 'format-failure', body);
  }
}

async function notifyDocker(github, context, run, logsUrl) {
  // A rerun keeps the same run ID; report each failed run only once, even if closed.
  const marker = `<!-- recstore:docker-test-failure:${run.id} -->`;
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    ...context.repo, state: 'all', creator: 'github-actions[bot]', per_page: 100,
  });
  if (issues.some(issue => !issue.pull_request && isActionsBot(issue) && issue.body?.includes(marker))) return;
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRunAttempt, {
    ...context.repo, run_id: run.id, attempt_number: run.run_attempt, per_page: 100,
  });
  const artifacts = await github.paginate(github.rest.actions.listWorkflowRunArtifacts, {
    ...context.repo, run_id: run.id, per_page: 100,
  });
  const failures = jobs.filter(job => failed(job.conclusion));
  // Follow .github/ISSUE_TEMPLATE/bug_report.yml's fields and bug label.
  const body = [
    marker,
    '',
    '### Type',
    '',
    'Other',
    '',
    '### 🐛 Description',
    '',
    `Docker Test CI failed: [run #${run.run_number}, attempt ${run.run_attempt}](${run.html_url}).`,
    ...(run.pull_requests || []).map(pr => `Related PR: #${pr.number}`),
    '',
    'Failed jobs and steps:',
    ...failures.map(job => {
      const steps = (job.steps || []).filter(step => failed(step.conclusion));
      return `- [${code(job.name)}](${job.html_url})` +
        (steps.length ? `: ${steps.map(step => code(step.name)).join(', ')}` : '');
    }),
    ...(failures.length ? [] : ['See the workflow run for failure details.']),
    '',
    '### 🛠️ Environment',
    '',
    `- Repository: ${code(`${context.repo.owner}/${context.repo.repo}`)}`,
    `- Event: ${code(run.event)}`,
    `- Branch: ${code(run.head_branch)}`,
    `- Commit: ${code(run.head_sha)}`,
    `- Conclusion: ${code(run.conclusion)}`,
    `- Runner labels: ${code([...new Set(failures.flatMap(job => job.labels || []))].join(', ') || 'See run logs')}`,
    '',
    '### Logs',
    '',
    logsUrl ? `[Download full workflow logs (ZIP)](${logsUrl})` :
      `The workflow log archive could not be attached. [View the run logs](${run.html_url}).`,
    ...artifacts.filter(artifact => !artifact.expired && artifact.name === 'docker-image-test-logs')
      .map(artifact => `[Download docker-image-test-logs (ZIP)](${run.html_url}/artifacts/${artifact.id})`),
  ].join('\n');
  await github.rest.issues.create({
    ...context.repo, title: `[Bug]: Docker Test CI failed (run #${run.run_number})`,
    labels: ['bug'], body,
  });
}

async function notifyFailure(github, context, logsUrl) {
  const run = context.payload.workflow_run;
  if (!failed(run.conclusion)) return;
  if (run.name === 'Format') await notifyFormat(github, context, run);
  if (run.name === 'Docker Test' && context.repo.owner === 'RecStore' && context.repo.repo === 'RecStore' &&
      run.head_repository?.full_name === 'RecStore/RecStore' && run.head_branch === 'main' &&
      !['pull_request', 'pull_request_target'].includes(run.event)) {
    await notifyDocker(github, context, run, logsUrl);
  }
}

module.exports = { upsertComment, notifySize, notifyFailure };
