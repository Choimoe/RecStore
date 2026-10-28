// Run with: node --test tools/ci/tests/test_ci_notifications.js
const assert = require('node:assert/strict');
const test = require('node:test');
const { upsertComment, notifySize, notifyFailure } = require('../../../.github/scripts/ci/notifications.js');

const repo = { owner: 'RecStore', repo: 'RecStore' };
const bot = { login: 'github-actions[bot]', type: 'Bot' };
const docs = '## 📚 Documentation Update\n\nReview docs/storage.md';
const sizeWarning = 'This PR exceeds the recommended size of 1000 lines. ' +
  'Please make sure you are NOT addressing multiple issues with one PR. ' +
  'Note this PR might be rejected due to its size.';

function fixture() {
  const state = {
    comments: [], issues: [], writes: [], reads: [], jobs: [], artifacts: [],
    prs: [{
      number: 42, state: 'open', additions: 800, deletions: 300,
      head: { sha: 'current', ref: 'fix/format', repo: { html_url: 'https://github.com/contributor/RecStore' } },
    }],
  };
  const read = (name, value) => async args => {
    state.reads.push({ name, args });
    return { data: value(args) };
  };
  const write = (name, effect) => async args => {
    state.writes.push({ name, args });
    effect(args);
  };
  const github = {
    rest: {
      issues: {
        listComments: read('comments', args => state.comments.filter(c => c.issue_number === args.issue_number)),
        listForRepo: read('issues', () => state.issues),
        createComment: write('createComment', args => state.comments.push({ id: state.comments.length + 1, user: bot, ...args })),
        updateComment: write('updateComment', args => Object.assign(state.comments.find(c => c.id === args.comment_id), { body: args.body })),
        addLabels: write('addLabels', args => {
          const issue = state.issues.find(item => item.number === args.issue_number);
          issue.labels = [...(issue.labels || []), ...args.labels];
        }),
        update: write('updateIssue', args => Object.assign(
          state.issues.find(item => item.number === args.issue_number), args,
        )),
        create: write('createIssue', args => state.issues.push({ number: 101, state: 'open', user: bot, ...args })),
      },
      pulls: {
        get: read('pr', args => state.prs.find(pr => pr.number === args.pull_number)),
        list: read('prs', () => state.prs),
      },
      actions: {
        listJobsForWorkflowRunAttempt: read('jobs', () => state.jobs),
        listWorkflowRunArtifacts: read('artifacts', () => state.artifacts),
      },
    },
    paginate: async (endpoint, args) => {
      assert.equal(args.per_page, 100);
      return (await endpoint(args)).data;
    },
  };
  const run = {
    id: 700, run_number: 50, run_attempt: 1, name: 'Format', conclusion: 'failure',
    head_sha: 'current', head_branch: 'fix/format', event: 'pull_request',
    html_url: 'https://github.com/RecStore/RecStore/actions/runs/700', pull_requests: [],
  };
  const context = { repo, issue: { number: 42 }, payload: { workflow_run: run } };
  return { github, state, context, run };
}

test('documentation comments are created once, stay silent, then update in place', async () => {
  const { github, state } = fixture();
  const notify = body => upsertComment(github, repo, 42, 'documentation-update', body, '## 📚 Documentation Update');
  await notify(docs);
  await notify(`${docs}\n`);
  assert.deepEqual(state.writes.map(w => w.name), ['createComment']);
  await notify(`${docs}\nReview docs/ps.md`);
  assert.deepEqual(state.writes.map(w => w.name), ['createComment', 'updateComment']);
  assert.equal(state.comments.length, 1);
  await notify('');
  assert.equal(state.writes.length, 2);
});

test('old unmarked bot comments are reused without modifying identical content', async () => {
  const { github, state, context } = fixture();
  state.comments.push(
    { id: 1, issue_number: 42, user: bot, body: `${docs}\n` },
    { id: 2, issue_number: 42, user: bot, body: `${sizeWarning}\n` },
  );
  await upsertComment(github, repo, 42, 'documentation-update', docs, '## 📚 Documentation Update');
  await notifySize(github, context);
  assert.equal(state.writes.length, 0);
  await upsertComment(github, repo, 42, 'documentation-update', `${docs}\nNew doc`, '## 📚 Documentation Update');
  assert.equal(state.writes[0].args.comment_id, 1);
});

test('human comments cannot impersonate notification state', async () => {
  const { github, state } = fixture();
  state.comments.push({
    id: 1, issue_number: 42, user: { login: 'contributor', type: 'User' },
    body: `<!-- recstore:documentation-update -->\n\n${docs}`,
  });
  await upsertComment(github, repo, 42, 'documentation-update', docs, '## 📚 Documentation Update');
  assert.equal(state.writes[0].name, 'createComment');
  assert.equal(state.comments[0].user.type, 'User');
});

test('size warning counts additions and deletions, matches the cutoff, and stays silent on further commits', async () => {
  const { github, state, context } = fixture();
  state.prs[0].additions = 699;
  await notifySize(github, context);
  assert.equal(state.writes.length, 0);
  state.prs[0].additions = 700;
  await notifySize(github, context);
  state.prs[0].additions = 1200;
  await notifySize(github, context);
  assert.equal(state.writes.length, 1);
  assert.ok(state.comments[0].body.includes(sizeWarning));
});

test('fork Format failures without PR metadata use the source repo and branch', async () => {
  const { github, state, context, run } = fixture();
  await notifyFailure(github, context);
  const body = state.comments[0].body;
  assert.ok(body.includes('https://github.com/contributor/RecStore/actions/workflows/format.yml'));
  assert.ok(body.includes('<code>fix/format</code>'));
  for (const text of ['Run workflow', '**mode**', '**sweep**', 'create and push a new commit']) {
    assert.ok(body.includes(text));
  }
  run.id += 1;
  await notifyFailure(github, context);
  assert.equal(state.writes.length, 1);
});

test('Format failures refresh linked PRs and ignore stale, closed, missing, or unrelated heads', async () => {
  for (const change of ['sha', 'branch', 'closed', 'deleted']) {
    const { github, state, context, run } = fixture();
    run.pull_requests = [{ number: 42 }];
    if (change === 'sha') state.prs[0].head.sha = 'newer';
    if (change === 'branch') state.prs[0].head.ref = 'another-branch';
    if (change === 'closed') state.prs[0].state = 'closed';
    if (change === 'deleted') state.prs[0].head.repo = null;
    await notifyFailure(github, context);
    assert.equal(state.writes.length, 0, change);
    assert.ok(state.reads.some(r => r.name === 'pr'));
  }
});

test('Format supports same-repository PRs and escapes branch names', async () => {
  const { github, state, context, run } = fixture();
  run.pull_requests = [{ number: 42 }];
  run.head_branch = state.prs[0].head.ref = 'fix/<tag>&branch';
  state.prs[0].head.repo.html_url = 'https://github.com/RecStore/RecStore';
  await notifyFailure(github, context);
  assert.ok(state.comments[0].body.includes('https://github.com/RecStore/RecStore/actions/workflows/format.yml'));
  assert.ok(state.comments[0].body.includes('<code>fix/&lt;tag&gt;&amp;branch</code>'));
});

test('successful, cancelled, skipped, and unrelated workflows do not notify', async () => {
  for (const name of ['Format', 'Docker Test']) {
    for (const conclusion of ['success', 'cancelled', 'skipped']) {
      const { github, state, context, run } = fixture();
      Object.assign(run, { name, conclusion });
      await notifyFailure(github, context);
      assert.equal(state.reads.length, 0);
      assert.equal(state.writes.length, 0);
    }
  }
  const { github, state, context, run } = fixture();
  run.name = 'Other workflow';
  await notifyFailure(github, context);
  assert.equal(state.writes.length, 0);
});

test('Docker failures follow the bug template, use CI labels, and deduplicate the same run', async () => {
  const { github, state, context, run } = fixture();
  Object.assign(run, {
    name: 'Docker Test', head_branch: 'main', event: 'push',
    head_repository: { full_name: 'RecStore/RecStore' },
  });
  run.pull_requests = [{ number: 42 }];
  state.jobs.push({
    name: 'Pull image and run tests', conclusion: 'failure', html_url: `${run.html_url}/job/900`,
    labels: ['ubuntu-latest'], steps: [{ name: 'Build <core>', conclusion: 'failure' }],
  });
  state.artifacts.push({ id: 800, name: 'docker-image-test-logs', expired: false });
  const logsUrl = 'https://github.com/RecStore/RecStore/actions/runs/701/artifacts/801';
  await notifyFailure(github, context, logsUrl);
  const issue = state.issues[0];
  assert.equal(issue.title, '[Bug][CI]: Docker Test CI failed (run #50)');
  assert.deepEqual(issue.labels, ['bug', 'CI/CD']);
  for (const text of ['### Type\n\nOther', '### 🐛 Description', '### 🛠️ Environment', run.html_url,
    '#42', 'Build &lt;core&gt;', 'ubuntu-latest', 'docker-image-test-logs', logsUrl, `${run.html_url}/artifacts/800`]) {
    assert.ok(issue.body.includes(text), text);
  }
  assert.equal(state.reads.find(r => r.name === 'jobs').args.attempt_number, 1);
  issue.state = 'closed';
  run.run_attempt = 2;
  await notifyFailure(github, context);
  assert.equal(state.writes.length, 1);
  run.id += 1;
  await notifyFailure(github, context);
  assert.equal(state.writes.length, 2);
});

test('Docker failures reply to an existing open tracker instead of creating another issue', async () => {
  const { github, state, context, run } = fixture();
  Object.assign(run, {
    name: 'Docker Test', head_branch: 'main', event: 'push',
    head_repository: { full_name: 'RecStore/RecStore' },
  });
  state.issues.push({
    number: 99, state: 'open', user: bot, labels: [{ name: 'bug' }],
    title: '[Bug]: Docker Test CI failed (run #49)',
    body: '<!-- recstore:docker-test-failure:699 -->',
  });
  state.issues.push({
    number: 100, state: 'open', user: bot, labels: ['bug', 'CI/CD'],
    title: '[Bug][CI]: Docker Test CI failed (run #50)',
    body: '<!-- recstore:docker-test-failure:698 -->',
  });

  await notifyFailure(github, context);

  assert.equal(state.writes.some(write => write.name === 'createIssue'), false);
  assert.deepEqual(state.writes.map(write => write.name), ['updateIssue', 'addLabels', 'createComment']);
  assert.equal(state.issues[0].title, '[Bug][CI]: Docker Test CI failed (run #49)');
  assert.deepEqual(state.writes[1].args.labels, ['CI/CD']);
  assert.equal(state.comments[0].issue_number, 99);
  assert.ok(state.comments[0].body.includes('<!-- recstore:docker-test-failure:700 -->'));
  assert.ok(state.comments[0].body.includes(run.html_url));

  run.run_attempt = 2;
  await notifyFailure(github, context);
  assert.equal(state.comments.length, 1);
  assert.equal(state.writes.at(-1).name, 'updateComment');
  assert.ok(state.comments[0].body.includes('attempt 2'));
});

test('Docker timeouts without failed job details still create an actionable issue', async () => {
  const { github, state, context, run } = fixture();
  Object.assign(run, {
    name: 'Docker Test', conclusion: 'timed_out', event: 'push', head_branch: 'main',
    head_repository: { full_name: 'RecStore/RecStore' },
  });
  state.issues.push({ user: { login: 'contributor', type: 'User' }, body: '<!-- recstore:docker-test-failure:700 -->' });
  await notifyFailure(github, context);
  assert.equal(state.writes.length, 1);
  assert.ok(state.writes[0].args.body.includes('See the workflow run for failure details.'));
  assert.ok(state.writes[0].args.body.includes('<code>timed_out</code>'));
  assert.ok(state.writes[0].args.body.includes('log archive could not be attached'));
});

test('Docker reports only upstream main runs, excluding forks, other branches, and PR tests', async () => {
  for (const excluded of ['fork-repository', 'fork-head', 'other-branch', 'pull_request', 'pull_request_target']) {
    const { github, state, context, run } = fixture();
    Object.assign(run, {
      name: 'Docker Test', event: 'push', head_branch: 'main',
      head_repository: { full_name: 'RecStore/RecStore' },
    });
    context.repo = { ...repo };
    if (excluded === 'fork-repository') context.repo.owner = 'contributor';
    if (excluded === 'fork-head') run.head_repository.full_name = 'contributor/RecStore';
    if (excluded === 'other-branch') run.head_branch = 'feature';
    if (excluded.startsWith('pull_request')) run.event = excluded;
    await notifyFailure(github, context);
    assert.equal(state.reads.length, 0, excluded);
    assert.equal(state.writes.length, 0, excluded);
  }
});
