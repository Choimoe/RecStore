#!/bin/bash
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
cd "$SCRIPT_DIR"
set -x
set -e
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
bash install.sh
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

nvm install --lts
npm install -g @openai/codex

mkdir -p ~/.codex
cp ${SCRIPT_DIR}/auth.json ~/.codex/auth.json
cp ${SCRIPT_DIR}/config.toml ~/.codex/config.toml

curl https://cursor.com/install -fsS | bash

ln -s /app/RecStore/dockerfiles/codex/.cursor-agent ~/.cursor

export HAPI_API_URL=http://10.0.2.196:3006
ln -sfn /app/RecStore/dockerfiles/codex/.hapi ~/.hapi


npm install -g @twsxtd/hapi --registry=https://registry.npmjs.org

# HAPI_LISTEN_HOST=10.0.2.196 hapi hub --relay


git config --global user.name "Minhui Xie"
git config --global user.email "645214784@qq.com"

cp -r /app/RecStore/dockerfiles/codex/.claude ~/.claude