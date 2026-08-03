# run litellm proxy
export INFINI_API_KEY=sk-xxx
export CODEX_PROXY_STRIP_EXEC=1
PORT=4100 bash scripts/codex_proxy/start_proxy.sh

# before run main.py
export CODEX_BASE_URL=http://127.0.0.1:4100
export CODEX_API_KEY=sk-codex-proxy
export CODEX_MODEL=deepseek-v4-flash
