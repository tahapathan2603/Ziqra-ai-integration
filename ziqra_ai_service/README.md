# ziqra_ai_service

Modal deployment for the Ziqra audio-extraction API. This folder is infra
only — the FastAPI app, routes, and pipeline logic live in `backend/api/`
and are served here unchanged.

## Deploy

```bash
pip install modal
modal setup                              # one-time browser auth
modal deploy ziqra_ai_service/app.py      # persistent URL
```

Requires a Modal Secret named `custom-secret` with a `CLIENT_TOKEN` key
(see `backend/api/main.py`'s `verify_client_token`).

## Behavior

- Cold-starts a GPU (A10G) container on the first request after idle.
- Stays warm for 30s after the last request (`scaledown_window=30`), then
  scales back to zero.
- `enable_memory_snapshot=True` skips re-importing torch/transformers on
  repeat cold starts (a real chunk of first-hit latency) by restoring a
  post-import snapshot instead of re-running Python's import machinery.
- The two ML models are baked into the image at *build* time
  (`_prefetch_models`), so a cold start never re-downloads them — only
  loads them into GPU memory on that container's first request.
- `max_containers=2` is a cost guard for the testing phase, not a
  performance ceiling.

## Regenerate requirements-modal.txt

Whenever `requirements.txt` (repo root) changes:

```bash
python ziqra_ai_service/app.py
```

## Dev / debug (no Modal)

```bash
uvicorn backend.api.main:app --reload --port 8000
```
