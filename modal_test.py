"""
One-off REMOTE TEST entrypoint for the audio feature-extraction pipeline --
NOT a deployment. Runs the EXISTING pipeline (backend/api/pipeline.py's
extract_features -- unmodified, not duplicated) on a Modal GPU container for
exactly one audio file per invocation, and prints/optionally saves its
Level 1 + Level 2 output. No HTTP server, no persistent endpoint -- that is
modal_app.py's job (`modal deploy modal_app.py`), which this script does not
touch or need running.

Reuses modal_app.py's `image` (same apt/pip installs, same model prefetch,
same local source mount) and `MODAL_SECRET_NAME` rather than redefining
them, so there is exactly one place that decides what the container looks
like, whether you're testing (`modal run`, this file) or serving (`modal
deploy`, modal_app.py).

Usage:
    modal run modal_test.py --audio-path backend/test_audio/audio_ana2.mp3

Save the output locally via Modal's own -w/--write-result flag. Note the
position: it goes BEFORE the file argument, not after -- verified against
the installed CLI (modal run --help), this is Modal's own argument-parsing
order (everything after the file is handed to this script's own
--audio-path, not to `modal run` itself):
    modal run --write-result out.json modal_test.py --audio-path backend/test_audio/audio_ana2.mp3

On an unreliable connection, add -d/--detach:
    modal run --detach modal_test.py --audio-path backend/test_audio/audio_ana2.mp3

`modal run` normally creates an EPHEMERAL app whose lifetime is tied to the
local process: the client holds a heartbeat to Modal, and if that heartbeat
stops being answered the app is torn down -- even though the actual work
(image build, model downloads, GPU inference) all runs server-side and
doesn't need your machine at all. On a flaky link this shows up as repeated
"Loop attempt for _run_app.<locals>.heartbeat failed" tracebacks and an
apparently-hung "Creating objects..." spinner. --detach decouples the two so
the run survives the local process disconnecting; recover the output with
`modal app logs <app-id>`. Note that --detach and --write-result don't
combine usefully -- there's no local process left to receive the return
value -- so detached runs rely on the printed summary and logs instead.
"""

import modal

from modal_app import MODAL_SECRET_NAME, image

# A distinct App name from modal_app.py's "ziqra-audio-api" -- keeps ad hoc
# `modal run` sessions (ephemeral; torn down when the command exits)
# visibly separate from the real deployed web app in the Modal dashboard,
# even though both share the identical container image defined once in
# modal_app.py.
app = modal.App("ziqra-audio-api-test", image=image)


@app.function(
    gpu="A10G",  # same accelerator the deployed endpoint uses -- Whisper large-v3 and
                 # the wav2vec2 phoneme model both run on it (see phoneme_accuracy.py's
                 # _resolve_device, which now prefers CUDA).
    # Deliberately more generous than modal_app.py's serving timeout (1200):
    # this is a watched, one-off run where the failure mode to avoid is a
    # long file getting killed near the end after minutes of real work.
    # A cold container spends its first minutes importing torch/transformers
    # and loading Whisper large-v3 + wav2vec2 onto the GPU before any audio
    # is touched, and all of that counts against this budget.
    timeout=1800,
    # Requirement: inject the existing Modal Secret. Note this has no
    # functional effect in THIS script specifically -- CLIENT_TOKEN is
    # read by backend/api/main.py's HTTP auth dependency, and there is no
    # HTTP layer anywhere in this local_entrypoint -> remote function call.
    # It's included anyway because it was explicitly required, and because
    # this is a reasonable place to confirm the secret actually resolves
    # and injects correctly before you ever run `modal deploy` for real.
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
def run_extraction(audio_bytes: bytes, filename: str) -> dict:
    """
    Runs ON the Modal container. Writes the uploaded bytes to a temp file --
    there's no shared filesystem between your machine and the container, so
    the audio has to travel as a function argument, not a path -- then
    calls the EXISTING pipeline: backend.api.pipeline.extract_features, the
    exact same function backend/api/main.py's /extract endpoint calls.
    Returns its {"session_id", "level1", "level2", "meta"} dict unchanged.

    No feature-extraction logic lives here; this function is orchestration
    only, mirroring backend/api/main.py's own temp-file handling
    (TemporaryDirectory, cleaned up automatically on the way out).
    """
    import os
    import tempfile

    from backend.api.pipeline import extract_features

    ext = os.path.splitext(filename)[1] or ".wav"
    with tempfile.TemporaryDirectory(prefix="ziqra_modal_test_") as tmp_dir:
        audio_path = os.path.join(tmp_dir, f"upload{ext}")
        chunks_dir = os.path.join(tmp_dir, "chunks")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        return extract_features(audio_path, chunks_dir)


@app.local_entrypoint()
def main(audio_path: str):
    """
    Runs LOCALLY (this is what `modal run` actually executes on your
    machine): reads the file at audio_path off your disk, ships its bytes
    to run_extraction() on the Modal container, prints a summary, and
    returns the full result as a JSON STRING -- deliberately a string, not
    a dict, so Modal's `-w/--write-result` flag (which only accepts str or
    bytes -- verified against the installed CLI) can save it directly. See
    this file's module docstring for the exact command.
    """
    import json
    from pathlib import Path

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    print(f"Uploading {path.name} ({path.stat().st_size:,} bytes) -- running remotely on Modal GPU...")
    result = run_extraction.remote(path.read_bytes(), path.name)

    print(f"session_id: {result['session_id']}")
    print(f"meta: {result['meta']}")
    print(f"level1 keys: {sorted(result['level1'].keys())}")
    print(f"level2.analysis keys: {sorted(result['level2']['analysis'].keys())}")

    return json.dumps(result, indent=2, ensure_ascii=False)
