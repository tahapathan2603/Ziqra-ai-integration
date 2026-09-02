"""
What the onboarding voice check actually receives.

    modal run modal_voicecheck_probe.py

Calls the DEPLOYED /extract from inside Modal, so the client token comes from
the container's own environment and never leaves Modal, and prints exactly the
paths app/src/components/ui/ScoreCard.tsx reads:

    level2.analysis.engagement.engagement_score          the ring
    level2.analysis.pronunciation.pronunciation_score    the five bars
    level2.analysis.intonation.intonation_score
    level2.analysis.engagement.energy_patterns.energy_engagement_score
    level2.analysis.engagement.pause_patterns.pause_engagement_score
    level2.analysis.engagement.speaking_dynamics.dynamics_score

Written because that screen rendered an empty ring and five full bars, which
is what `undefined` looks like in both of those widgets — so the question is
whether the pipeline stopped sending these, or sends them as null, or the app
is reading the wrong path.
"""

import json

import modal

image = modal.Image.debian_slim().pip_install("requests").add_local_dir(
    "backend/test_audio/saa_indian_samples", remote_path="/samples"
)
# The full analysis is written here as well as summarised, so the app can be
# tested against what the pipeline really returns instead of a hand-written
# imitation of it.
cache = modal.Volume.from_name("ziqra-tts-cache", create_if_missing=True)
FIXTURE_DIR = "/cache/fixtures"
app = modal.App("ziqra-voicecheck-probe", image=image)

URL = "https://ramramjibkn--ziqra-audio-api-fastapi-app.modal.run"

# The paths the ScoreCard reads, as (label, dotted path) pairs.
READS = [
    ("ring: engagement_score", "engagement.engagement_score"),
    ("bar: pronunciation", "pronunciation.pronunciation_score"),
    ("bar: intonation", "intonation.intonation_score"),
    ("bar: energy", "engagement.energy_patterns.energy_engagement_score"),
    ("bar: pause control", "engagement.pause_patterns.pause_engagement_score"),
    ("bar: speaking dynamics", "engagement.speaking_dynamics.dynamics_score"),
]


def dig(node, path: str):
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return "<<MISSING>>"
        node = node[part]
    return node


@app.function(timeout=900, volumes={"/cache": cache}, secrets=[modal.Secret.from_name("custom-secret")])
def probe(sample: str) -> dict:
    import os
    from pathlib import Path

    import requests

    audio = Path("/samples") / sample
    with audio.open("rb") as handle:
        response = requests.post(
            f"{URL}/extract",
            headers={"X-Client-Token": os.environ["CLIENT_TOKEN"]},
            files={"file": (audio.name, handle, "audio/mpeg")},
            timeout=600,
        )
    if response.status_code != 200:
        return {"sample": sample, "status": response.status_code, "body": response.text[:500]}

    body = response.json()
    analysis = (body.get("level2") or {}).get("analysis") or {}

    directory = Path(FIXTURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"analysis-{sample.replace('.', '-')}.json").write_text(json.dumps(analysis, indent=2, default=str))
    cache.commit()

    return {
        "sample": sample,
        "status": 200,
        "top_level_keys": sorted(body.keys()),
        "level2_keys": sorted((body.get("level2") or {}).keys()),
        "analysis_keys": sorted(analysis.keys()),
        "engagement_keys": sorted((analysis.get("engagement") or {}).keys()),
        "score_card_reads": {label: dig(analysis, path) for label, path in READS},
        "meta": body.get("meta"),
    }


@app.local_entrypoint()
def main(samples: str = "hindi3.mp3,tamil12.mp3"):
    for sample in samples.split(","):
        print(json.dumps(probe.remote(sample.strip()), indent=2, default=str))
