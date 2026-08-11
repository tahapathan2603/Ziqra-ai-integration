"""FastAPI wrapper around the existing audio feature-extraction pipeline.

This package adds no feature-extraction logic of its own — it orchestrates
the same functions the Gradio dev UI (backend/preprocessing/vad_ui.py) already
calls, and returns their Level 1 + Level 2 output as JSON instead of as a
formatted display. See pipeline.py's module docstring for the exact call
chain reused.
"""
