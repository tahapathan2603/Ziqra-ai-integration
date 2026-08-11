"""
Training pipeline for the knowledge-distillation student models
(Articulation -> Llama, Delivery -> Qwen).

Currently contains only `data/` -- preparing the finished teacher-generation
output into a format supervised fine-tuning can consume. No model loading,
LoRA, or training-loop logic lives here yet; see `data/`'s docstring for
what is implemented so far and what deliberately is not.
"""
