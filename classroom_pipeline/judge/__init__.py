"""LLM late-fusion teacher → lightweight student judge pipeline.

Pipeline:
  structured multimodal evidence
  → LLM teacher (text-only, per-candidate judgements)
  → silver labels
  → student judge training (VerifierMLP-compatible)
  → replace verifier scoring step
"""

__version__ = "0.1.0"
