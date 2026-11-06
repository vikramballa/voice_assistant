"""Deciding what to do with a sentence. Both brains return the same plan: a list of steps
[{"action": name, "args": [...]}] (see actions/__init__.py).

    fast      Laya, a ModernBERT classifier fine-tuned on Mac commands (~80 ms). Exact text rules first,
              then Laya picks an intent per clause; None = "not sure, ask the slow brain".
    slow      a local LLM (qwen3 4B on MLX, or Ollama) plans the steps (~0.8 s)
    prompts   what the LLM is told: the actions, installed apps, web services, rules
    repair    every LLM step is checked against the user's words before it runs
    mlx_llm   the MLX engine: grammar-constrained decoding and a saved prompt cache
"""
