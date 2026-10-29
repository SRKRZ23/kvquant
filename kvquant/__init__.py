"""kvquant — tested KV-cache quantization for Hugging Face transformers >= 5.

Most KV-quant snippets online silently no-op on transformers 5.x because they mutate
``cache.key_cache[i]`` (a non-persisting compat accessor) instead of the real
``cache.layers[i].keys`` / ``.values``. kvquant mutates the right tensors and ships a
stress test so you can never be fooled.
"""
from .core import quantize_cache, quantize_dequantize
from .bench import kv_ppl_sweep

__all__ = ["quantize_cache", "quantize_dequantize", "kv_ppl_sweep"]
__version__ = "0.1.0"
