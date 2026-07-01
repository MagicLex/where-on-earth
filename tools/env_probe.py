"""Probe what the torch job env actually ships (versions + thread count)."""
import importlib
for m in ["torch", "torchvision", "transformers", "PIL", "pandas", "numpy", "sklearn", "requests"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"), flush=True)
    except ImportError as e:
        print(m, "MISSING", flush=True)
import torch, os
print("torch threads:", torch.get_num_threads(), "| cpus:", os.cpu_count(), flush=True)
