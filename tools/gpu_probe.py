import torch
print("cuda available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), flush=True)
