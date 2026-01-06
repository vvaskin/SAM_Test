from dotenv import load_dotenv
import os
import torch
import torchvision

print("PyTorch versions:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is avaliable:", torch.cuda.is_available())

# Loading the access key
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is not None:
    os.environ["HF_TOKEN"] = hf_token
