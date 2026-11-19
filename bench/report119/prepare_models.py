"""Cache only the files used by the supervised baselines."""
from huggingface_hub import snapshot_download

for repo in ("openai/clip-vit-base-patch32", "sentence-transformers/clip-ViT-B-32-multilingual-v1",
             "beomi/KcELECTRA-base-v2022"):
    print(repo, flush=True)
    snapshot_download(repo, allow_patterns=["*.json", "*.txt", "model.safetensors", "2_Dense/model.safetensors",
                                            "pytorch_model.bin", "2_Dense/pytorch_model.bin"])
