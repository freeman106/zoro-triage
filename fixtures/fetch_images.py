"""Download only the CrisisMMD photos referenced by the checked-in demo manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    images = sorted({r["image_path"] for r in json.loads(Path(__file__).with_name("reports.json").read_text())})
    def fetch(name):
        if not (args.directory / name).is_file():
            hf_hub_download("QCRI/CrisisMMD", name, repo_type="dataset", local_dir=args.directory)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, images))
    print(f"Ready: {len(images)} images in {args.directory}")
