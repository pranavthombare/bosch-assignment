import os
import requests
from pathlib import Path

def download_file(url, dest_path, token=None):
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code in [401, 403]:
            print(f"Error: Authentication required to download {url}")
            print(f"Please download {dest_path.name} manually from the dataset website and place it in {dest_path.parent}.")
            return
        else:
            raise e
    bytes_written = 0
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            bytes_written += file.write(data)
    print(f"Saved {dest_path.name} ({bytes_written:,} bytes)")

def main():
    # Setup directories
    base_dir = Path(__file__).parent.parent.parent
    drivelm_dir = base_dir / "data" / "drivelm"
    drivelm_dir.mkdir(parents=True, exist_ok=True)
    
    # DriveLM URLs from HuggingFace
    urls = {
        "v1_1_train_nus.json": "https://huggingface.co/datasets/OpenDriveLab/DriveLM/resolve/main/v1_1_train_nus.json"
    }
    
    for filename, url in urls.items():
        dest_path = drivelm_dir / filename
        if not dest_path.exists():
            print(f"Downloading {filename}...")
            download_file(url, dest_path, token=os.getenv("HF_TOKEN"))
        else:
            print(f"{filename} already exists at {dest_path}")

if __name__ == "__main__":
    main()
