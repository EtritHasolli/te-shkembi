from pathlib import Path

import kagglehub

dataset_handle = "jangedoo/utkface-new"
base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "curated-dataset"

if output_dir.exists() and any(output_dir.iterdir()):
    print("Dataset already exists at:", output_dir)
else:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(dataset_handle, output_dir=str(output_dir))
    print("Path to dataset files:", path)

