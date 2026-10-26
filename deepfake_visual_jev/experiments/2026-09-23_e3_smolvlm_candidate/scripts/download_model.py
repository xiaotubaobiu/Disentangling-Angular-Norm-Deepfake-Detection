"""Fetch the pinned official model; record local hashes before experimentation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from huggingface_hub import snapshot_download

PHASE = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"


def main():
    target = PHASE / "results/pretrained/smolvlm_256m"
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=target,
                      allow_patterns=["*.json", "*.txt", "model.safetensors", "README.md"],
                      ignore_patterns=["onnx/*"])
    files = []
    for path in sorted(target.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"path": str(path.relative_to(PHASE)), "bytes": path.stat().st_size, "sha256": digest})
    result = {"model_id": MODEL_ID, "revision": REVISION, "files": files,
              "dependencies": {name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "huggingface-hub", "safetensors", "tokenizers")}}
    (PHASE / "results/pretrained/manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
