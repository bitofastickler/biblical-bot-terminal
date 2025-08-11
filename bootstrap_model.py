# bootstrap_model.py
import os
from pathlib import Path

# Defaults (you can override with env vars)
DEFAULT_REPO = os.getenv("MODEL_REPO", "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF")
SMALL_FILE   = os.getenv("MODEL_SMALL",  "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")
LARGE_FILE   = os.getenv("MODEL_LARGE",  "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf")

def _available_ram_gb():
    """Return available RAM in GB, or None if psutil not installed."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        print("[bootstrap_model] psutil not available — skipping RAM check.")
        return None

def _select_filename():
    """
    Pick filename by RAM (or env override).
    - MODEL_FILENAME env var wins.
    - Else choose LARGE if >= ~24 GB available, else SMALL.
    """
    override = os.getenv("MODEL_FILENAME")
    if override:
        print(f"[bootstrap_model] Using MODEL_FILENAME override: {override}")
        return override

    ram = _available_ram_gb()
    if ram is None:
        print("[bootstrap_model] Defaulting to SMALL model.")
        return SMALL_FILE
    if ram >= 24:
        print(f"[bootstrap_model] Detected ~{ram:.1f} GB available — choosing LARGE model.")
        return LARGE_FILE
    else:
        print(f"[bootstrap_model] Detected ~{ram:.1f} GB available — choosing SMALL model.")
        return SMALL_FILE

def ensure_model(models_dir: str = "models", auto_download: bool = True) -> str:
    """
    Ensure the chosen GGUF file exists locally. If not and auto_download=True,
    fetch it from Hugging Face (resumable). Returns absolute path to the file.

    Raises FileNotFoundError if the model is missing and auto-download isn't possible.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    repo_id = os.getenv("MODEL_REPO", DEFAULT_REPO)
    filename = _select_filename()
    target = models_path / filename

    if target.exists():
        print(f"[bootstrap_model] Found model: {target.name}")
        return str(target.resolve())

    if not auto_download:
        raise FileNotFoundError(
            f"[bootstrap_model] Model not found at {target}. "
            f"Download it manually from https://huggingface.co/{repo_id} "
            f"and place it in '{models_path}'."
        )

    # Try to download from Hugging Face
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        raise FileNotFoundError(
            f"[bootstrap_model] Missing model and huggingface_hub is not installed.\n"
            f"Run: pip install huggingface_hub\n"
            f"Then re-run the app to auto-download '{filename}'."
        )

    print(f"[bootstrap_model] ⬇️ First run: downloading '{filename}' from {repo_id} …")
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(models_path),
            local_dir_use_symlinks=False,  # real file, not symlink
            resume_download=True,
        )
        print(f"[bootstrap_model] ✅ Downloaded: {Path(local).name}")
        return str(Path(local).resolve())
    except Exception as e:
        raise RuntimeError(
            "[bootstrap_model] Could not download the model. "
            "Please ensure you have an internet connection for the first run, "
            f"or download manually from https://huggingface.co/{repo_id} and place '{filename}' in '{models_path}'.\n"
            f"Details: {e}"
        )

# Backwards-compat alias
def bootstrap_model():
    return ensure_model()

if __name__ == "__main__":
    print(ensure_model())
