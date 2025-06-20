import os
from huggingface_hub import snapshot_download

def download_tinyllama_model(target_dir):
    """
    Download TinyLlama-1.1B-Chat-v1.0 model into the specified directory if not already present.

    Args:
        target_dir (str): The full path where the model should be stored.
    """
    os.makedirs(target_dir, exist_ok=True)

    print(" Checking for existing TinyLlama model...")
    config_path = os.path.join(target_dir, "config.json")
    model_path = os.path.join(target_dir, "model.safetensors")

    if os.path.exists(config_path) and os.path.exists(model_path):
        print(" Model already downloaded. Skipping download.")
        return

    print("📥 Downloading TinyLlama-1.1B-Chat-v1.0...")
    print(f"Target directory: {target_dir}")

    try:
        downloaded_path = snapshot_download(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f" Model successfully downloaded to: {downloaded_path}")

        # List files for verification
        print("\n Downloaded files:")
        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size:.1f} MB)")

        # Verify model config exists
        if os.path.exists(config_path):
            print("\n Download verification successful!")
            print(" TinyLlama model is ready to use!")
        else:
            print("\n Warning: config.json not found. Download may be incomplete.")

    except Exception as e:
        print(f" Error during download: {str(e)}")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space (~2.5GB)")
        print("3. Try running: pip install --upgrade huggingface_hub")
        print("4. If behind a firewall, configure proxy settings")

# 🔒 Safe entry point: only runs if this file is executed directly
if __name__ == "__main__":
    default_dir = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\plugins\ai_modules\TinyLlama_1_1B"
    download_tinyllama_model(default_dir)
