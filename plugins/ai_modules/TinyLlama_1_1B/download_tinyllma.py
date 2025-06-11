import os
from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your target directory
target_dir = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\plugins\ai_modules\TinyLlama_1_1B"

# Create directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

print("Starting download of TinyLlama-1.1B-Chat-v1.0...")
print(f"Target directory: {target_dir}")

try:
    # Download the model files
    print("📥 Downloading model files... This may take a few minutes.")
    downloaded_path = snapshot_download(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        local_dir=target_dir,
        local_dir_use_symlinks=False,  # Download actual files, not symlinks
        resume_download=True  # Resume if interrupted
    )

    print(f"✅ Model successfully downloaded to: {downloaded_path}")

    # List downloaded files
    print("\n📁 Downloaded files:")
    for file in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  - {file} ({size:.1f} MB)")

    # Basic verification
    config_path = os.path.join(target_dir, "config.json")
    if os.path.exists(config_path):
        print("\n✅ Download verification successful!")
        print("🎉 TinyLlama model is ready to use!")
    else:
        print("\n⚠️ Warning: config.json not found. Download may be incomplete.")

except Exception as e:
    print(f"❌ Error during download: {str(e)}")
    print("Possible solutions:")
    print("1. Check your internet connection")
    print("2. Make sure you have enough disk space (need ~2.5GB)")
    print("3. Try running: pip install --upgrade huggingface_hub")
    print("4. If behind a firewall, you may need to configure proxy settings")