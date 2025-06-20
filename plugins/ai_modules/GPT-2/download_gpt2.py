from huggingface_hub import snapshot_download

# Download the entire GPT-2 model
snapshot_download(
    repo_id="openai-community/gpt2",
    local_dir=r"C:\Users\10169062\Desktop\AU_IndusMaintdb\plugins\ai_modules\GPT-2",
    local_dir_use_symlinks=False
)