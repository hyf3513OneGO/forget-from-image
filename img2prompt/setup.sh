echo "Starting setup"

echo "Installing BLIP"
pip install -e git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip

echo "Cloning clip interrogator"
git clone -b open-clip https://github.com/pharmapsychotic/clip-interrogator.git

echo "Downloading cache files"
CACHE_URLS=(
    "https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_artists.safetensors"
    "https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_flavors.safetensors"
    "https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_mediums.safetensors"
    "https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_movements.safetensors"
    "https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_trendings.safetensors"
)
mkdir -p cache
for url in "${CACHE_URLS[@]}"; do
    wget "$url" -P cache
done

echo "Setup completed"