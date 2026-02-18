#!/bin/bash
set -e

###############################################
# Globals
###############################################
REPO_DIR="$(pwd)"              # Install where the repo was cloned
VENV_DIR="$REPO_DIR/.venv"
AIRLLM_DIR="$REPO_DIR/airllm"

PACKAGES=("git" "python3" "python3-dev" "python3-pip" "python3-venv" "wget" "curl")
TO_INSTALL=()
KEYRING_URL=""
CUDA_VERSION="12-6"
PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu126"

###############################################
# Detect WSL
###############################################
is_wsl() {
    grep -qi "microsoft" /proc/version || [[ -n "$WSL_DISTRO_NAME" ]]
}

###############################################
# Detect NVIDIA GPU
###############################################
has_nvidia_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi &>/dev/null && return 0
    fi
    return 1
}

###############################################
# Distro detection
###############################################
get_distro_settings() {
    if is_wsl; then
        echo "Detected WSL environment"
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
        return 0
    fi

    if [ -f /etc/os-release ]; then
        . /etc/os-release
    else
        echo "Error: Cannot detect OS"
        return 1
    fi

    case "$ID" in
        ubuntu)
            KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        debian)
            KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/debian${VERSION_ID}/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        *)
            echo "Unsupported distribution: $ID"
            return 1
            ;;
    esac
}

###############################################
# Install base packages
###############################################
install_base_packages() {
    for pkg in "${PACKAGES[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
            TO_INSTALL+=("$pkg")
        fi
    done

    if [ ${#TO_INSTALL[@]} -gt 0 ]; then
        echo "Installing packages: ${TO_INSTALL[*]}"
        sudo apt-get update
        sudo apt-get install -y "${TO_INSTALL[@]}"
    fi
}

###############################################
# Install CUDA Toolkit 12.6
###############################################
install_cuda() {
    if ! dpkg-query -W -f='${Status}' "cuda-keyring" 2>/dev/null | grep -q "install ok installed"; then
        echo "Installing CUDA Toolkit ${CUDA_VERSION}..."

        # Remove the old key
        sudo apt-key del 7fa2af80

        wget -q "$KEYRING_URL" -O cuda-keyring.deb
        sudo dpkg -i cuda-keyring.deb
        rm cuda-keyring.deb

        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-${CUDA_VERSION}

        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

        echo "CUDA Toolkit installed."
    fi
}

install_cuda() {
    # Install keyring if missing
    if ! dpkg-query -W -f='${Status}' "cuda-keyring" 2>/dev/null | grep -q "install ok installed"; then
        echo "Installing CUDA keyring..."

        wget -q "$KEYRING_URL" -O cuda-keyring.deb
        sudo dpkg -i cuda-keyring.deb
        rm cuda-keyring.deb
        sudo apt-get update
    else
        echo "CUDA keyring already installed — skipping."
    fi

    # Install CUDA toolkit
    if ! dpkg-query -W -f='${Status}' "cuda-toolkit-${CUDA_VERSION}" 2>/dev/null | grep -q "install ok installed"; then
        echo "Installing cuda-toolkit-${CUDA_VERSION}..."
        sudo apt-get install -y cuda-toolkit-${CUDA_VERSION}
    else
        echo "cuda-toolkit-${CUDA_VERSION} already installed — skipping."
    fi

    # Add PATH only for native Linux
    if ! is_wsl; then
        if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        fi
    fi

    echo "CUDA installation complete."
}



###############################################
# Install PyTorch (CUDA 12.6 wheel)
###############################################
install_pytorch() {
    echo "Installing PyTorch (CUDA 12.6 wheel)..."
    pip install torch --index-url "$PYTORCH_CUDA_INDEX"
}

###############################################
# Install AirLLM + dependencies
###############################################
install_airllm() {
    echo "Installing AirLLM + dependencies..."

    if [ ! -d "$AIRLLM_DIR" ]; then
        git submodule add https://github.com/lyogavin/airllm.git "$AIRLLM_DIR" || true
    else
        git submodule update --init --recursive
    fi

    pip install -r "$REPO_DIR/requirements.txt"
    pip install -e "$AIRLLM_DIR"
}

###############################################
# MAIN
###############################################
echo "Installing into repo directory: $REPO_DIR"

if ! get_distro_settings; then
    exit 1
fi

echo "Checking for NVIDIA GPU..."
if ! has_nvidia_gpu; then
    echo "ERROR: No NVIDIA GPU detected."
    exit 1
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name --format=csv,noheader

install_base_packages
install_cuda

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

install_pytorch
install_airllm

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
