#!/usr/bin/env bash
# Run ONCE, on the GCP VM itself (after `gcloud compute ssh` in), from inside a clone of this
# repo. Installs Docker + the NVIDIA Container Toolkit, then builds the rl-wbc-g1-stairs image.
#
# Prerequisites this script does NOT automate (need your own credentials):
#   - An NGC account (free) and an NGC API key: https://ngc.nvidia.com
#   - `git clone <your fork of this repo>` already done on the VM, and you're `cd`'d into it.
set -euo pipefail

if ! command -v docker &>/dev/null; then
  echo "Installing Docker Engine..."
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "${USER}"
  echo "Added ${USER} to the docker group -- log out and back in (or re-SSH) before continuing."
fi

if ! command -v nvidia-ctk &>/dev/null; then
  echo "Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

echo
echo "Checking the GPU is visible to the driver (not yet inside a container)..."
nvidia-smi || { echo "nvidia-smi failed -- driver install may still be finishing; reboot the VM and retry."; exit 1; }

echo
echo "Log in to NVIDIA's container registry (paste your NGC API key as the password, username='\$oauthtoken'):"
docker login nvcr.io

echo
echo "Building the rl-wbc-g1-stairs image..."
docker build -f docker/Dockerfile -t rl-wbc-g1-stairs .

echo
echo "Sanity check: GPU visible inside a container?"
docker run --rm --gpus all rl-wbc-g1-stairs nvidia-smi

echo
echo "Setup complete. Next: scripts/train_amp.sh"
