# COS30018-vLLM-Cybersecurity-Support-Agent-

# Setup the  Cloud VM from the shell 
# 1. Set your Project ID
export PROJECT_ID="cos30018-vllm-agent"
export TPU_NAME="cos30018-agent-tpu"
export ZONE="us-central1-a"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE

# 2. Start Container (on the VM)
docker run -it --rm --privileged --net host \
  us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11 \
  bash

  # Hugging Box Setup:

  docker run -it --rm --gpus all -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=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 \
  vllm/vllm-openai:latest \
  --model google/gemma-2-2b-it