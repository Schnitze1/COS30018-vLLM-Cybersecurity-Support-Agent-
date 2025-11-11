# COS30018-vLLM-Cybersecurity-Support-Agent-

# Setup the  Cloud VM from the shell 
# 1. Set your Project ID
export PROJECT_ID="cos30018-vllm-agent"

# 2. Tell gcloud to use this project
gcloud config set project $PROJECT_ID

# 3. Set all the GPU VM variables for the US zone
export GPU_VM_NAME="cos30018-vllm-server"
export ZONE="us-central1-a"
export MACHINE_TYPE="g2-standard-8"
export GPU_TYPE="nvidia-l4"
export IMAGE_PROJECT="deeplearning-platform-release"
export IMAGE_FAMILY="common-cu128-ubuntu-2204-nvidia-570"

# 4. Run the create command
# (The firewall rule "vllm-api-allow" is already created, so we don't need to do that again)
gcloud compute instances create $GPU_VM_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --accelerator="type=$GPU_TYPE,count=1" \
  --image-family=$IMAGE_FAMILY \
  --image-project=$IMAGE_PROJECT \
  --tags=vllm-server \
  --maintenance-policy=TERMINATE