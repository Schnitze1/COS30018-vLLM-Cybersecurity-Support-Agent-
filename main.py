
import torch
import torch_xla.core.xla_model as xm
import monarch
import torchforge
import requests
import os


# Config
# This is the public URL from the ngrok tunnel
VLLM_BRAIN_URL = "https://YOUR-NGROK-STRING.ngrok.app" 


def test_tpu_connection():
    """Confirms access and use the TPU."""
    try:
        device = xm.xla_device()
        t_cpu = torch.randn(2, 2)
        t_tpu = t_cpu.to(device)

        print(f" [TPU Trainer]: Connection SUCCESSFUL.")
        print(f"    - Device: {device}")
        print(f"    - Tensor on TPU: {t_tpu.device}")
        return True
    except Exception as e:
        print(f" [TPU Trainer]: Connection FAILED: {e}")
        return False

def test_brain_connection(url):
    """Confirms we can reach the vLLM server."""
    try:
        # ping the /docs endpoint to see if it's alive
        response = requests.get(f"{url}/docs", timeout=5)

        if response.status_code == 200:
            print(f" [vLLM Brain]: Connection SUCCESSFUL.")
            print(f"    - Reached server at: {url}")
            return True
        else:
            print(f" [vLLM Brain]: Connection FAILED. Status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        print(f" [vLLM Brain]: Connection FAILED. Server not reachable at {url}.")
        print("    - Is your vLLM server running on your laptop?")
        print("    - Is your ngrok tunnel running?")
        return False

def main():
    """Main entry point for the agent."""
    print("--- Initializing Cybersecurity Agent System ---")

    # 1. Test the Trainer's TPU connection
    tpu_ok = test_tpu_connection()

    # 2. Test the Brain's vLLM connection
    brain_ok = test_brain_connection(VLLM_BRAIN_URL)

    if tpu_ok and brain_ok:
        print(" System is ONLINE. Ready to train.")
    else:
        print(" System OFFLINE. Please check error messages.")

if __name__ == "__main__":
    main()
