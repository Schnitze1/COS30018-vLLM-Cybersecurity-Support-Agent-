import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch.optim as optim
import requests
import os
from src.environment import LogSimulator
from src.agent import create_agent # <-- Import our TPU agent brain

# --- CONFIGURATION ---
VLLM_BRAIN_URL = "https://coaly-uxorious-caren.ngrok-free.dev" 
TRAINING_STEPS = 20
LEARNING_RATE = 0.001

# --- Helper functions (from before) ---

def test_tpu_connection():
    """Confirms we can access and use the TPU."""
    try:
        device = xm.xla_device()
        t_cpu = torch.randn(2, 2)
        t_tpu = t_cpu.to(device)
        print(f"✅ [TPU Trainer]: Connection SUCCESSFUL. (Device: {device})")
        return device # Return the device for later use
    except Exception as e:
        print(f"❌ [TPU Trainer]: Connection FAILED: {e}")
        return None

def test_brain_connection(url):
    """Confirms we can reach the vLLM server."""
    try:
        response = requests.get(f"{url}/docs", timeout=5)
        if response.status_code == 200:
            print(f"✅ [vLLM Brain]: Connection SUCCESSFUL. (Server: {url})")
            return True
        else:
            print(f"❌ [vLLM Brain]: Connection FAILED. Status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        print(f"❌ [vLLM Brain]: Connection FAILED. Server not reachable.")
        return False

def analyze_log_with_brain(log_message):
    """Sends a log to the vLLM brain and returns its raw text prediction."""
    api_url = f"{VLLM_BRAIN_URL}/v1/chat/completions"
    prompt = (
        "Classify the following system log as either 'Normal' or 'Attack'.\n"
        f"Log: \"{log_message}\"\n"
        "Classification:"
    )
    body = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.0,
        "stop": ["\n"] 
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=body, timeout=30)
        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content'].strip()
            return answer
        else:
            return None
    except Exception as e:
        return None

# --- NEW HELPER FUNCTION ---

def process_brain_output(prediction_text, true_label):
    """
    Processes the vLLM's text and the true_label into tensors
    for our TPU agent to learn from.
    """
    # 1. Process vLLM's prediction into an 'input' tensor
    if prediction_text:
        if "normal" in prediction_text.lower():
            # vLLM thinks it's Normal
            vllm_opinion = torch.tensor([1.0, 0.0]) # [Normal, Attack]
        elif "attack" in prediction_text.lower():
            # vLLM thinks it's an Attack
            vllm_opinion = torch.tensor([0.0, 1.0]) # [Normal, Attack]
        else:
            # vLLM gave a confusing answer, show uncertainty
            vllm_opinion = torch.tensor([0.5, 0.5])
    else:
        # vLLM failed, show uncertainty
        vllm_opinion = torch.tensor([0.5, 0.5])
        
    # 2. Process the 'ground truth' into a 'label' tensor
    # Note: CrossEntropyLoss expects a single class index, not a one-hot vector
    true_label_tensor = torch.tensor(true_label, dtype=torch.long) # 0 for Normal, 1 for Attack
    
    return vllm_opinion, true_label_tensor

# --- NEW MAIN FUNCTION ---

def main():
    print("--- Initializing Cybersecurity Agent System ---")
    
    # 1. Test connections and get TPU device
    tpu_device = test_tpu_connection()
    if not tpu_device or not test_brain_connection(VLLM_BRAIN_URL):
        print("🛑 System OFFLINE. Please check error messages above.")
        return

    # 2. Create our Agent, Optimizer, and Loss Function
    agent_brain = create_agent() # This already moves the agent to the TPU
    optimizer = optim.Adam(agent_brain.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # 3. Create our Environment
    simulator = LogSimulator()

    print(f"🚀 System is ONLINE. Starting {TRAINING_STEPS}-step training loop...")
    print("-----------------------------------------------")
    
    # 4. Run the Training Loop
    for step in range(TRAINING_STEPS):
        # --- GET DATA (ENVIRONMENT) ---
        log, true_label = simulator.get_next_log() # true_label is 0 or 1
        
        # --- GET INSTINCT (vLLM BRAIN) ---
        vllm_prediction_text = analyze_log_with_brain(log)
        
        # --- PREPARE DATA FOR TPU AGENT ---
        # vllm_opinion is our input, true_label_tensor is what we're learning
        vllm_opinion, true_label_tensor = process_brain_output(vllm_prediction_text, true_label)
        
        # Move tensors to the TPU
        # We use .unsqueeze(0) to create a "batch" of 1
        input_tensor = vllm_opinion.to(tpu_device).unsqueeze(0)
        label_tensor = true_label_tensor.to(tpu_device).unsqueeze(0) # Also needs to be in a batch

        # --- TRAIN THE AGENT (TPU BRAIN) ---
        
        # 1. Forward pass: Get the agent's decision
        # agent_brain takes the vLLM's opinion and makes its OWN decision
        agent_logits = agent_brain(input_tensor)
        
        # 2. Calculate Loss: How wrong was the agent?
        loss = loss_fn(agent_logits, label_tensor)
        
        # 3. Backward pass (Training)
        optimizer.zero_grad() # Reset gradients
        loss.backward()       # Calculate gradients
        xm.optimizer_step(optimizer) # Update the agent's weights on the TPU
        
        # --- LOGGING ---
        # Get the agent's final choice by seeing which logit was higher
        agent_choice = torch.argmax(agent_logits, dim=1).item() # 0 or 1
        agent_choice_text = "Attack" if agent_choice == 1 else "Normal"
        true_label_text = "Attack" if true_label == 1 else "Normal"

        print(f"--- Step {step+1}/{TRAINING_STEPS} ---")
        print(f"  Log:       '{log}'")
        print(f"  vLLM says: '{vllm_prediction_text}'")
        print(f"  Agent says:  '{agent_choice_text}' (Loss: {loss.item():.4f})")
        print(f"  Truth is:    '{true_label_text}'")
        
        if agent_choice == true_label:
            print("  ✅ Result: Agent was CORRECT!")
        else:
            print("  ❌ Result: Agent was INCORRECT.")

if __name__ == "__main__":
    main()
