import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch.optim as optim
import requests
import os
from src.environment import LogSimulator
from src.agent import create_agent # Our agent brain

# --- CONFIGURATION ---
VLLM_BRAIN_URL = "https://coaly-uxorious-caren.ngrok-free.dev" 
TRAINING_STEPS = 20
LEARNING_RATE = 0.001

# --- Helper functions (no changes here) ---

def test_tpu_connection():
    try:
        device = xm.xla_device()
        t_cpu = torch.randn(2, 2)
        t_tpu = t_cpu.to(device)
        print(f"✅ [TPU Trainer]: Connection SUCCESSFUL. (Device: {device})")
        return device
    except Exception as e:
        print(f"❌ [TPU Trainer]: Connection FAILED: {e}")
        return None

def test_brain_connection(url):
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

# --- UPDATED HELPER FUNCTION ---

def process_inputs(vllm_prediction_text, log_message, true_label):
    """
    Processes all our data into tensors for the TPU agent.
    """
    # 1. Process vLLM's prediction
    if vllm_prediction_text:
        if "normal" in vllm_prediction_text.lower():
            vllm_normal = 1.0
            vllm_attack = 0.0
        elif "attack" in vllm_prediction_text.lower():
            vllm_normal = 0.0
            vllm_attack = 1.0
        else:
            vllm_normal = 0.5
            vllm_attack = 0.5
    else:
        vllm_normal = 0.5
        vllm_attack = 0.5
        
    # 2. Process our NEW feature: log length
    # We normalize it (divide by 100) so it's on a similar scale
    # to our 0.0-1.0 features.
    log_length_feature = len(log_message) / 100.0 
    
    # 3. Create the final 3-feature input tensor
    input_tensor = torch.tensor([
        vllm_normal,
        vllm_attack,
        log_length_feature
    ])
    
    # 4. Process the 'ground truth' into a 'label' tensor
    true_label_tensor = torch.tensor(true_label, dtype=torch.long) # 0 for Normal, 1 for Attack
    
    return input_tensor, true_label_tensor

# --- UPDATED MAIN FUNCTION ---

def main():
    print("--- Initializing Cybersecurity Agent System ---")
    
    # 1. Test connections and get TPU device
    tpu_device = test_tpu_connection()
    if not tpu_device or not test_brain_connection(VLLM_BRAIN_URL):
        print("🛑 System OFFLINE. Please check error messages above.")
        return

    # 2. Create our Agent, Optimizer, and Loss Function
    # This now creates the 3-input agent
    agent_brain = create_agent() 
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
        # This now returns our 3-feature tensor
        input_data, label_data = process_inputs(vllm_prediction_text, log, true_label)
        
        # Move tensors to the TPU
        input_tensor = input_data.to(tpu_device).unsqueeze(0) # Batch of 1
        label_tensor = label_data.to(tpu_device).unsqueeze(0) # Batch of 1

        # --- TRAIN THE AGENT (TPU BRAIN) ---
        
        # 1. Forward pass: Get the agent's decision
        agent_logits = agent_brain(input_tensor)
        
        # 2. Calculate Loss: How wrong was the agent?
        loss = loss_fn(agent_logits, label_tensor)
        
        # 3. Backward pass (Training)
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)
        
        # --- LOGGING ---
        agent_choice = torch.argmax(agent_logits, dim=1).item()
        agent_choice_text = "Attack" if agent_choice == 1 else "Normal"
        true_label_text = "Attack" if true_label == 1 else "Normal"

        print(f"--- Step {step+1}/{TRAINING_STEPS} ---")
        print(f"  Log:       '{log}' (Length: {len(log)})")
        print(f"  vLLM says: '{vllm_prediction_text}'")
        print(f"  Agent says:  '{agent_choice_text}' (Loss: {loss.item():.4f})")
        print(f"  Truth is:    '{true_label_text}'")
        
        if agent_choice == true_label:
            print("  ✅ Result: Agent was CORRECT!")
        else:
            print("  ❌ Result: Agent was INCORRECT.")

if __name__ == "__main__":
    main()
