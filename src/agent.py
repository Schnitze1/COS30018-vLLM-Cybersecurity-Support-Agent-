import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

# This is our agent's 'brain' - a simple PyTorch Neural Network.
class CyberAgent(nn.Module):
    """
    It will learn to take the vLLM's prediction (and other data)
    and decide the *final* classification.
    """
    def __init__(self, input_size, output_size):
        super(CyberAgent, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

def create_agent():
    """
    A helper function to build our agent and move it to the TPU.
    """
    # --- THIS IS THE CHANGE ---
    # Our agent will now take 3 inputs:
    # 1. The vLLM's 'Normal' prediction confidence
    # 2. The vLLM's 'Attack' prediction confidence
    # 3. The length of the log message
    input_dim = 3
    
    # It still has 2 outputs (logits):
    # 1. The agent's 'Normal' score
    # 2. The agent's 'Attack' score
    output_dim = 2

    # Create the agent
    agent_brain = CyberAgent(input_size=input_dim, output_size=output_dim)
    
    # Get the TPU device
    device = xm.xla_device()
    
    # Move the agent to the TPU!
    agent_brain.to(device)
    
    print(f"✅ [Agent Brain]: Successfully created agent on {device} (Input size: {input_dim})")
    return agent_brain

# --- This is just for testing the file directly ---
if __name__ == "__main__":
    print("--- Testing Agent Brain Creation ---")
    try:
        agent = create_agent()
        
        # Create a dummy input (vLLM: 90% Normal, Log Length: 75)
        dummy_input = torch.tensor([0.9, 0.1, 75.0], dtype=torch.float32)
        
        # Send the dummy input to the TPU
        dummy_input_tpu = dummy_input.to(xm.xla_device())
        
        # Get the agent's decision
        decision = agent(dummy_input_tpu.unsqueeze(0))
        
        print(f"✅ [Agent Brain]: Successfully processed 3-feature input.")
        print(f"    - Input: {dummy_input.tolist()}")
        print(f"    - Output (Logits): {decision.tolist()}")
        
    except Exception as e:
        print(f"❌ [Agent Brain]: FAILED to create or run agent: {e}")
