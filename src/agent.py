import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

# This is our agent's 'brain' - a simple PyTorch Neural Network.
# We don't need the torchforge/monarch wrappers.
class CyberAgent(nn.Module):
    """
    It will learn to take the vLLM's prediction (and other data)
    and decide the *final* classification.
    """
    def __init__(self, input_size, output_size):
        super(CyberAgent, self).__init__()
        # A simple 3-layer network
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
    # Our agent will take 2 inputs:
    # 1. The vLLM's 'Normal' prediction confidence
    # 2. The vLLM's 'Attack' prediction confidence
    input_dim = 2 
    
    # It will have 2 outputs (logits):
    # 1. The agent's 'Normal' score
    # 2. The agent's 'Attack' score
    output_dim = 2

    # Create the agent
    agent_brain = CyberAgent(input_size=input_dim, output_size=output_dim)
    
    # Get the TPU device
    device = xm.xla_device()
    
    # Move the agent to the TPU!
    agent_brain.to(device)
    
    print(f"✅ [Agent Brain]: Successfully created agent on {device}")
    return agent_brain

# --- This is just for testing the file directly ---
if __name__ == "__main__":
    print("--- Testing Agent Brain Creation ---")
    try:
        agent = create_agent()
        
        # Create a dummy input (e.g., vLLM is 90% sure it's 'Normal')
        dummy_input = torch.tensor([0.9, 0.1], dtype=torch.float32)
        
        # Send the dummy input to the TPU
        dummy_input_tpu = dummy_input.to(xm.xla_device())
        
        # Get the agent's decision
        # Note: We must wrap it in a 'batch' (the [dummy_input_tpu])
        decision = agent(dummy_input_tpu.unsqueeze(0))
        
        print(f"✅ [Agent Brain]: Successfully processed input.")
        print(f"    - Input: {dummy_input.tolist()}")
        print(f"    - Output (Logits): {decision.tolist()}")
        
    except Exception as e:
        print(f"❌ [Agent Brain]: FAILED to create or run agent: {e}")
