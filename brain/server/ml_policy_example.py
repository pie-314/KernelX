"""
Example: Integrating an AI Model into KernelX

This file demonstrates how to replace the ManualPolicy with a trained
machine learning model. Use this as a template when you're ready to deploy AI.
"""

from typing import Optional
from brain.models import Action, Observation
import numpy as np


class ExampleMLPolicy:
    """
    Example trained ML policy - replace with your actual model.
    
    This shows the interface your model needs to implement.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize with a trained model.
        
        Args:
            model_path: Path to your trained model (e.g., "model.pth")
        """
        self.model_path = model_path
        # Load your model here:
        # Example PyTorch:
        #   import torch
        #   self.model = torch.load(model_path)
        #   self.model.eval()
        
        if model_path:
            print(f"[ML Policy] Loaded model from {model_path}")
    
    def decide(self, obs: Observation) -> Action:
        """
        Make a scheduling decision.
        
        Args:
            obs: 24D kernel observation
            
        Returns:
            Action with 4 priority weights [-100, 100]
        """
        features = np.array(obs.features, dtype=np.float32)
        
        # Example: PyTorch inference
        # with torch.no_grad():
        #     tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        #     output = self.model(tensor)
        #     weights = output.squeeze().cpu().numpy().tolist()
        
        # For testing, return random valid weights
        weights = [float(np.random.uniform(-100, 100)) for _ in range(4)]
        return Action(weights=weights)


# How to use this template:
# ========================
#
# 1. Train your model (PyTorch example):
#
#    import torch
#    model = YourTrainedModel()
#    torch.save(model.state_dict(), "kernelx_policy.pth")
#
# 2. Create your policy class:
#
#    class MyTrainedPolicy(ExampleMLPolicy):
#        def __init__(self, model_path):
#            super().__init__(model_path)
#            # Your custom initialization
#        
#        def decide(self, obs):
#            # Your custom inference logic
#            pass
#
# 3. Update brain/server/kernelx_environment.py:
#
#    from .ml_policy_example import MyTrainedPolicy
#    
#    class KernelXEnvironment(Environment[...]):
#        def __init__(self):
#            # Replace:
#            #   self.policy = ManualPolicy()
#            # With:
#            self.policy = MyTrainedPolicy("kernelx_policy.pth")
#
# 4. Restart brain server:
#
#    python3 -m server.app
#
# Done! System uses your trained model for decisions.
