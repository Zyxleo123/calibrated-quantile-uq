import pickle as pkl
import torch
import torch.nn as nn
import sys

# --- PRE-LOADING PATCH: Define the missing classes ---

# Step 1: We need the 'EnhancedMLP' class because the 'VanillaModel' contains it.
# I've copied its definition here from `utils/q_model_ens.py` for completeness.
# If you have the `utils` folder, you can just do:
# from utils.q_model_ens import EnhancedMLP, QModelEns
from utils.q_model_ens import EnhancedMLP, QModelEns

# --- MAIN LOADING LOGIC ---

# Step 4: Explicitly add our dummy class to the `__main__` module namespace.
# This is the most robust way to ensure pickle finds it.
import __main__
class VanillaModel(nn.Module):
    def __init__(self, nfeatures):
        super().__init__()
        self.net = None
    def forward(self, x):
        return self.net(x)
__main__.VanillaModel = VanillaModel
__main__.EnhancedMLP = EnhancedMLP
__main__.QModelEns = QModelEns # Also add any other custom classes
file_path = '/home/scratch/yixiz/results/full/elevator/nl-4_hs-128/elevator_losscalipso_ens1_bootFalse_resTrue_lnFalse_bnFalse_dr0.0_lr0.001_bs64_nl4_hs128_1_models.pkl'

print(f"Attempting to load model from: {file_path}")

with open(file_path, 'rb') as f:
    # It's crucial to load on the CPU first to avoid GPU memory issues
    # if the loading environment is different from the saving one.
    loaded_models = pkl.load(f)

print("\nSuccessfully loaded the model!")
print(f"Type of loaded object: {type(loaded_models)}")
if isinstance(loaded_models, list):
    print(f"Number of models in list: {len(loaded_models)}")
    # Inspect the first model in the list
    first_model = loaded_models[0]
    import torch
    input = torch.randn(1, first_model.X.shape[1], device='cuda:0')  # Assuming the model has an attribute 'X' for input features
    output = first_model(input)
    # print(f"Output from the first model: {output}")
