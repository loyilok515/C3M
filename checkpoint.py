import torch

checkpoint = torch.load('log_OCTA_30/model_best.pth.tar', map_location='cpu')
print("Epoch saved in checkpoint:", checkpoint['epoch'])
print("Best accuracy in checkpoint:", checkpoint['precs'])