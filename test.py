import torch
from models.models_Prithvi import prithvi

sd = torch.load('/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/Prithvi_100M.pt', map_location='cpu')
model = prithvi(checkpoint=sd, freeze_body=True, classifier=False)
x = model(torch.randn((4, 6, 224, 224)))
print()