import torch
import numpy as np
import buteo as beo
from collections import OrderedDict
from utils.load_data import preprocess_image_prithvi, sentinelNormalize

class InferenceScript():
    def __init__(self, model):
       
        super(InferenceScript, self).__init__()
        self.model = model
        self.model.eval()
    
    def preprocess(self, x):
        x_norm = np.empty_like(x, dtype=np.float32)
        np.divide(x, 10000.0, out=x_norm)
        x_norm = beo.channel_last_to_first(x_norm)
        return torch.from_numpy(x_norm)
    
    def predict(self, x):
        x = self.preprocess(x)
        if len(x.shape) == 3:
           x = torch.unsqueeze(x, dim=0)
        x = self.model(x)
        return x.detach().numpy()
    

class PrithviInference(InferenceScript):
    
    def preprocess(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        # HLS bands: 0-B02, 1-B03, 2-B04, 4-B05, 5-B06, 6-B07,
        x = x[:, :, (0, 1, 2, 4, 5, 6)] # throw away unused bands
        x_norm = preprocess_image_prithvi(x)
        x_norm = beo.channel_last_to_first(x_norm)
        return torch.from_numpy(x_norm)
    

class SatMAEInference(InferenceScript):
    
    def preprocess(self, x):
        x_norm = sentinelNormalize(x)
        y = y.astype(np.float32, copy=False)

        x_norm = x_norm[16:-16, 16:-16, :]
        if len(y.shape) > 2:
            y = y[16:-16, 16:-16, :]

        x_norm = beo.channel_last_to_first(x_norm)
        return torch.from_numpy(x_norm)


