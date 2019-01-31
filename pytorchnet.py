import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchNet(nn.Module):
    # Filters come as a 4D Tensor of  [out_channels, in_channels, fil_height, fil_width]
    def __init__(self, convfilters, poolings=False):
        super(PytorchNet, self).__init__()
        self.conv_filters_ = []
        for w_array in convfilters:         
            # Pytorch F.conv2d takes filter as a 4D Tensor of  [out_channels, in_channels, fil_height, fil_width]
            w_tensor = torch.tensor(np.copy(w_array), dtype=torch.float32)
            self.conv_filters_.append(w_tensor)
            
        self.poolings_ = poolings
        
        
    def forward(self, x):
        # Pytorch nn.Conv2d takes in a 4D Tensor of [nSamples x nChannels x Height x Width]
        x = torch.tensor(x, dtype=torch.float32) # returns copy
        
        layer = 0
        t0 = time.time()
        for w in self.conv_filters_:                       
            x = F.conv2d(x, w, stride=1, padding=0)
            if self.poolings_:
                x = F.max_pool2d(x, self.poolings_[layer])
                layer += 1
        
        t1 = time.time()
        return (x.numpy(), (t1-t0))
