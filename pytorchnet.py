import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchNet(nn.Module):

    def __init__(self, convweights, poolings=False):
        super(PytorchNet, self).__init__()
        self.conv_ops_ = []
        for w in convweights:
            #nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width
            conv_op = nn.Conv2d(w[0], w[1], kernel_size=w[2], stride=1, padding=0)
            self.conv_ops_.append(conv_op)
        self.poolings_ = poolings
        
        
    def forward(self, x):
        x = torch.from_numpy(x)
        layer = 0
        t0 = time.time()
        for conv_op in self.conv_ops_:
            x =  conv_op(x)
            if self.poolings_:
                x = F.max_pool2d(x, self.poolings_[layer])
                layer += 1
        
        t1 = time.time()
        return (x, (t1-t0))