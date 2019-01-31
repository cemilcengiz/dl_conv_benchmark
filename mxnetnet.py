import numpy as np
import time
import mxnet as mx
from mxnet import nd, autograd, gluon

class MXNetNet():
    # Filters come as a 4D Tensor of  [out_channels, in_channels, fil_height, fil_width]
    def __init__(self, convfilters, poolings=False):
        self.conv_filters_ = []
        for w_array in convfilters:
            # MXNet nd.Convolution takes filter as a 4D Tensor of  [out_channels, in_channels, fil_height, fil_width]
            w_tensor = mx.nd.array(np.copy(w_array))
            self.conv_filters_.append(w_tensor)
            
        self.poolings_ = poolings
        
        
    def forward(self, x):
        # MXNet nd.Convolution takes in a 4D Tensor of [nSamples x nChannels x Height x Width]
        x = mx.nd.array(x)
        
        layer = 0
        t0 = time.time()
        for w in self.conv_filters_:                       
            x = nd.Convolution(data=x, weight=w, kernel=w.shape[-2:], num_filter=w.shape[0], no_bias=True)
            if self.poolings_:
                window = (self.poolings_[layer], self.poolings_[layer])
                x = nd.Pooling(data=x, pool_type="max", kernel=window, stride=window)
                layer += 1
        
        t1 = time.time()
        return (x.asnumpy(), (t1-t0))