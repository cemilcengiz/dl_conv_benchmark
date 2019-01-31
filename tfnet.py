import numpy as np

import time
import tensorflow as tf


class TFNet():
    # Filters come as a 4D Tensor of  [out_channels, in_channels, fil_height, fil_width]
    def __init__(self, convfilters, poolings=False):
        # TF conv.2d takes filter as a 4D Tensor of  [fil_height, fil_width, in_channels, out_channels] 
        self.conv_filters_ = []
        for w_array in convfilters:
            w_array = np.transpose(np.copy(w_array), (2,3,1,0)) # convert to TF filter format            
            self.conv_filters_.append(w_array)
                
        self.poolings_ = poolings
      
        
    def forward_(self, x):
        # input x is in form of [nSamples x nChannels x Height x Width] i.e. "NCHW"
        layer = 0
        t0 = time.time()
        for w in self.conv_filters_:
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
            if self.poolings_:
                window = (1, 1, self.poolings_[layer], self.poolings_[layer])
                x = tf.nn.max_pool(x, window, window, padding='VALID', data_format='NCHW')
                layer += 1
                
        t1 = time.time()
        return (x, (t1-t0))
        
    
    def forward(self, x):
        x, runTime = self.forward_(x)
        with tf.Session() as sess:  
            x = sess.run(x) # session returns numpy array
            return (x, runTime)        