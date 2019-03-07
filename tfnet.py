import numpy as np
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
        for w in self.conv_filters_:
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
            if self.poolings_:
                window = (1, 1, self.poolings_[layer], self.poolings_[layer])
                x = tf.nn.max_pool(x, window, window, padding='VALID', data_format='NCHW')
                layer += 1
        return x
        
    
    def forward(self, x_):
        x = self.forward_(x_)
        with tf.Session() as sess:
            # Evaluate the tensor `x`
            x_out = sess.run(x) # session returns numpy array
        return x_out
      