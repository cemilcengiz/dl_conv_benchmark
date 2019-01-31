import numpy as np

import time
import tensorflow as tf

class TFNet():
    # Filters come as a 4D Tensor of  [out_channels, in_channels, fil_height, fil_width]
    def __init__(self, convfilters, poolings=False):
        # TF conv.2d takes filter as a 4D Tensor of  [fil_height, fil_width, in_channels, out_channels] 
        self.conv_filters_ = []
        for w_array in convfilters:
            outCh, inCh, filHeight, filWidth = w_array.shape
            w_array = w_array.reshape(filHeight, filWidth, inCh, outCh) # convert to TF filter format
            w_array = np.copy(w_array)
            #w_tensor = torch.tensor(np.copy(w_array), dtype=torch.float32)
            self.conv_filters_.append(w_array)
                
        self.poolings_ = poolings

        
        
    def forward_(self, x):
         # TF conv.2d takes input a 4D Tensor of [nSamples x Height x Width x nChannels]
        inBS, inCh, inHeight, inWidth = x.shape
        x = x.reshape(inBS, inHeight, inWidth, inCh) # convert to default data format "NHWC"
        #x_tensor = tf.placeholder(tf.float32, shape=(x.shape))  

        layer = 0
        t0 = time.time()
        for w in self.conv_filters_:
            print("layer:", layer)
            print("size of x before:", x.shape)
            print("size of filter :", w.shape)

            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID')
            print("size of x after filter:", x.shape)
            if self.poolings_:
                window = (1, self.poolings_[layer], self.poolings_[layer], 1)
                x = tf.nn.max_pool(x, window, window, padding='VALID')
                print("size of x after pool:", x.shape)
                print(" ")
                layer += 1

        
        t1 = time.time()
        return (x, (t1-t0))
        
    
    def forward(self, x):
        x, runTime = self.forward_(x)
        with tf.Session() as sess:  
            x = sess.run(x) # session returns numpy array
            #return (x, runTime)
            return (x, runTime, self.conv_filters_)
        