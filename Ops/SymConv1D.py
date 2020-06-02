import tensorflow as tf
import keras

global_layers_list = {}  # same as for losses

class SymConv1D(keras.engine.Layer):
    def __init__(self,
        filter_size,
        kernel_var_size,
        strides=1,
        padding='same',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        use_bias=True,
        **kwargs
    ):
        super(SymConv1D, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.kernel_var_size = kernel_var_size  
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        

    def build(self, input_shape):
        self.channel_size = input_shape[2]
        
                
        self.kernel_left = self.add_weight(
            shape=(self.kernel_var_size,self.channel_size,self.filter_size),
            initializer=self.kernel_initializer,
            name = 'kernel_left'
        )
        self.kernel_middle = self.add_weight(
            shape=(1,self.channel_size,self.filter_size),
            initializer=self.kernel_initializer,
            name = 'kernel_middle'
        )
        
        
        self.kernel_right = tf.reverse(self.kernel_left,axis=[0]) #reverse kernel axis
        self.kernel = tf.concat([
            self.kernel_left,
            self.kernel_middle,
            self.kernel_right
        ],axis=0)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filter_size,),
                initializer=self.bias_initializer,
                name = 'bias'
            )
        
        super(SymConv1D, self).build(input_shape) 
        
        
    def call(self, inputs):
        output = tf.nn.conv1d(
            inputs, 
            self.kernel, 
            stride=self.strides, 
            padding=self.padding.upper(),
            data_format='NWC', 
        )
        
        if self.use_bias:
            return output + self.bias
            
        return output
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filter_size)
        


global_layers_list['SymConv1D'] = SymConv1D

    
