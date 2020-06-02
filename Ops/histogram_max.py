import tensorflow as tf
import keras
import os

global_layers_list = {}  # same as for losses

histogram_max_sample_module = tf.load_op_library(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'libHistogramMaxSample.so'
    )
)

class HistogramMax(keras.engine.Layer):
    
    @tf.RegisterGradient("HistogramMaxSample")
    def _histogram_max_sample_grad(op, grad):
        grad_hists, grad_randoms = histogram_max_sample_module.histogram_max_sample_grad(
            op.inputs[0],
            op.inputs[1],
            grad,
            op.get_attr('bias')
        )
        return [grad_hists, grad_randoms]
    
    def __init__(self,
        bias = 1e-4,
        **kwargs
    ):
        self.bias = bias
        self.supports_masking = False
        super(HistogramMax, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        if type(input_shape)==type(list()):
            assert len(input_shape)==1 or len(input_shape)==2
            if len(input_shape)==1:
                hists_shape = input_shape[0]
                randoms_shape = None
            elif len(input_shape)==2:
                hists_shape,randoms_shape = input_shape
        else:
            hists_shape = input_shape
            randoms_shape = None
  
        assert len(hists_shape)==3
        if randoms_shape!=None:
            assert len(randoms_shape)==2
        
        super(HistogramMax, self).build(input_shape)

    def call(self, inputs, mask=None):
        if type(inputs)==type(list()):
            if len(inputs)==1:
                hists = inputs[0]
                randoms = None
            elif len(inputs)==2:
                hists,randoms = inputs
        else:
            hists = inputs
            randoms = None
    
    
        if randoms==None:
            randoms = tf.random.uniform(
                (tf.shape(hists)[0],tf.shape(hists)[2]),
                0,
                1
            )
       
        return tf.case([(
            tf.keras.backend.learning_phase(),
            lambda: histogram_max_sample_module.histogram_max_sample(hists,randoms,bias=self.bias)
        )],default=lambda: tf.cast(tf.argmax(hists,axis=1),dtype=tf.float32))
        
        #return histogram_max_sample_module.histogram_max_sample(hists,randoms,bias=self.bias)
        
    def compute_output_shape(self,input_shape):
        if type(input_shape)==type(list()):
            if len(input_shape)==1:
                hists_shape = input_shape[0]
            elif len(input_shape)==2:
                hists_shape,randoms_shape = input_shape
        else:
            hists_shape = input_shape
        return (hists_shape[0],hists_shape[2])


global_layers_list['HistogramMax'] = HistogramMax

