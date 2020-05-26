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
            grad
        )
        return [grad_hists, grad_randoms]
    
    def __init__(self,
        **kwargs
    ):
        super(HistogramMax, self).__init__(**kwargs)
        self.supports_masking = False
        
    def build(self, input_shape):
        hists_shape = input_shape
        assert len(hists_shape)==3
        
        
        super(HistogramMax, self).build(input_shape)

    def call(self, inputs, mask=None):
        hists = inputs
        randoms = tf.random.uniform(
            (tf.shape(hists)[0],tf.shape(hists)[2]),
            0,
            1
        )
        return tf.case([(
            tf.keras.backend.learning_phase(),
            lambda: histogram_max_sample_module.histogram_max_sample(hists,randoms)
        )],default=lambda: tf.cast(tf.argmax(hists,axis=1),dtype=tf.float32))
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[2])


global_layers_list['HistogramMax'] = HistogramMax

