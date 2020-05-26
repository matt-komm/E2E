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
    '''
    @tf.RegisterGradient("KDEHistogram")
    def _sub_grad(op, grad):
        grad_values, grad_weights = kde_histogram_module.kde_histogram_grad(
            op.inputs[0],
            op.inputs[1],
            grad,
            nbins=op.get_attr("nbins"),
            start=op.get_attr("start"),
            end=op.get_attr("end"),
            kernel=op.get_attr("kernel"),
            bandwidth_grad=op.get_attr("bandwidth_grad"),
            add_overflow=op.get_attr("add_overflow"),
        )
        return [grad_values, grad_weights]
    '''
    def __init__(self,
        **kwargs
    ):
        super(HistogramMax, self).__init__(**kwargs)
        self.supports_masking = False
        
    def build(self, input_shape):
        assert len(input_shape)==2
        hists_shape,randoms_shape = input_shape
        assert len(hists_shape)==3
        assert len(randoms_shape)==2
        
        super(HistogramMax, self).build(input_shape)

    def call(self, inputs, mask=None):
        hists,randoms = inputs
        return tf.case([(
            tf.keras.backend.learning_phase(),
            lambda: histogram_max_sample_module.histogram_max_sample(hists,randoms)
        )],default=lambda: tf.argmax(hists,axis=1))
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[2])

    '''
    def get_config(self):
        config = {}
        base_config = super(KdeHistogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    '''

global_layers_list['HistogramMax'] = HistogramMax

