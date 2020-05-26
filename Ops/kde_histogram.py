import tensorflow as tf
import keras
import os

global_layers_list = {}  # same as for losses

kde_histogram_module = tf.load_op_library(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'libKDEHistogram.so'
    )
)

class KDEHistogram(keras.engine.Layer):
    

    @tf.RegisterGradient("KDEHistogram")
    def _kde_histogram_grad(op, grad):
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

    def __init__(self,
        nbins=256,
        start=-15.,
        end=15.,
        kernel="flat",
        bandwidth_hist = 1e-12,
        bandwidth_grad = None,
        add_overflow = True,
        **kwargs
    ):
        super(KDEHistogram, self).__init__(**kwargs)
        self.supports_masking = False

        self.nbins = nbins
        self.start = start
        self.end = end
        self.kernel = kernel
        self.bandwidth_hist = bandwidth_hist
        if bandwidth_grad==None:
            self.bandwidth_grad = self.bandwidth_hist
        else:
            self.bandwidth_grad = bandwidth_grad
        self.add_overflow = add_overflow

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape)==2
        
        value_shape, weights_shape = input_shape
        
        assert(len(value_shape)==2)
        assert(len(weights_shape)==3)
        
        super(KDEHistogram, self).build(input_shape) 
        
        
    def call(self, x):
        assert isinstance(x, list)
        assert len(x)==2
        
        value, weights = x
        
        assert(len(value.shape)==2)
        assert(len(weights.shape)==3)
        
        hists = []
        
        for weight in tf.unstack(weights,axis=2):
            hists.append(
                kde_histogram_module.kde_histogram(
                    value,
                    weight,
                    self.nbins,
                    self.start,
                    self.end,
                    self.kernel,
                    self.bandwidth_hist,
                    self.bandwidth_grad,
                    self.add_overflow
                )
            )
        
        if len(hists)==1:
            return tf.expand_dims(hists[0],axis=2)
        else:
            return tf.stack(hists,axis=2)
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape)==2
        
        value_shape, weights_shape = input_shape
        return (weights_shape[0], self.nbins, weights_shape[2])
        


global_layers_list['KdeHistogram'] = KDEHistogram

    
