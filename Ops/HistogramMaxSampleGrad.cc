#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "Utils.h"
#include "Kernels.h"

#include <iostream>
#include <limits>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("HistogramMaxSampleGrad")
    .Input("hists: float")
    .Input("randoms: float")
    .Input("gradients: float")
    .Output("gradients_hists: float")
    .Output("gradients_randoms: float")
    .SetShapeFn([](InferenceContext* c) 
    {
        c->set_output(0,c->input(0));
        c->set_output(1,c->input(1));
        return Status::OK();
    })
    .Doc(R"doc(Produces a 1D Histogram using the given KDE from the input values and weights)doc");


class HistogramMaxSampleGradOp:
    public OpKernel
{  
    public:
        explicit HistogramMaxSampleGradOp(OpKernelConstruction* context):
            OpKernel(context)
        {
        }

        virtual ~HistogramMaxSampleGradOp()
        {
        }

        void Compute(OpKernelContext* context)
        { 
            const Tensor& hists_input_tensor = context->input(0);
            const Tensor& random_input_tensor = context->input(1);
            const Tensor& gradients_input_tensor = context->input(2);


            Tensor* output_gradients_hists = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, hists_input_tensor.shape(), &output_gradients_hists)
            );
            
            Tensor* output_gradients_randoms = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(1, random_input_tensor.shape(), &output_gradients_randoms)
            );
            
            
            //inputs shape: [batch,bins,hists]
            const auto hists = hists_input_tensor.tensor<float,3>();
            
            //randoms shape: [batch,hists]
            const auto randoms = random_input_tensor.tensor<float,2>();
            
            //gradients shape: [batch,hists]
            const auto gradients = gradients_input_tensor.tensor<float,2>();
            
            auto gradients_hists = output_gradients_hists->tensor<float,3>();
            auto gradients_randoms = output_gradients_randoms->tensor<float,2>();
            
            for (int ibatch = 0; ibatch<hists_input_tensor.dim_size(0); ++ibatch)
            {   
                for (int ihist = 0; ihist<hists_input_tensor.dim_size(2); ++ihist)
                {
                    //adding small minimum value ensures sum>0 so if histogram empty or all negative
                    //uniform sampling is still applied
                    float sum = 0;
                    for (int ibin = 0; ibin<hists_input_tensor.dim_size(1); ++ibin)
                    {
                        sum += std::max<float>(std::numeric_limits<float>::min(),hists(ibatch,ibin,ihist));
                    }
                    float rnd_value = utils::clamp<float>(randoms(ibatch,ihist),0,1)*sum;
                    
                    //set no gradients for randoms
                    //note: returning a simple constant in tf.RegisterGradient does not work
                    gradients_randoms(ibatch,ihist) = 0; 
                    
                    for (int ibin = 0; ibin<hists_input_tensor.dim_size(1); ++ibin)
                    {
                        //init gradients here so do not break loop
                        gradients_hists(ibatch,ibin,ihist) = 0;
                        
                        sum -= std::max<float>(std::numeric_limits<float>::min(),hists(ibatch,ibin,ihist));
                        if (sum<rnd_value)
                        {
                            gradients_hists(ibatch,ibin,ihist) = gradients(ibatch,ihist);
                            rnd_value = -1; //yields false for all following bins since sum>0 w/o breaking loop
                        }
                    }
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("HistogramMaxSampleGrad").Device(DEVICE_CPU),HistogramMaxSampleGradOp)

