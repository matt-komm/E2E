#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "Utils.h"
#include "Kernels.h"

#include <iostream>
#include <limits>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("HistogramMaxSample")
    .Input("hists: float")
    .Input("randoms: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) 
    {
        ShapeHandle values_input_shape = c->input(0);

        ShapeHandle ouput_shape = c->MakeShape({
            c->Dim(values_input_shape,0), //batches
            c->Dim(values_input_shape,2) //hists
        
        });

        c->set_output(0,ouput_shape);

        return Status::OK();
    })
    .Doc(R"doc(Produces a 1D Histogram using the given KDE from the input values and weights)doc");

#include <random>

namespace 
{
    static thread_local std::unique_ptr<std::mt19937> generator_;
}


class HistogramMaxSampleOp:
    public OpKernel
{  
    public:
        explicit HistogramMaxSampleOp(OpKernelConstruction* context):
            OpKernel(context)
        {
        }

        virtual ~HistogramMaxSampleOp()
        {
        }

        void Compute(OpKernelContext* context)
        { 
        
            if (not generator_)
            {
                generator_.reset(new std::mt19937(123456));
            }
        
            const Tensor& values_input_tensor = context->input(0);
            const Tensor& random_input_tensor = context->input(1);

            TensorShape outputShape({
                values_input_tensor.dim_size(0),
                values_input_tensor.dim_size(2)
            });

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, outputShape, &output_tensor)
            );
            
            //inputs shape: [batch,bins,hists]
            const auto inputs = values_input_tensor.tensor<float,3>();
            
            //randoms shape: [batch,hists]
            const auto randoms = random_input_tensor.tensor<float,2>();
            
            //outputs shape: [batch,hists]
            auto outputs = output_tensor->tensor<float,2>();
            
            for (int ibatch = 0; ibatch<values_input_tensor.dim_size(0); ++ibatch)
            {   
                for (int ihist = 0; ihist<values_input_tensor.dim_size(2); ++ihist)
                {
                    //adding small minimum value ensures sum>0 so if histogram empty or all negative
                    //uniform sampling is still applied
                    float sum = 0;
                    for (int ibin = 0; ibin<values_input_tensor.dim_size(1); ++ibin)
                    {
                        sum += std::max<float>(std::numeric_limits<float>::min(),inputs(ibatch,ibin,ihist));
                    }
                    float rnd_value = utils::clamp<float>(randoms(ibatch,ihist),0,1)*sum;
                    
                    outputs(ibatch,ihist) = 0;
                    for (int ibin = 0; ibin<values_input_tensor.dim_size(1); ++ibin)
                    {
                        sum -= std::max<float>(std::numeric_limits<float>::min(),inputs(ibatch,ibin,ihist));
                        if (sum<rnd_value)
                        {
                            outputs(ibatch,ihist) = ibin;
                            break;
                        }
                    }
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("HistogramMaxSample").Device(DEVICE_CPU),HistogramMaxSampleOp)

