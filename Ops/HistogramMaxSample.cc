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
    .Attr("bias: float")
    .SetShapeFn([](InferenceContext* c) 
    {
        ShapeHandle hists_input_shape = c->input(0);

        ShapeHandle ouput_shape = c->MakeShape({
            c->Dim(hists_input_shape,0), //batches
            c->Dim(hists_input_shape,2) //hists
        
        });

        c->set_output(0,ouput_shape);

        return Status::OK();
    })
    .Doc(R"doc(Produces a 1D Histogram using the given KDE from the input values and weights)doc");
    

class HistogramMaxSampleOp:
    public OpKernel
{  
    protected:
        float bias_;

    public:
        explicit HistogramMaxSampleOp(OpKernelConstruction* context):
            OpKernel(context)
        {
            OP_REQUIRES_OK(context,context->GetAttr("bias",&bias_));
            OP_REQUIRES(
                context,
                (bias_>0),
                errors::InvalidArgument("Bias required to be >0")
            );
        }

        virtual ~HistogramMaxSampleOp()
        {
        }

        void Compute(OpKernelContext* context)
        { 
            const Tensor& hists_input_tensor = context->input(0);
            const Tensor& random_input_tensor = context->input(1);

            TensorShape outputShape({
                hists_input_tensor.dim_size(0),
                hists_input_tensor.dim_size(2)
            });

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, outputShape, &output_tensor)
            );
            
            //inputs shape: [batch,bins,hists]
            const auto hists = hists_input_tensor.tensor<float,3>();
            
            //randoms shape: [batch,hists]
            const auto randoms = random_input_tensor.tensor<float,2>();
            
            //outputs shape: [batch,hists]
            auto outputs = output_tensor->tensor<float,2>();
            
            for (int ibatch = 0; ibatch<hists_input_tensor.dim_size(0); ++ibatch)
            {   
                for (int ihist = 0; ihist<hists_input_tensor.dim_size(2); ++ihist)
                {
                    //adding small minimum value ensures sum>0 so if histogram empty or all negative
                    //uniform sampling is still applied
                    float sum = 0;
                    for (int ibin = 0; ibin<hists_input_tensor.dim_size(1); ++ibin)
                    {
                        sum += std::max<float>(0.,hists(ibatch,ibin,ihist));
                    }
                    const float minFill = sum*bias_/hists_input_tensor.dim_size(1);
                    sum += minFill*hists_input_tensor.dim_size(1);
                    
                    float rnd_value = utils::clamp<float>(randoms(ibatch,ihist),0,1)*sum;
                    
                    outputs(ibatch,ihist) = 0;
                    for (int ibin = 0; ibin<hists_input_tensor.dim_size(1); ++ibin)
                    {
                        const float value = std::max<float>(0.0f,hists(ibatch,ibin,ihist))+minFill;
                        sum -= value;
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

