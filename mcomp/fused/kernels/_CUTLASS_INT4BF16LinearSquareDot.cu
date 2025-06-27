#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>
#include "src/INT4BF16LinearSquareDot.hpp"

namespace py = pybind11;

struct INT4BF16LinearSquareDot{
    int input_dim;
    int output_dim;
    int group_size;
    INT4BF16SquareDot kernel;

    INT4BF16LinearSquareDot(torch::Tensor weights, 
        torch::Tensor weight_scales, 
        int input_dim, 
        int output_dim, int group_size) : input_dim(input_dim), output_dim(output_dim), group_size(group_size),
        kernel(reinterpret_cast<void*>(weights.data_ptr()), 
                                    reinterpret_cast<void*>(weight_scales.data_ptr()), 
                                    input_dim, output_dim, group_size) {};

    void forward(torch::Tensor inputs, torch::Tensor input_scales, 
        torch::Tensor residuals, torch::Tensor outputs,
        torch::Tensor square_sum, int batch_size, int seq_len) 
    {
        at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
        kernel.forward(reinterpret_cast<void*>(inputs.data_ptr()), 
        reinterpret_cast<void*>(input_scales.data_ptr()),
        reinterpret_cast<void*>(residuals.data_ptr()),
        reinterpret_cast<void*>(outputs.data_ptr()), 
        reinterpret_cast<void*>(square_sum.data_ptr()),
            batch_size, seq_len, defaultStream.stream());
    };
};

PYBIND11_MODULE(_CUTLASS_INT4BF16LinearSquareDot, m) {
    py::class_<INT4BF16LinearSquareDot>(m, "INT4BF16LinearSquareDot")
        .def(py::init<torch::Tensor, torch::Tensor, int, int, int>())
        .def_readwrite("kernel", &INT4BF16LinearSquareDot::kernel)
        .def_readwrite("input_dim", &INT4BF16LinearSquareDot::input_dim)
        .def_readwrite("output_dim", &INT4BF16LinearSquareDot::output_dim)
        .def_readwrite("group_size", &INT4BF16LinearSquareDot::group_size)
        .def("forward", &INT4BF16LinearSquareDot::forward);
};