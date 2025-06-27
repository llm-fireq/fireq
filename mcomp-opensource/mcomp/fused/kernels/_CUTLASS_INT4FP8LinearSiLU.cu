#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "src/INT4FP8LinearSiLU.hpp"

namespace py = pybind11;

struct INT4FP8LinearSiLU{
    int input_dim;
    int output_dim;
    int group_size;
    INT4FP8SiLU kernel;

    INT4FP8LinearSiLU(torch::Tensor weights, 
        torch::Tensor weight_scales, 
        int input_dim, 
        int output_dim, int group_size) : input_dim(input_dim), output_dim(output_dim), group_size(group_size),
        kernel(reinterpret_cast<void*>(weights.data_ptr()), 
                                    reinterpret_cast<void*>(weight_scales.data_ptr()), 
                                    input_dim, output_dim, group_size) {};

    void forward(torch::Tensor inputs, torch::Tensor input_scales, 
        torch::Tensor outputs, int batch_size, int seq_len) 
    {
        at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
        kernel.forward(reinterpret_cast<void*>(inputs.data_ptr()), 
            reinterpret_cast<void*>(input_scales.data_ptr()),
            reinterpret_cast<void*>(outputs.data_ptr()), 
            batch_size, seq_len, defaultStream.stream());
    };
};

PYBIND11_MODULE(_CUTLASS_INT4FP8LinearSiLU, m) {
    py::class_<INT4FP8LinearSiLU>(m, "INT4FP8LinearSiLU")
        .def(py::init<torch::Tensor, torch::Tensor, int, int, int>())
        .def_readwrite("kernel", &INT4FP8LinearSiLU::kernel)
        .def_readwrite("input_dim", &INT4FP8LinearSiLU::input_dim)
        .def_readwrite("output_dim", &INT4FP8LinearSiLU::output_dim)
        .def_readwrite("group_size", &INT4FP8LinearSiLU::group_size)
        .def("forward", &INT4FP8LinearSiLU::forward);
};