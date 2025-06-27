#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>
#include "src/FP8FP8FmhaFwd.hpp"

namespace py = pybind11;

struct FP8FP8FmhaFwd{
    int head_group;
    int num_heads;
    int head_size;
    FwdRunner kernel;

    FP8FP8FmhaFwd(int head_group, int num_heads, int head_size) 
                    : head_group(head_group), num_heads(num_heads),
                    head_size(head_size), kernel(head_group, num_heads, head_size) {};

    void forward(torch::Tensor query, torch::Tensor key, torch::Tensor value, 
        torch::Tensor output, int batch_size, int seq_len) 
    {
        at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
        kernel.forward(reinterpret_cast<void*>(query.data_ptr()), 
            reinterpret_cast<void*>(key.data_ptr()),
            reinterpret_cast<void*>(value.data_ptr()),
            reinterpret_cast<void*>(output.data_ptr()),
            batch_size, seq_len, defaultStream.stream());
    };
};

PYBIND11_MODULE(_CUTLASS_FP8FP8FmhaFwd, m) {
    py::class_<FP8FP8FmhaFwd>(m, "FP8FP8FmhaFwd")
        .def(py::init<int, int, int>())
        .def_readwrite("kernel", &FP8FP8FmhaFwd::kernel)
        .def_readwrite("head_group", &FP8FP8FmhaFwd::head_group)
        .def_readwrite("head_size", &FP8FP8FmhaFwd::head_size)
        .def_readwrite("num_heads", &FP8FP8FmhaFwd::num_heads)
        .def("forward", &FP8FP8FmhaFwd::forward);
};