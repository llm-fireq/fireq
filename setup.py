import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cur_dir = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')

setup(
    name='mcomp',
    version='0.0.1',
    description='FireQ LLM Model Compressor',
    author='jm2.son',
    keywords='mcomp',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="_CUTLASS_INT4FP8Linear",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4FP8Linear.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            libraries=['INT4FP8Linear'],
            library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-t 0', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4FP8LinearAdd",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4FP8LinearAdd.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            libraries=['INT4FP8LinearAdd'],
            library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-t 0', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4FP8LinearMul",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4FP8LinearMul.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            libraries=['INT4FP8LinearMul'],
            library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4FP8LinearSiLU",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4FP8LinearSiLU.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            libraries=['INT4FP8LinearSiLU'],
            library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4FP8LinearSquareDot",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4FP8LinearSquareDot.cu'],
            include_dirs=[
                os.path.join(CUDA_HOME, 'include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                ],
            libraries=['INT4FP8LinearSquareDot','cuda'],
            library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs'), '/usr/lib/x86_64-linux-gnu'],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4BF16Linear",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4BF16Linear.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            # libraries=['INT4BF16Linear'],
            # library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-t 0', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4BF16LinearAdd",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4BF16LinearAdd.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            # libraries=['INT4BF16LinearAdd'],
            # library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-t 0', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4BF16LinearMul",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4BF16LinearMul.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            # libraries=['INT4BF16LinearMul'],
            # library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4BF16LinearSiLU",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4BF16LinearSiLU.cu'],
            include_dirs=[
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                os.path.join(CUDA_HOME, 'include'),
                ],
            # libraries=['INT4BF16LinearSiLU'],
            # library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_INT4BF16LinearSquareDot",
            sources=['mcomp/fused/kernels/_CUTLASS_INT4BF16LinearSquareDot.cu'],
            include_dirs=[
                os.path.join(CUDA_HOME, 'include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                ],
            # libraries=['INT4BF16LinearSquareDot','cuda'],
            libraries=['cuda'],
            # library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs'), '/usr/lib/x86_64-linux-gnu'],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3'], 'nvcc': ['-O3', '-std=c++17', '-Xcompiler=-fPIC','--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a']
            }
        ),
        CUDAExtension(
            name="_CUTLASS_FP8FP8FmhaFwd",
            sources=['mcomp/fused/kernels/_CUTLASS_FP8FP8FmhaFwd.cu'],
            include_dirs=[
                os.path.join(CUDA_HOME, 'include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/include'),
                os.path.join(cur_dir, 'mcomp/fused/kernels/tools/util/include'),
                ],
            libraries=['FP8FP8FmhaFwd'],
            library_dirs=[os.path.join(cur_dir, 'mcomp/fused/kernels/libs')],
            # runtime_library_dirs=[os.path.join(cur_dir, "mcomp/fused/kernels/libs")],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++20'], 'nvcc': ['-O3', '-std=c++20', '-Xcompiler=-fPIC',
                                                       '-U__CUDA_NO_HALF_OPERATORS__',
                                                         '-U__CUDA_NO_HALF_CONVERSIONS__', 
                                                         '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', 
                                                         '-U__CUDA_NO_HALF2_OPERATORS__',
                                                         '--expt-relaxed-constexpr',
                                                        '-gencode=arch=compute_90a,code=sm_90a']
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

