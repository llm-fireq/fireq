import torch
import triton
import triton.language as tl

@triton.jit
def _fp8_gemm_fp8f16xfp8_bf16_kernel_(
    A_ptr, A_scale_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_bk, stride_bn,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
    BLOCK_K:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_scale_ptr + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    sA = tl.load(a_scale_ptrs).to(tl.float32)
    acc = acc * sA[:, None]
    
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def _fp8_gemm_fp8f16xfp8_fp16_kernel_(
    A_ptr, A_scale_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_bk, stride_bn,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
    BLOCK_K:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_scale_ptr + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    sA = tl.load(a_scale_ptrs).to(tl.float32)
    acc = acc * sA[:, None]
    
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_fp8_matmul_fp8f16xfp8(A: torch.Tensor, A_scale: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    a_scale_dtype = A_scale.dtype
    leading_shape =  A.shape[:-1]
    M = leading_shape.numel()
    K = A.shape[-1]
    K, N = B.shape
    C = torch.empty((*leading_shape,N), dtype= a_scale_dtype, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    if a_scale_dtype == torch.bfloat16:
        _fp8_gemm_fp8f16xfp8_bf16_kernel_[grid](
            A, A_scale, B, C,
            M, N, K,
            B.stride(0), B.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )
    elif a_scale_dtype == torch.float16:
        _fp8_gemm_fp8f16xfp8_fp16_kernel_[grid](
            A, A_scale, B, C,
            M, N, K,
            B.stride(0), B.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

    return C


@triton.jit
def _fp8_gemm_fp8f16xfp8f16_bf16_kernel_(
    A_ptr, A_scale_ptr, B_ptr, B_scale_ptr, C_ptr,
    M, N, K,
    stride_bk, stride_bn,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
    BLOCK_K:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_scale_ptr + offs_m[:, None]
    b_scale_ptrs = B_scale_ptr + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tmp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)

        a_scale = tl.load(a_scale_ptrs).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs).to(tl.float32)

        tmp_acc = tl.dot(a, b, allow_tf32=False)
        tmp_acc = tmp_acc*a_scale
        tmp_acc = tmp_acc*b_scale
        
        acc += tmp_acc
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]    
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def _fp8_gemm_fp8f16xfp8f16_fp16_kernel_(
    A_ptr, A_scale_ptr, B_ptr, B_scale_ptr, C_ptr,
    M, N, K,
    stride_bk, stride_bn,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
    BLOCK_K:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_scale_ptr + offs_m[:, None]
    b_scale_ptrs = B_scale_ptr + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tmp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)

        a_scale = tl.load(a_scale_ptrs).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs).to(tl.float32)
        
        tmp_acc = tl.dot(a, b, allow_tf32=False)
        tmp_acc = tmp_acc*a_scale
        tmp_acc = tmp_acc*b_scale
        
        acc += tmp_acc
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]    
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_fp8_matmul_fp8f16xfp8f16_per_tok_per_ch(A: torch.Tensor, A_scale: torch.Tensor, B: torch.Tensor, B_scale:torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    a_scale_dtype = A_scale.dtype
    leading_shape =  A.shape[:-1]
    M = leading_shape.numel()
    K = A.shape[-1]
    K, N = B.shape
    C = torch.empty((*leading_shape,N), dtype= a_scale_dtype, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    if a_scale_dtype == torch.bfloat16:
        _fp8_gemm_fp8f16xfp8f16_bf16_kernel_[grid](
            A, A_scale, B, B_scale, C,
            M, N, K,
            B.stride(0), B.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )
    elif a_scale_dtype == torch.float16:
        _fp8_gemm_fp8f16xfp8f16_fp16_kernel_[grid](
            A, A_scale, B, B_scale, C,
            M, N, K,
            B.stride(0), B.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

    return C


@triton.jit
def _i8_gemm_i8f16xi8f16_bf16_kernel_(
    A_ptr, A_scale_ptr, B_ptr, B_scale_ptr, C_ptr,
    M, N, K,
    stride_bk, stride_bn,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
    BLOCK_K:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_scale_ptr + offs_m[:, None]
    b_scale_ptrs = B_scale_ptr + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tmp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)

        a_scale = tl.load(a_scale_ptrs).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs).to(tl.float32)

        tmp_acc = tl.dot(a, b, allow_tf32=False).to(tl.float32)
        #tmp_acc = tmp_acc.to(tl.float32)
        tmp_acc = tmp_acc*a_scale
        tmp_acc = tmp_acc*b_scale
        
        acc += tmp_acc
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]    
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def _i8_gemm_i8f16xi8f16_fp16_kernel_(
    A_ptr, A_scale_ptr, B_ptr, B_scale_ptr, C_ptr,
    M, N, K,
    stride_bk, stride_bn,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
    BLOCK_K:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_scale_ptr + offs_m[:, None]
    b_scale_ptrs = B_scale_ptr + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tmp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)

        a_scale = tl.load(a_scale_ptrs).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs).to(tl.float32)

        tmp_acc = tl.dot(a, b, allow_tf32=False).to(tl.float32)
        #tmp_acc = tmp_acc.to(tl.float32)
        tmp_acc = tmp_acc*a_scale
        tmp_acc = tmp_acc*b_scale
        
        acc += tmp_acc
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]    
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_int8_matmul_i8f16xi8f16_per_tok_per_ch(A: torch.Tensor, A_scale: torch.Tensor, B: torch.Tensor, B_scale:torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    a_scale_dtype = A_scale.dtype
    leading_shape =  A.shape[:-1]
    M = leading_shape.numel()
    K = A.shape[-1]
    K, N = B.shape
    C = torch.empty((*leading_shape,N), dtype= a_scale_dtype, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    if a_scale_dtype == torch.bfloat16:
        _i8_gemm_i8f16xi8f16_bf16_kernel_[grid](
            A, A_scale, B, B_scale, C,
            M, N, K,
            B.stride(0), B.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )
    elif a_scale_dtype == torch.float16:
        _i8_gemm_i8f16xi8f16_fp16_kernel_[grid](
            A, A_scale, B, B_scale, C,
            M, N, K,
            B.stride(0), B.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

    return C

