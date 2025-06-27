import torch
import triton
import triton.language as tl

__all__ = [
  "fp8_bf16_matmul_",
  "int8_matmul_",
  "fp8_fp16_matmul_",
  "fp8_matmul_"
]

@triton.jit
def int8_gemm_kernel_(
    A_ptr, B_ptr, C_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    
    c = acc
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def int8_matmul_(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    leading_shape =  A.shape[:-1]
    M = leading_shape.numel()
    K = A.shape[-1]
    K, N = B.shape
    C = torch.empty((*leading_shape,N), dtype= torch.int32, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    int8_gemm_kernel_[grid](
        A, B, C,
        M, N, K,
        B.stride(0), B.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2
    )

    return C


@triton.jit
def fp8_gemm_bf16_kernel_(
    A_ptr, B_ptr, C_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def fp8_bf16_matmul_(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    leading_shape =  A.shape[:-1]
    M = leading_shape.numel()
    K = A.shape[-1]
    K, N = B.shape
    C = torch.empty((*leading_shape,N), dtype= torch.bfloat16, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    fp8_gemm_bf16_kernel_[grid](
        A, B, C,
        M, N, K,
        B.stride(0), B.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2
    )

    return C
                         
    
@triton.jit
def fp8_gemm_fp16_kernel_(
    A_ptr, B_ptr, C_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def fp8_fp16_matmul_(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    leading_shape =  A.shape[:-1]
    M = leading_shape.numel()
    K = A.shape[-1]
    K, N = B.shape
    C = torch.empty((*leading_shape,N), dtype= torch.float16, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    fp8_gemm_fp16_kernel_[grid](
        A, B, C,
        M, N, K,
        B.stride(0), B.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2
    )

    return C
                         
@triton.jit
def fp8_gemm_kernel_(
    A_ptr, B_ptr, C_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] +k < K), other=0.0) #.to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] +k < K) & (offs_n[None, :] < N), other=0.0) #.to(tl.bfloat16)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def fp8_matmul_(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    leading_shape =  A.shape[:-1]
    M = leading_shape.numel()
    K = A.shape[-1]
    K, N = B.shape
    C = torch.empty((*leading_shape,N), dtype= torch.bfloat16, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    fp8_gemm_kernel_[grid](
        A, B, C,
        M, N, K,
        B.stride(0), B.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2
    )

    return C
                         
    
