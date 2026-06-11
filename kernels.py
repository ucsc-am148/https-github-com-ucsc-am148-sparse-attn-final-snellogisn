"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl
 
LOG2E = tl.constexpr(1.4426950408889634)

 
# =============================================================================
# A1 -- DSD: block-sparse (BCSR) A @ dense B -> dense C
# =============================================================================
 
@triton.jit
def _dsd_kernel(values_ptr, row_offsets_ptr, column_indices_ptr, b_ptr, c_ptr,
                M, N,
                stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK: tl.constexpr,      # BCSR block size
                BLOCK_M: tl.constexpr,    # rows per program (divides BLOCK)
                BLOCK_N: tl.constexpr,    # cols per program
                BLOCK_K: tl.constexpr):   # inner-K sub-tile (divides BLOCK)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
 
    row_start = pid_m * BLOCK_M
    block_row = row_start // BLOCK          # the single BCSR row we live in
    row_in_block = row_start % BLOCK
 
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
 
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
    start = tl.load(row_offsets_ptr + block_row)
    end = tl.load(row_offsets_ptr + block_row + 1)
 
    for p in range(start, end):
        p64 = p.to(tl.int64)
        col = tl.load(column_indices_ptr + p64).to(tl.int64)
        # walk the block's K extent in BLOCK_K sub-tiles
        for k0 in range(0, BLOCK, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a_ptrs = (values_ptr + p64 * BLOCK * BLOCK
                      + (row_in_block + tl.arange(0, BLOCK_M))[:, None] * BLOCK
                      + offs_k[None, :])
            a = tl.load(a_ptrs)                                   # (BLOCK_M, BLOCK_K) fp32
            b_ptrs = (b_ptr + (col * BLOCK + offs_k)[:, None] * stride_bk
                      + offs_n[None, :] * stride_bn)
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)  # (BLOCK_K, BLOCK_N) fp32
            acc = tl.dot(a, b, acc, input_precision="ieee")
 
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & n_mask[None, :])
 
 
def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.
 
    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32
    """
    assert M % block == 0 and K % block == 0
    values = values.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=B.device, dtype=torch.float32)
 
    BLOCK_M = min(block, 64)
    BLOCK_K = min(block, 64)
    BLOCK_N = 64
    grid = (M // BLOCK_M, triton.cdiv(N, BLOCK_N))
    _dsd_kernel[grid](
        values, row_offsets, column_indices, B, C,
        M, N,
        B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK=block, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return C
 
 
# =============================================================================
# A2 -- block-sparse flash attention forward
# =============================================================================
 
@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
                qro_ptr, qci_ptr,
                sm_scale, T,
                stride_z, stride_t,        # shared by Q/K/V/O: (Z, T, d), d contiguous
                stride_lz,                 # L: (Z, T), T contiguous
                D: tl.constexpr,
                BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                BLOCK_D: tl.constexpr):
    pid_q = tl.program_id(0)               # query block index i
    pid_z = tl.program_id(1).to(tl.int64)  # flattened (b, h)
 
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_q < T
    d_mask = offs_d < D
 
    base = pid_z * stride_z
    q_ptrs = q_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
 
    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
 
    qk_scale = sm_scale * LOG2E            # base-2 softmax: exp(x) = exp2(x*LOG2E)
 
    start = tl.load(qro_ptr + pid_q)
    end = tl.load(qro_ptr + pid_q + 1)
    for p in range(start, end):
        j = tl.load(qci_ptr + p.to(tl.int64)).to(tl.int64)
        offs_k = j * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < T
 
        k_ptrs = k_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
 
        qk = tl.dot(q, tl.trans(k)) * qk_scale            # (BLOCK_Q, BLOCK_K) fp32
        qk = tl.where(k_mask[None, :], qk, float("-inf"))  # dead keys vanish
 
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)                  # rescale of old state
        p_tile = tl.math.exp2(qk - m_new[:, None])
 
        l_i = l_i * alpha + tl.sum(p_tile, 1)
        acc = acc * alpha[:, None]
 
        v_ptrs = v_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :]
        v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        acc = tl.dot(p_tile.to(v.dtype), v, acc)
 
        m_i = m_new
 
    acc = acc / l_i[:, None]
    L = m_i + tl.math.log2(l_i)            # log2 of the softmax denominator
 
    o_ptrs = o_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty),
             mask=q_mask[:, None] & d_mask[None, :])
    tl.store(l_ptr + pid_z * stride_lz + offs_q, L, mask=q_mask)
 
 
def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.
 
    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator
    """
    B, H, T, d = Q.shape
    Z = B * H
    Qf = Q.contiguous().view(Z, T, d)
    Kf = K.contiguous().view(Z, T, d)
    Vf = V.contiguous().view(Z, T, d)
 
    O = torch.empty((Z, T, d), device=Q.device, dtype=torch.float16)
    L = torch.empty((Z, T), device=Q.device, dtype=torch.float32)
 
    BLOCK_D = max(16, triton.next_power_of_2(d))
    n_q_blocks = triton.cdiv(T, BLOCK_Q)
    grid = (n_q_blocks, Z)
    _fwd_kernel[grid](
        Qf, Kf, Vf, O, L,
        q_row_offsets, q_col_indices,
        sm_scale, T,
        Qf.stride(0), Qf.stride(1),
        L.stride(0),
        D=d, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4 if BLOCK_Q <= 64 else 8,
    )
    return O.view(B, H, T, d), L.view(B, H, T)
 
 
# =============================================================================
# A3 -- block-sparse flash attention backward
# =============================================================================
 
@triton.jit
def _bwd_dkdv_kernel(q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, d_ptr,
                     dk_ptr, dv_ptr,
                     kro_ptr, kci_ptr,
                     sm_scale, T,
                     stride_z, stride_t,
                     stride_lz,
                     D: tl.constexpr,
                     BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                     BLOCK_D: tl.constexpr):
    pid_k = tl.program_id(0)               # key block index j
    pid_z = tl.program_id(1).to(tl.int64)
 
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = offs_k < T
    d_mask = offs_d < D
    base = pid_z * stride_z
 
    k = tl.load(k_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :],
                mask=k_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :],
                mask=k_mask[:, None] & d_mask[None, :], other=0.0)
 
    acc_dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
 
    qk_scale = sm_scale * LOG2E
 
    start = tl.load(kro_ptr + pid_k)
    end = tl.load(kro_ptr + pid_k + 1)
    for p in range(start, end):
        i = tl.load(kci_ptr + p.to(tl.int64)).to(tl.int64)
        offs_q = i * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < T
 
        q = tl.load(q_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :],
                    mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        do = tl.load(do_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :],
                     mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        l_i = tl.load(l_ptr + pid_z * stride_lz + offs_q, mask=q_mask, other=float("inf"))
        d_i = tl.load(d_ptr + pid_z * stride_lz + offs_q, mask=q_mask, other=0.0)
 
        qk = tl.dot(q, tl.trans(k)) * qk_scale            # (BLOCK_Q, BLOCK_K)
        p_tile = tl.math.exp2(qk - l_i[:, None])           # recovered P_ij
        p_tile = tl.where(q_mask[:, None] & k_mask[None, :], p_tile, 0.0)
 
        # dV_j += P^T @ dO_i
        acc_dv = tl.dot(tl.trans(p_tile.to(do.dtype)), do, acc_dv)
 
        # dS = P * (dP - D),  dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))                       # (BLOCK_Q, BLOCK_K) fp32
        ds = p_tile * (dp - d_i[:, None])
 
        # dK_j += sigma * dS^T @ Q_i  (sigma applied once at the end)
        acc_dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc_dk)
 
    acc_dk = acc_dk * sm_scale
 
    out_mask = k_mask[:, None] & d_mask[None, :]
    tl.store(dk_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :],
             acc_dk.to(dk_ptr.dtype.element_ty), mask=out_mask)
    tl.store(dv_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :],
             acc_dv.to(dv_ptr.dtype.element_ty), mask=out_mask)
 
 
@triton.jit
def _bwd_dq_kernel(q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, d_ptr,
                   dq_ptr,
                   qro_ptr, qci_ptr,
                   sm_scale, T,
                   stride_z, stride_t,
                   stride_lz,
                   D: tl.constexpr,
                   BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                   BLOCK_D: tl.constexpr):
    pid_q = tl.program_id(0)               # query block index i
    pid_z = tl.program_id(1).to(tl.int64)
 
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_q < T
    d_mask = offs_d < D
    base = pid_z * stride_z
 
    q = tl.load(q_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :],
                mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    do = tl.load(do_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :],
                 mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    l_i = tl.load(l_ptr + pid_z * stride_lz + offs_q, mask=q_mask, other=float("inf"))
    d_i = tl.load(d_ptr + pid_z * stride_lz + offs_q, mask=q_mask, other=0.0)
 
    acc_dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
 
    qk_scale = sm_scale * LOG2E
 
    start = tl.load(qro_ptr + pid_q)
    end = tl.load(qro_ptr + pid_q + 1)
    for p in range(start, end):
        j = tl.load(qci_ptr + p.to(tl.int64)).to(tl.int64)
        offs_k = j * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < T
 
        k = tl.load(k_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :],
                    mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :],
                    mask=k_mask[:, None] & d_mask[None, :], other=0.0)
 
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        p_tile = tl.math.exp2(qk - l_i[:, None])
        p_tile = tl.where(k_mask[None, :], p_tile, 0.0)
 
        dp = tl.dot(do, tl.trans(v))                       # (BLOCK_Q, BLOCK_K)
        ds = p_tile * (dp - d_i[:, None])
 
        # dQ_i += sigma * dS @ K_j  (sigma applied once at the end)
        acc_dq = tl.dot(ds.to(k.dtype), k, acc_dq)
 
    acc_dq = acc_dq * sm_scale
 
    tl.store(dq_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :],
             acc_dq.to(dq_ptr.dtype.element_ty),
             mask=q_mask[:, None] & d_mask[None, :])
 
 
def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.
 
    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16
    """
    B, H, T, d = Q.shape
    Z = B * H
    Qf = Q.contiguous().view(Z, T, d)
    Kf = K.contiguous().view(Z, T, d)
    Vf = V.contiguous().view(Z, T, d)
    Of = O.contiguous().view(Z, T, d)
    dOf = dO.contiguous().view(Z, T, d)
    Lf = L.contiguous().view(Z, T)
 
    # (1) D_i = rowsum(dO * O), fp32, O(B*H*T) memory.
    Dmat = (dOf.float() * Of.float()).sum(dim=-1)          # (Z, T) fp32
 
    dQ = torch.empty((Z, T, d), device=Q.device, dtype=torch.float16)
    dK = torch.empty((Z, T, d), device=Q.device, dtype=torch.float16)
    dV = torch.empty((Z, T, d), device=Q.device, dtype=torch.float16)
 
    BLOCK_D = max(16, triton.next_power_of_2(d))
    n_k_blocks = triton.cdiv(T, BLOCK_K)
    n_q_blocks = triton.cdiv(T, BLOCK_Q)
    nw_k = 4 if BLOCK_K <= 64 else 8
    nw_q = 4 if BLOCK_Q <= 64 else 8
 
    # (2) dK, dV: one program per key block, walking the key-block view.
    _bwd_dkdv_kernel[(n_k_blocks, Z)](
        Qf, Kf, Vf, dOf, Lf, Dmat,
        dK, dV,
        k_row_offsets, k_col_indices,
        sm_scale, T,
        Qf.stride(0), Qf.stride(1),
        Lf.stride(0),
        D=d, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=nw_k,
    )
 
    # (3) dQ: one program per query block, walking the query-block view.
    _bwd_dq_kernel[(n_q_blocks, Z)](
        Qf, Kf, Vf, dOf, Lf, Dmat,
        dQ,
        q_row_offsets, q_col_indices,
        sm_scale, T,
        Qf.stride(0), Qf.stride(1),
        Lf.stride(0),
        D=d, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=nw_q,
    )
 
    return dQ.view(B, H, T, d), dK.view(B, H, T, d), dV.view(B, H, T, d)