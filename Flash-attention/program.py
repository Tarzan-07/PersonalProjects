import torch
import triton
import triton.language as tl

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, dtype=torch.float16):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=""
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    # Simple naive flash attention implementation
    dO = torch.randn_like(Q)
    softmax_scale = 1.0/ (HEAD_DIM**0.5)
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), dtype=dtype, device="cuda"))

    P = torch.matmul(Q, K.transpose((2, 3))) * softmax_scale
    if causal:
        P[:, :, MASK ==0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # Triton flash attention implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # Compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol)