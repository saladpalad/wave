from wave_lang.kernel._support.indexing import sym
from wave_lang.kernel._support.dtype import f16, f32
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
import torch

M = sym.M
N = sym.N
K = sym.K
BLOCK_M = sym.BLOCK_M
BLOCK_N = sym.BLOCK_N
BLOCK_K = sym.BLOCK_K
NUM_CTAS = sym.NUM_CTAS
CTA_ID = sym.CTA_ID
ADDRESS_SPACE = sym.ADDRESS_SPACE

i = tkw.IndexMapping.iterator(0)
j = tkw.IndexMapping.iterator(1)
cta_id_mapping = tkw.IndexMapping(
    num_iterators=2,
    inputs={NUM_CTAS: CTA_ID, M: i, N: j},
    outputs={M: i, N: j},
)

constraints = [
    tkw.WorkgroupConstraint(M, BLOCK_M, 0),
    tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    tkw.TilingConstraint(K, BLOCK_K),
    tkw.WaveConstraint(M, BLOCK_M / 2),
    tkw.WaveConstraint(N, BLOCK_N / 2),
    tkw.HardwareConstraint(
        threads_per_wave=64,
        mma_type=tkw.MMAType.F32_16x16x16_F16,
        vector_shapes={M: 16, N: 16}
    )
]


@tkw.wave(constraints)
def test_kernel(
    a: Memory[M, K, ADDRESS_SPACE, f16],
    b: Memory[N, K, ADDRESS_SPACE, f16],
    c: Memory[M, N, GLOBAL_ADDRESS_SPACE, f32],
):
    c_reg = Register[M, N, f32](0.0)

    @tkw.iterate(axis=K, init_args=[c_reg])
    def mac_loop(acc):
        a_reg = tkw.read(a)
        b_reg = tkw.read(b)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc
    
    cta_id = tkw.scalar(WORKGROUP_0, tkl.i32)
    is_first = cta_id == tkw.scalar(0, tkl.i32)

    @tkw.conditional(is_first)
    def use_result():
        new_acc = mac_loop
        tkw.set_symbol(CTA_ID, cta_id)
        # gives old error: wave_lang.kernel.compiler.base.CodegenError: Node get_result_M:0_N:0_K:0 has no IR Value
        # had a previous fix for this
        # https://github.com/iree-org/wave/pull/443 no longers work though...
        @tkw.iterate(K, init_args=[new_acc]) # new_acc has no proxy value
        def repeat_v2(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            c_reg = Register[M,N, f32](1.0)
            acc += c_reg
            return acc
        
        tkw.write(repeat_v2, c)


def test():
    m, n, k = 4, 4, 4
    block_m, block_n, block_k = 4, 4, 4
    num_ctas = 304

    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        NUM_CTAS: num_ctas,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        print_grid=True,
        canonicalize=True,
 #       print_mlir=True,
    #    print_ir_after="all",
    )
    options = set_default_run_config(options)

    partial_buffer = torch.zeros((num_ctas, m, n), dtype=torch.float32, device="cuda")

    compiled_kernel = wave_compile(options, test_kernel)

    compiled_kernel(a, b, partial_buffer, c)

    expected = torch.matmul(a, b.t())

    print("C", c)
    print("Expected", expected)

    assert torch.allclose(c.to(torch.float16), expected, rtol=1e-2, atol=1e-2), \
        f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"
    
    print("test passed")


if __name__ == "__main__":
    test()
