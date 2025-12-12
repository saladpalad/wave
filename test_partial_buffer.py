
from wave_lang.kernel._support.indexing import sym
from wave_lang.kernel._support.dtype import f16, f32, i32
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
import torch
import sympy
from sympy import ceiling, floor    

# Define symbolic dimensions for our matrices
M = sym.M  # Rows of A and C
N = sym.N  # Rows of B and columns of C
K = sym.K  # Columns of A and B

# Define workgroup tile sizes
BLOCK_M = sym.BLOCK_M
BLOCK_N = sym.BLOCK_N
BLOCK_K = sym.BLOCK_K

# Define the address space for our memory buffers
ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C

NUM_CTAS = sym.NUM_CTAS
CTA_ID = sym.CTA_ID
THREAD_ID = sym.THREAD_ID

NUM_K_TILES = sym.NUM_K_TILES
START_K_TILE = sym.START_K_TILE
M_OFFSET = sym.M_OFFSET

# Define constraints for the kernel
constraints = [
    tkw.GridConstraint(NUM_CTAS),
    tkw.WorkgroupConstraint(M, BLOCK_M, 0),
    tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    tkw.TilingConstraint(K, BLOCK_K),
    tkw.WaveConstraint(M, BLOCK_M / 2),
    tkw.WaveConstraint(N, BLOCK_N / 2),
    tkw.HardwareConstraint(
        threads_per_wave=64,
        mma_type=tkw.MMAType.F32_16x16x16_F16,
        vector_shapes={M: 16, N: 16},
    ),
]

i = tkw.IndexMapping.iterator(0)
j = tkw.IndexMapping.iterator(1)

CTA_M_OFFSET = sym.CTA_M_OFFSET
CTA_N_OFFSET = sym.CTA_N_OFFSET
N_TILES = sym.N_TILES

partial_buffer_write_mapping = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: i, N: j},
    outputs={NUM_CTAS: CTA_ID, M: i, N: j},
)

a_read_mapping = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: i + CTA_M_OFFSET, K: j}, 
    outputs={M: i, K: j}, 
)

b_read_mapping = tkw.IndexMapping(
    num_iterators=2,
    inputs={N: i + CTA_N_OFFSET, K: j}, 
    outputs={N: i, K: j}, 
)

partial_buffer_read_mapping = tkw.IndexMapping(
    num_iterators=2,
    inputs={NUM_CTAS: CTA_ID, M: i, N: j}, 
    outputs={M: i, N: j},
)

c_write_mapping_v2 = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: i, N: j},
    outputs={M: i+CTA_M_OFFSET, N: j+CTA_N_OFFSET},
)

c_write_mapping_test = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: i, N: j},
    outputs={NUM_CTAS: CTA_ID, M: i, N: j},
)


@tkw.wave(constraints)
def wave_kernel(
    a: Memory[M, K, ADDRESS_SPACE_A, f16],
    b: Memory[N, K, ADDRESS_SPACE_B, f16],
    partial_buffer: Memory[NUM_CTAS, M, N, GLOBAL_ADDRESS_SPACE, f32, tkl.MemoryLayout(shape=(NUM_CTAS, BLOCK_M, BLOCK_N))],
    c2: Memory[NUM_CTAS, M, N, GLOBAL_ADDRESS_SPACE, f32, tkl.MemoryLayout(shape=(NUM_CTAS, BLOCK_M, BLOCK_N))],
    c: Memory[M, N, ADDRESS_SPACE_C, f32],
):
    cta_id = tkw.scalar(WORKGROUP_0, i32)
    tkw.set_symbol(CTA_ID, cta_id)
    m_offset = (cta_id // tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_M, i32)
    n_offset = (cta_id % tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_N, i32)
    tkw.set_symbol(CTA_M_OFFSET, m_offset)
    tkw.set_symbol(CTA_N_OFFSET, n_offset)

    c_reg = Register[N, M, f32](0.0)
    @tkw.iterate(K, init_args=[c_reg])
    def gemm_loop(acc: Register[N, M, f32]) -> Register[N, M, f32]:
        a_reg = tkw.read(a, mapping=a_read_mapping)
        b_reg = tkw.read(b, mapping=b_read_mapping)
        acc = tkw.mma(b_reg, a_reg, acc)
        return acc
    gemm_loop_t = tkw.permute(gemm_loop, target_shape=[M, N])

    tkw.write(gemm_loop_t, partial_buffer, mapping=partial_buffer_write_mapping) # access pattern should end here

    tid = tkw.scalar(THREAD_0, i32)
    tkw.set_symbol(THREAD_ID, tid)
    c_reg = Register[M, N, f32](0.0)

    # this works now write nodes were propagated to the read nodes
    # or w/o my changes in index_sequence_analysis.py the mlir_works.mlir file works
    p_reg = tkw.read(
        partial_buffer,
        mapping=partial_buffer_read_mapping,
    )
    

    tkw.write(
        p_reg,
        c,
        mapping=c_write_mapping_v2,
    )

    tkw.write(
        p_reg,
        c2,
        mapping=c_write_mapping_test,
    )

m, n, k = 128, 128, 32
num_ctas = 1
n_tiles = (N + BLOCK_N - 1)  // BLOCK_N

a = torch.zeros(m, k, dtype=torch.float16, device="cuda")
a[0:128, :] = 1.0   
#a[128:256, :] = 2.0 

b = torch.zeros(n, k, dtype=torch.float16, device="cuda")
b[0:128, :] = 1.0 
#b[128:256, :] = 3.0 

block_m = 128
block_n = 128
partial_buffer = torch.zeros((num_ctas, block_m, block_n), dtype=torch.float32, device="cuda")
c2 = torch.zeros((num_ctas, block_m, block_n), dtype=torch.float32, device="cuda")
c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

hyperparams = {
    ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
    ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
    ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
    BLOCK_M: block_m,
    BLOCK_N: block_n, 
    BLOCK_K: 32,
    M: m,
    N: n,
    K: k,
    N_TILES: n_tiles,
    NUM_CTAS: num_ctas,
}

# Compile the kernel
options = WaveCompileOptions(
    subs=hyperparams,
    print_grid=True,
    print_mlir=True
)
options = set_default_run_config(options)
compiled_gemm = wave_compile(options, wave_kernel)

# Run the kernel
compiled_gemm(a, b, partial_buffer, c2, c)
#breakpoint()
print("parital_buffer", partial_buffer)
print("parital_buffer[0]", partial_buffer[0])
#print("parital_buffer[1]", partial_buffer[1])
#print("parital_buffer[2]", partial_buffer[2])
#print("parital_buffer[3]", partial_buffer[3])

print("c2 to test what p_reg is reading", c2)
print("c2[0]", c2[0])
#print("c2[1]", c2[1])
#print("c2[2]", c2[2])
#print("c2[3]", c2[3])


print("number of non-zeros in c2", (c2!=0).sum().item())


print("c", c)

expected = torch.matmul(a, b.t())
print("expected", expected)

diff = (c - expected).abs()
max_diff = diff.max()

diff_c2 = (c2[0] - expected).abs()
max_diff_c2 = diff_c2.max()
print(f"Max diff between c2[0] and expected: {max_diff_c2}")

assert torch.allclose(c2[0].to(torch.float16), expected, rtol=1e-2, atol=1e-2), \
    f"GEMM result (c2) doesn't match expected output\nMax difference: {max_diff_c2}"
print('test passed (c2 matches expected)')