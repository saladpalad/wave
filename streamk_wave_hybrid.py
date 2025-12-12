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

M = sym.M
N = sym.N
K = sym.K

BLOCK_M = sym.BLOCK_M
BLOCK_N = sym.BLOCK_N
BLOCK_K = sym.BLOCK_K

ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C

NUM_CTAS = sym.NUM_CTAS
DP_TILE_ID_AXIS = sym.DP_TILE_ID_AXIS
TOTAL_TILES = sym.TOTAL_TILES
LOCK_DIM = sym.LOCK_DIM
READY_FLAG = sym.READY_FLAG
SPINLOCK_WAIT_FLAG = sym.SPINLOCK_WAIT_IDX
CTA_ID = sym.CTA_ID
CTA_ID_AXIS = sym.CTA_ID_AXIS
THREAD_ID = sym.THREAD_ID
NUM_K_TILES = sym.NUM_K_TILES
START_K_TILE = sym.START_K_TILE
ITERS_PER_OUTPUT_TILE = sym.ITERS_PER_OUTPUT_TILE
WORK_UNIT_START = sym.WORK_UNIT_START
WORK_UNIT_END = sym.WORK_UNIT_END
CTA_M_OFFSET = sym.CTA_M_OFFSET
CTA_N_OFFSET = sym.CTA_N_OFFSET
N_TILES = sym.N_TILES
NEW_CTA_K_END = sym.NEW_CTA_K_END
OUTPUT_TILE_ITER_END = sym.OUTPUT_TILE_ITER_END
TOTAL_STREAMK_ITERS = sym.TOTAL_STREAMK_ITERS
STREAMK_ITERS_PCU = sym.STREAMK_ITERS_PCU
STREAMK_EXTRA_ITERS = sym.STREAMK_EXTRA_ITERS
STREAMK_TILES = sym.STREAMK_TILES
DATA_PARALLEL_TILES = sym.DATA_PARALLEL_TILES
OUTPUT_TILE_ID = sym.OUTPUT_TILE_ID
SCALAR_DIM = sym.SCALAR_DIM
CTA_K_END = sym.CTA_K_END
STREAMK_TILE_ID = sym.STREAMK_TILE_ID

constraints = [
    tkw.GridConstraint(NUM_CTAS),
    tkw.WorkgroupConstraint(M, BLOCK_M, 0),
    tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    tkw.TilingConstraint(K, BLOCK_K, iters=NUM_K_TILES, start=START_K_TILE*BLOCK_K),
    tkw.TilingConstraint(DP_TILE_ID_AXIS),
    tkw.TilingConstraint(WORK_UNIT_START),
    tkw.TilingConstraint(CTA_ID_AXIS),
    tkw.TilingConstraint(SPINLOCK_WAIT_FLAG),
    tkw.WaveConstraint(M, BLOCK_M / 2),
    tkw.WaveConstraint(N, BLOCK_N / 2),
    tkw.HardwareConstraint(
        threads_per_wave=64,
        mma_type=tkw.MMAType.F32_16x16x16_F16,
        vector_shapes={DP_TILE_ID_AXIS: 0, LOCK_DIM: 1, SCALAR_DIM: 1, CTA_ID: 0, WORK_UNIT_START:0, SPINLOCK_WAIT_FLAG: 0, CTA_ID_AXIS: 0, M: 16, N: 16},
    ),
]

i = tkw.IndexMapping.iterator(0)
j = tkw.IndexMapping.iterator(1)

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
    inputs={NUM_CTAS: CTA_ID, M: THREAD_ID, N: j},
    outputs={M: i, N: j},
)

partial_buffer_write_mapping = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: i, N: j},
    outputs={NUM_CTAS: CTA_ID, M: i, N: j},
)

lock_iter = tkw.IndexMapping.iterator(0)
lock_buffer_read_mapping = tkw.IndexMapping(
    num_iterators=1,
    inputs={NUM_CTAS: CTA_ID, LOCK_DIM: lock_iter},
    outputs={LOCK_DIM: lock_iter},
)

lock_buffer_write_mapping = tkw.IndexMapping(
    num_iterators=1,
    inputs={LOCK_DIM: lock_iter},
    outputs={NUM_CTAS: CTA_ID, LOCK_DIM: lock_iter},
)

c_write_mapping = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: i, N: j},
    outputs={M: i + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
)

c_write_mapping_dp = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: i, N: j},
    outputs={M: i + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
)


@tkw.wave(constraints)
def wave_kernel(
    a: Memory[M, K, ADDRESS_SPACE_A, f16],
    b: Memory[N, K, ADDRESS_SPACE_B, f16],
    partial_buffer: Memory[NUM_CTAS, M, N, GLOBAL_ADDRESS_SPACE, f32, tkl.MemoryLayout(shape=(NUM_CTAS, BLOCK_M, BLOCK_N))],
    lock_buffer: Memory[NUM_CTAS, LOCK_DIM, GLOBAL_ADDRESS_SPACE, f32],
    c: Memory[M, N, ADDRESS_SPACE_C, f32],
):
    ### DATA PARALLEL PART
    dp_tiles_scalar = tkw.scalar(DATA_PARALLEL_TILES, tkl.i32)
    tkw.set_symbol(DATA_PARALLEL_TILES, dp_tiles_scalar)

    condition = DP_TILE_ID_AXIS < DATA_PARALLEL_TILES
    init_tile_id = tkw.scalar(WORKGROUP_0, tkl.i32)

    @tkw.iterate(DP_TILE_ID_AXIS, start=init_tile_id, condition=condition, init_args=[])
    def persistent_loop():
        tile_id = tkw.self_index(DP_TILE_ID_AXIS, tkl.i32)
        m_offset = (tile_id // tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_M, i32)
        n_offset = (tile_id % tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_N, i32)
        tkw.set_symbol(CTA_M_OFFSET, m_offset)
        tkw.set_symbol(CTA_N_OFFSET, n_offset)

        tkw.set_symbol(START_K_TILE, tkw.scalar(0, i32))
        tkw.set_symbol(NUM_K_TILES, tkw.scalar(ITERS_PER_OUTPUT_TILE, i32))

        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        @tkw.iterate(axis=K, init_args=[c_reg])
        def k_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, mapping=a_read_mapping)
            b_reg = tkw.read(b, mapping=b_read_mapping)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(k_loop, c, mapping=c_write_mapping_dp)

        num_cus_scalar = tkw.scalar(NUM_CTAS, tkl.i32)
        next_idx = tile_id + num_cus_scalar
        tkw.set_symbol(DP_TILE_ID_AXIS, next_idx)


    ### STREAMK PART
    cta_id = tkw.scalar(WORKGROUP_0, i32)
    tkw.set_symbol(CTA_ID, cta_id)
    iters_per_output_tile = tkw.scalar(ITERS_PER_OUTPUT_TILE, i32)
    total_full_tiles = tkw.scalar(DATA_PARALLEL_TILES, i32)
    sk_iters_pcu = tkw.scalar(STREAMK_ITERS_PCU, i32)
    sk_extra_iters = tkw.scalar(STREAMK_EXTRA_ITERS, i32)

    extra_iter = tkw.minimum(cta_id, sk_extra_iters)
    work_unit_start = total_full_tiles * iters_per_output_tile + (cta_id * sk_iters_pcu) + extra_iter
    next_extra_iter = tkw.minimum(cta_id + tkw.scalar(1, i32), sk_extra_iters)
    work_unit_end = total_full_tiles * iters_per_output_tile + ((cta_id + tkw.scalar(1, i32)) * sk_iters_pcu) + next_extra_iter
    tkw.set_symbol(WORK_UNIT_END, work_unit_end)
    sk_condition = WORK_UNIT_START < WORK_UNIT_END

    @tkw.iterate(WORK_UNIT_START, start=work_unit_start, condition=sk_condition, init_args=[])
    def sk_loop():
        cta_k_start = tkw.self_index(WORK_UNIT_START, i32)
        curr_k_iter = cta_k_start % iters_per_output_tile
        cta_k_end = tkw.minimum(cta_k_start + (iters_per_output_tile - curr_k_iter), work_unit_end)
        output_tile_id = cta_k_start // iters_per_output_tile
        tkw.set_symbol(OUTPUT_TILE_ID, output_tile_id)
        tkw.set_symbol(CTA_K_END, cta_k_end)

        m_offset = (output_tile_id // tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_M, i32)
        n_offset = (output_tile_id % tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_N, i32)
        tkw.set_symbol(CTA_M_OFFSET, m_offset)
        tkw.set_symbol(CTA_N_OFFSET, n_offset)

        tkw.set_symbol(START_K_TILE, curr_k_iter)
        tkw.set_symbol(NUM_K_TILES, cta_k_end - cta_k_start)

        c_reg = Register[M, N, f32](0.0)
        @tkw.iterate(K, init_args=[c_reg])
        def mac_loop(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            a_reg = tkw.read(a, mapping=a_read_mapping)
            b_reg = tkw.read(b, mapping=b_read_mapping)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc
        
        output_tile_iter_start = output_tile_id * iters_per_output_tile
        is_first_split = cta_k_start == output_tile_iter_start

        @tkw.conditional(~is_first_split)
        def store_partial():
            one_flag = Register[LOCK_DIM, f32](1.0)
            tkw.write(mac_loop, partial_buffer, mapping=partial_buffer_write_mapping, flags=tkw.MemoryAccessFlags.VOLATILE)
            tkw.write(one_flag, lock_buffer, mapping=lock_buffer_write_mapping)
        @tkw.conditional(is_first_split)
        def reduce_and_write():
            next_cta = cta_id + tkw.scalar(1, i32)
            output_tile_iter_end = output_tile_iter_start + iters_per_output_tile

            tkw.set_symbol(NEW_CTA_K_END, cta_k_end)
            new_cta_k_end = tkw.scalar(NEW_CTA_K_END, i32)

            tid = tkw.scalar(THREAD_0, i32)
            tkw.set_symbol(THREAD_ID, tid)

            # TODO: unnecessary write/read
            #tkw.write(mac_loop, partial_buffer, mapping=partial_buffer_write_mapping)
            #curr_acc = tkw.read(partial_buffer, mapping=partial_buffer_read_mapping, elements_per_thread=16)

            # 1 errors that is now blocking; after solving the partial_buffer read/write
            # since test_conditional_bug.py doesn't work anymore
            # error wave_lang.kernel.compiler.base.CodegenError: Node get_result_M:0_N:0_K:0 has no IR Value

            tkw.set_symbol(OUTPUT_TILE_ITER_END, output_tile_iter_end)

            aggregrate_partial_condition = (sympy.Lt(GET_ITER_ARG(0), OUTPUT_TILE_ITER_END)) & (CTA_ID_AXIS < NUM_CTAS)

            @tkw.iterate(CTA_ID_AXIS, start=next_cta, condition=aggregrate_partial_condition, init_args=[new_cta_k_end, mac_loop])
            def aggregate_partials_loop(loop_cta_k_end, acc):
                curr_cta = tkw.self_index(CTA_ID_AXIS, i32)
                not_ready = tkw.scalar(0, i32)
                condition = sympy.Eq(SPINLOCK_WAIT_FLAG, 0)

                tkw.set_symbol(CTA_ID, curr_cta)
                @tkw.iterate(SPINLOCK_WAIT_FLAG, start=not_ready, condition=condition, init_args=[])
                def spinlock_wait():
                    lock_val = tkw.read(lock_buffer, mapping=lock_buffer_read_mapping, flags=tkw.MemoryAccessFlags.VOLATILE)
                    one_val = Register[LOCK_DIM, f32](1.0)
                    is_ready = lock_val == one_val
                    one_int = Register[LOCK_DIM, i32](1)
                    zero_int = Register[LOCK_DIM, i32](0)
                    ready_flag = tkw.select(is_ready, one_int, zero_int)
                    tkw.set_symbol(SPINLOCK_WAIT_FLAG, ready_flag)

                tid = tkw.scalar(THREAD_0, i32)
                tkw.set_symbol(THREAD_ID, tid)
                peer_p_reg = tkw.read(
                    partial_buffer,
                    mapping=partial_buffer_read_mapping,
                )
                acc += peer_p_reg

                updated_k_end = loop_cta_k_end + tkw.scalar(STREAMK_ITERS_PCU, i32) + tkw.select(curr_cta < tkw.scalar(STREAMK_EXTRA_ITERS, i32), tkw.scalar(1, i32), tkw.scalar(0, i32))
                next_cta_val = curr_cta + tkw.scalar(1, i32)

                tkw.set_symbol(CTA_ID_AXIS, next_cta_val)
                return (updated_k_end, acc)

            tid = tkw.scalar(THREAD_0, i32)
            tkw.set_symbol(THREAD_ID, tid)
            final_k_end, final_acc = aggregate_partials_loop
            tkw.write(
                final_acc,
                c,
                mapping=c_write_mapping,
                elements_per_thread=16,
            )

        new_cta_k_start = cta_k_end
        tkw.set_symbol(WORK_UNIT_START, new_cta_k_start)


def run_test(m, n, k, streamk_tiles_count):
    block_m, block_n, block_k = 128, 256, 64

    torch.manual_seed(0)
    a = torch.rand(m, k, dtype=torch.float16, device="cuda")
    b = torch.rand(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    num_tiles_m = (m+block_m-1) // block_m
    num_tiles_n = (n+block_n-1) // block_n
    total_tiles = num_tiles_m*num_tiles_n

    print(f"\nProblem Size: M={m}, N={n}, K={k}")
    print(f"Block sizes: BLK_M={block_m}, BLK_N={block_n}, BLK_K={block_k}")
    print(f"Total tiles: {total_tiles}")

    num_ctas = 304
    iters_per_tile = (k+block_k-1) // block_k

    streamk_tiles = min(streamk_tiles_count, total_tiles)
    total_data_parallel_tiles = total_tiles - streamk_tiles

    print(f"  Grid size (NUM_CTAS): {num_ctas}")
    print(f"  Data-parallel tiles: {total_data_parallel_tiles}")
    print(f"  Stream-K tiles: {streamk_tiles}")

    total_streamk_iters = streamk_tiles * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // num_ctas
    streamk_extra_iters = total_streamk_iters % num_ctas

    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        N_TILES: num_tiles_n,
        NUM_CTAS: num_ctas,
        LOCK_DIM: 1,
        SCALAR_DIM: 1,
        ITERS_PER_OUTPUT_TILE: iters_per_tile,
        TOTAL_STREAMK_ITERS: total_streamk_iters,
        STREAMK_ITERS_PCU: streamk_iters_pcu,
        STREAMK_EXTRA_ITERS: streamk_extra_iters,
        TOTAL_TILES: total_tiles,
        STREAMK_TILES: streamk_tiles,
        DATA_PARALLEL_TILES: total_data_parallel_tiles,
    }

    partial_buffer = torch.zeros((num_ctas, block_m, block_n), dtype=torch.float32, device="cuda")
    lock_buffer = torch.zeros((num_ctas, 1), dtype=torch.float32, device="cuda")

    options = WaveCompileOptions(
        subs=hyperparams,
        print_grid=True,
        print_mlir=False
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    compiled_gemm(a, b, partial_buffer, lock_buffer, c)

    expected = torch.matmul(a, b.t())
    diff = (c - expected).abs()
    max_diff = diff.max().item()

    passed = torch.allclose(c.to(torch.float16), expected, rtol=1e-2, atol=1e-2)

    return passed, max_diff


if __name__ == "__main__":
    print("HYBRID STREAM-K GEMM TESTS")

    #m, n, k = 300, 300, 300
    #m, n, k = 2048, 2048, 2048
    #m, n, k = 1536, 3072, 19776
    #m, n, k = 1792, 2895, 2048

    #m, n, k = 128, 256, 64
    #m, n, k = 1536, 1792, 3200
    #m, n, k = 2048, 1920, 3072
    m, n, k = 1536, 3072, 19776

    # Test  Hybrid mode (most tiles DP, 1 tile SK)
    passed, diff = run_test(m=m, n=n, k=k, streamk_tiles_count=2)
    print(f"Test ({m}x{n}x{k} hybrid): {'PASSED' if passed else 'FAILED'} (max_diff={diff:.6f})")

