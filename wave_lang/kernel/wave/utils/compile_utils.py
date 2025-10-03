# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Optional, Any
import re

import torch

from iree.compiler import compile_str
from iree.compiler.dialects import (
    _structured_transform_ops_gen as structured_transform_ops,
)
from iree.compiler.dialects.transform import (
    any_op_t,
)
from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
)
from iree.compiler.dialects.transform import vector as vt

from wave_lang.kernel._support.indexing import is_literal, subs_idxc, IndexSymbol

from wave_lang.support.ir_imports import (
    InsertionPoint,
    Location,
    Module,
    Operation,
    StringAttr,
    UnitAttr,
    builtin_d,
    transform_d,
)

from ..compile_options import WaveCompileOptions


def compile_to_vmfb(
    asm: str,
    options: WaveCompileOptions,
):
    flags = [
        "--iree-vm-bytecode-module-strip-source-map=true",
        "--iree-opt-strip-assertions=true",
        "--iree-vm-target-truncate-unsupported-floats",
    ]

    # TODO: More targets/backends support.
    if options.device == "hip":
        flags.append(f"--iree-hip-target={options.target}")

    if options.device == "cuda":
        flags.append(f"--iree-cuda-target={options.target}")

    if options.mlir_print_ir_after_all:
        flags.append("--mlir-print-ir-after-all")

    if options.scalarize_packed_math:
        # scalarize_packed_math decomposes packed math into scalar math
        # so we need to disable SLP vectorization to prevent recombinining it
        # back on LLVM level.
        flags.append("--iree-hip-llvm-slp-vec=false")

    if options.iree_preprocessing_pass_pipeline:
        flags.append(
            f"--iree-preprocessing-pass-pipeline={options.iree_preprocessing_pass_pipeline}"
        )

    if options.dump_intermediates:
        flags.append(
            f"--iree-hal-dump-executable-intermediates-to={options.dump_intermediates}"
        )

    if options.dump_binaries:
        flags.append(f"--iree-hal-dump-executable-binaries-to={options.dump_binaries}")

    if options.run_bench:
        if options.benchmark_batch_size:
            flags.append(
                f"--iree-hal-benchmark-dispatch-repeat-count={options.benchmark_batch_size}"
            )

    if options.num_devices > 1:
        target_devices = [
            f"--iree-hal-target-device={options.device}[{i}]"
            for i in range(options.num_devices)
        ]
        flags += target_devices
    else:
        flags.append(f"--iree-hal-target-device={options.device}")
    res = compile_str(asm, extra_args=flags)
    return res


def apply_transform(
    module: Operation, transform_asm: str, symbols: dict[IndexSymbol, Any]
):
    symbols = {str(k): v for k, v in symbols.items()}
    pattern = r"%%[A-Za-z0-9_]+%%"

    def repl(match: re.Match) -> str:
        name = match.group()[2:-2]  # drop %% and %%
        res = symbols.get(name, None)

        if res is None:
            raise ValueError(f"Symbol {name} not found")

        res = subs_idxc(res)
        if not is_literal(res):
            raise ValueError(f"Symbol {name}: {res} is not a literal")

        return str(int(res))

    transform_asm = re.sub(pattern, repl, transform_asm)

    with module.context, Location.unknown():
        transform_module = Module.parse(transform_asm)

    transform_interpreter.apply_named_sequence(
        module,
        transform_module.body.operations[0],
        transform_module,
    )


def canonicalize_module(module: Operation):
    with module.context, Location.unknown():
        transform_module = builtin_d.Module.create()
        transform_module_op = module.operation
        transform_module_op.attributes["transform.with_named_sequence"] = UnitAttr.get()
        with InsertionPoint(transform_module.body):
            named_sequence = transform_d.NamedSequenceOp(
                "__transform_main", [any_op_t()], []
            )
            with InsertionPoint(named_sequence.body):
                target = named_sequence.body.arguments[0]
                apply_patterns = transform_d.ApplyPatternsOp(target)
                with InsertionPoint(apply_patterns.regions[0].blocks[0]):
                    transform_d.apply_patterns_canonicalization()
                    vt.apply_patterns_vector_sink_ops()
                    vt.apply_patterns_vector_sink_mem_ops()
                transform_d.apply_cse(target)
                loops = structured_transform_ops.structured_match(
                    any_op_t(), target, ops=["scf.for", "scf.while"]
                )
                transform_d.apply_licm(loops)
                transform_d.YieldOp([target])
        transform_interpreter.apply_named_sequence(
            module,
            transform_module.body.operations[0],
            transform_module,
        )


def set_default_compile_config(options: WaveCompileOptions) -> WaveCompileOptions:
    """Return default config for compilation."""
    if not torch.cuda.is_available():
        options.device = "hip"
        options.target = "gfx942"
    else:
        props = torch.cuda.get_device_properties(torch.device)
        if hasattr(props, "gcnArchName") and "NVIDIA" not in props.name:
            options.device = "hip"
            options.target = "gfx942"
        else:
            options.device = "cuda"
            options.target = "sm_86"
        return options


def get_wave_module_body_asm(module: Module) -> str:
    """
    Concatenates the MLIR of all operations within the
    body region of the top-level wave_compile() module and modifies the
    visibility of the top-level public FuncOp generated in wave_compile()
    to private, so that it gets removed when inlined.
    """
    block = module.operation.regions[0].blocks[0]
    ops_asm = []
    for op in block.operations:
        if op.operation.name == "func.func":
            op.attributes["sym_visibility"] = StringAttr.get("private")
        ops_asm.append(op.get_asm())

    return "\n".join(ops_asm)


def get_kernel_name(
    prefix: str,
    dims: Dict[str, Optional[int]],
    dtypes: Dict[str, str],
    tensor_dim_orders: Dict[str, List[str]],
) -> str:
    parts = [prefix]

    for tensor_name, dtype in dtypes.items():
        dim_order = tensor_dim_orders[tensor_name]
        if not dim_order:
            raise ValueError(f"No dimension order found for tensor '{tensor_name}'")
        shape_parts = [f"{dim}_{dims[dim]}" for dim in dim_order if dim in dims]
        parts.append("_".join(shape_parts + [dtype]))

    return "_".join(parts)
