from dataclasses import dataclass
from .._support.dtype import (
    f16,
    f8e5m2,
    f8e4m3fn,
    f6e2m3fn,
    f6e3m2fn,
    f4e2m1fn,
)


@dataclass
class InstructionDescriptor:
    """
    Instruction descriptor format for .kind::tf32, .kind::f16, .kind::f8f6f4 and .kind::i8 in tcgen05.mma
    For more info see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    """

    # TODO: Support sparsity, saturate and shift bits
    # TODO: Add support for Instruction descriptor format for 1) .kind::mxf8f6f4 2) .kind::mxf4 and .kind::mxf4nvf4

    @staticmethod
    def insert_bit(desc, val, start_bit_pos):
        return desc | (val << start_bit_pos)

    @staticmethod
    def create_tf32_desc(a_type, b_type, d_type, transpose_a, transpose_b, N, M):
        """
        .kind::tf32 Descriptor Format

        d_type is f32
        a_type is tf32
        b_type is tf32

        Returns a 32 bit descriptor
        """

        desc = 0
        desc = InstructionDescriptor.insert_bit(desc, 1, 4)
        desc = InstructionDescriptor.insert_bit(desc, 2, 7)
        desc = InstructionDescriptor.insert_bit(desc, 2, 10)
        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_a else 0, 15)
        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_b else 0, 16)
        desc = InstructionDescriptor.insert_bit(desc, N >> 3, 17)
        desc = InstructionDescriptor.insert_bit(desc, M >> 4, 24)

        return desc

    @staticmethod
    def create_f16_desc(a_type, b_type, d_type, transpose_a, transpose_b, N, M):
        """
        .kind::f16 Descriptor Format

        d_type is f16 or f32
        a_type is f16 or f32
        b_type is f16 or f32

        Returns a 32 bit descriptor
        """
        desc = 0
        desc = InstructionDescriptor.insert_bit(desc, 0 if d_type == f16 else 1, 4)
        desc = InstructionDescriptor.insert_bit(desc, 0 if a_type == f16 else 1, 7)
        desc = InstructionDescriptor.insert_bit(desc, 0 if b_type == f16 else 1, 10)
        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_a else 0, 15)
        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_b else 0, 16)
        desc = InstructionDescriptor.insert_bit(desc, N >> 3, 17)
        desc = InstructionDescriptor.insert_bit(desc, M >> 4, 24)

        return desc

    @staticmethod
    def create_f8f6f4_desc(a_type, b_type, d_type, transpose_a, transpose_b, N, M):
        """
        .kind::f8f6f4 Descriptor Format

        d_type is f16 or f32
        a_type is e4m3, e5m2, e2m3, e3m2, or e2m1
        b_type is e4m3, e5m2, e2m3, e3m2, or e2m1

        Returns a 32 bit descriptor
        """

        desc = 0
        desc = InstructionDescriptor.insert_bit(desc, 1, 4)

        if a_type == f8e4m3fn:
            desc = InstructionDescriptor.insert_bit(desc, 0, 7)
        elif a_type == f8e5m2:
            desc = InstructionDescriptor.insert_bit(desc, 1, 7)
        elif a_type == f6e2m3fn:
            desc = InstructionDescriptor.insert_bit(desc, 3, 7)
        elif a_type == f6e3m2fn:
            desc = InstructionDescriptor.insert_bit(desc, 4, 7)
        elif a_type == f4e2m1fn:
            desc = InstructionDescriptor.insert_bit(desc, 5, 7)

        if b_type == f8e4m3fn:
            desc = InstructionDescriptor.insert_bit(desc, 0, 10)
        elif b_type == f8e5m2:
            desc = InstructionDescriptor.insert_bit(desc, 1, 10)
        elif b_type == f6e2m3fn:
            desc = InstructionDescriptor.insert_bit(desc, 3, 10)
        elif b_type == f6e3m2fn:
            desc = InstructionDescriptor.insert_bit(desc, 4, 10)
        elif b_type == f4e2m1fn:
            desc = InstructionDescriptor.insert_bit(desc, 5, 10)

        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_a else 0, 15)
        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_b else 0, 16)
        desc = InstructionDescriptor.insert_bit(desc, N >> 3, 17)
        desc = InstructionDescriptor.insert_bit(desc, M >> 4, 24)

        return desc

    @staticmethod
    def create_i8_desc(a_type, b_type, d_type, transpose_a, transpose_b, N, M):
        """
        .kind::i8 Descriptor Format

        d_type is i32
        a_type is i8
        b_type is i8

        Returns a 32 bit descriptor
        """

        desc = 0
        desc = InstructionDescriptor.insert_bit(desc, 2, 4)
        # TODO: Add distinction between unsigned vs signed i8
        desc = InstructionDescriptor.insert_bit(desc, 0, 7)
        desc = InstructionDescriptor.insert_bit(desc, 0, 10)
        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_a else 0, 15)
        desc = InstructionDescriptor.insert_bit(desc, 1 if transpose_b else 0, 16)
        desc = InstructionDescriptor.insert_bit(desc, N >> 3, 17)
        desc = InstructionDescriptor.insert_bit(desc, M >> 4, 24)

        return desc
