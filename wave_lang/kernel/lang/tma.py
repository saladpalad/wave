from dataclasses import dataclass
from typing import Tuple, Optional
import torch.fx as fx

from wave_lang.kernel._support.dtype import DataType
from wave_lang.kernel.ops.wave_ops import CustomOp, define_op


class TensorMapSwizzle:
    SWIZZLE_NONE = 0
    SWIZZLE_32B = 1
    SWIZZLE_64B = 2
    SWIZZLE_128B = 3

    @staticmethod
    def bytes(mode: int) -> int:
        return (2**mode) * 16 if mode > 0 else 0


class TMADescriptorType:
    pass


@dataclass
class TMADescriptor(TMADescriptorType):
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    def __init__(self, 
                 tensor_shape: tuple[int, ...],
                 tensor_strides: tuple[int, ...],
                 tile_dims: tuple[int, ...],
                 dtype: DataType,
                 swizzle_mode: int = 0,
                 is_k_major: bool = True,
                 use_nvidia: bool = True): 

        self.tensor_shape = tensor_shape
        self.tensor_strides = tensor_strides
        self.tile_dims = tile_dims
        self.dtype = dtype
        self.swizzle_mode = swizzle_mode
        self.is_k_major = is_k_major
        self.use_nvidia = use_nvidia 
        self._descriptor_bytes = None  
        
    def create_on_host(self, device_ptr: int):
        """Create CUtensorMap on host using CUDA Driver API"""
        if not self.use_nvidia:
            self._descriptor_bytes = b'\x00' * 128 
            return self
        
        # Real NVIDIA path
        try:
            from cuda import cuda
            import ctypes
        except ImportError:
            raise RuntimeError(
                "TMA requires cuda-python package. Install with: pip install cuda-python"
            )
        
        result = cuda.cuGetProcAddress("cuTensorMapEncodeTiled", cuda.CUdriverProcAddressQueryResult())
        if result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to get cuTensorMapEncodeTiled from CUDA Driver API: {result[0]}")
        
        cuTensorMapEncodeTiled = result[1]
        
        # tma descriptor is 128 bytes
        self._descriptor_bytes = ctypes.create_string_buffer(128)
        
        dtype_map = {
            'f16': cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            'f32': cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            'bf16': cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        }
        
        rank = len(self.tile_dims)
        
        global_dims = (ctypes.c_uint64 * rank)(*reversed(self.tensor_shape))
        global_strides = (ctypes.c_uint64 * (rank-1))(*[s * self.dtype.size() for s in reversed(self.tensor_strides[1:])])
        box_dims = (ctypes.c_uint32 * rank)(*reversed(self.tile_dims))
        elem_strides = (ctypes.c_uint32 * rank)(*([1] * rank))
        
        swizzle_map = {
            0: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
            32: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
            64: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
            128: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        }
        
        """
        CUresult cuTensorMapEncodeTiled ( CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill )
        """
        result = cuTensorMapEncodeTiled(
            tensorMap=ctypes.byref(self._descriptor_bytes),
            tensorDataType=dtype_map.get(str(self.dtype), cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16),
            tensorRank=rank,
            globalAddress=ctypes.c_void_p(device_ptr),
            globalDim=ctypes.byref(global_dims),
            globalStrides=ctypes.byref(global_strides),
            boxDim=ctypes.byref(box_dims),
            elementStrides=ctypes.byref(elem_strides),
            interleave=cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle=swizzle_map[self.swizzle_mode],
            l2Promotion=cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            oobFill=cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
        
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")
        
        return self