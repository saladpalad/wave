#map = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map1 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
#map2 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
#map3 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map4 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map5 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
#map6 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16)>
#map7 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 16)>
#map8 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 32)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 48)>
#map10 = affine_map<()[s0, s1] -> (s0 * 64 + ((s1 mod 64) floordiv 16) * 4)>
#map11 = affine_map<()[s0, s1] -> (s0 * 64 + ((s1 mod 64) floordiv 16) * 4 + 16)>
#map12 = affine_map<()[s0, s1] -> (s0 * 64 + ((s1 mod 64) floordiv 16) * 4 + 32)>
#map13 = affine_map<()[s0, s1] -> (s0 * 64 + ((s1 mod 64) floordiv 16) * 4 + 48)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @wave_kernel {
    stream.executable.export public @wave_kernel workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @wave_kernel(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<f16>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<f16>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<f32>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
        %thread_id_x = gpu.thread_id  x upper_bound 128
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [128, 32], strides: [32, 1] : memref<f16> to memref<128x32xf16, strided<[32, 1], offset: ?>>
        %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [128, 32], strides: [32, 1] : memref<f16> to memref<128x32xf16, strided<[32, 1], offset: ?>>
        %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [1, 128, 128], strides: [16384, 128, 1] : memref<f32> to memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>
        %reinterpret_cast_2 = memref.reinterpret_cast %3 to offset: [%c0], sizes: [1, 128, 128], strides: [16384, 128, 1] : memref<f32> to memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>
        %reinterpret_cast_3 = memref.reinterpret_cast %4 to offset: [%c0], sizes: [128, 128], strides: [128, 1] : memref<f32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
        %5 = affine.apply #map()[%thread_id_x]
        %6 = affine.apply #map1()[%thread_id_x]
        %7 = vector.load %reinterpret_cast[%5, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %8 = affine.apply #map2()[%thread_id_x]
        %9 = vector.load %reinterpret_cast[%5, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %10 = affine.apply #map3()[%thread_id_x]
        %11 = vector.load %reinterpret_cast[%10, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %12 = vector.load %reinterpret_cast[%10, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %13 = affine.apply #map4()[%thread_id_x]
        %14 = vector.load %reinterpret_cast[%13, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %15 = vector.load %reinterpret_cast[%13, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %16 = affine.apply #map5()[%thread_id_x]
        %17 = vector.load %reinterpret_cast[%16, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %18 = vector.load %reinterpret_cast[%16, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %19 = affine.apply #map6()[%thread_id_x, %thread_id_y]
        %20 = vector.load %reinterpret_cast_0[%19, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %21 = vector.load %reinterpret_cast_0[%19, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %22 = affine.apply #map7()[%thread_id_x, %thread_id_y]
        %23 = vector.load %reinterpret_cast_0[%22, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %24 = vector.load %reinterpret_cast_0[%22, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %25 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %26 = vector.load %reinterpret_cast_0[%25, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %27 = vector.load %reinterpret_cast_0[%25, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %28 = affine.apply #map9()[%thread_id_x, %thread_id_y]
        %29 = vector.load %reinterpret_cast_0[%28, %6] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %30 = vector.load %reinterpret_cast_0[%28, %8] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<4xf16>
        %31 = amdgpu.mfma %20 * %7 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %32 = amdgpu.mfma %21 * %9 + %31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %33 = amdgpu.mfma %23 * %7 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %34 = amdgpu.mfma %24 * %9 + %33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %35 = amdgpu.mfma %26 * %7 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %36 = amdgpu.mfma %27 * %9 + %35 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %37 = amdgpu.mfma %29 * %7 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %38 = amdgpu.mfma %30 * %9 + %37 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %39 = amdgpu.mfma %20 * %11 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %40 = amdgpu.mfma %21 * %12 + %39 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %41 = amdgpu.mfma %23 * %11 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %42 = amdgpu.mfma %24 * %12 + %41 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %43 = amdgpu.mfma %26 * %11 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %44 = amdgpu.mfma %27 * %12 + %43 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %45 = amdgpu.mfma %29 * %11 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %46 = amdgpu.mfma %30 * %12 + %45 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %47 = amdgpu.mfma %20 * %14 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %48 = amdgpu.mfma %21 * %15 + %47 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %49 = amdgpu.mfma %23 * %14 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %50 = amdgpu.mfma %24 * %15 + %49 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %51 = amdgpu.mfma %26 * %14 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %52 = amdgpu.mfma %27 * %15 + %51 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %53 = amdgpu.mfma %29 * %14 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %54 = amdgpu.mfma %30 * %15 + %53 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %55 = amdgpu.mfma %20 * %17 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %56 = amdgpu.mfma %21 * %18 + %55 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %57 = amdgpu.mfma %23 * %17 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %58 = amdgpu.mfma %24 * %18 + %57 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %59 = amdgpu.mfma %26 * %17 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %60 = amdgpu.mfma %27 * %18 + %59 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %61 = amdgpu.mfma %29 * %17 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %62 = amdgpu.mfma %30 * %18 + %61 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %63 = affine.apply #map10()[%thread_id_y, %thread_id_x]
        vector.store %32, %reinterpret_cast_1[%c0, %5, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %64 = affine.apply #map11()[%thread_id_y, %thread_id_x]
        vector.store %34, %reinterpret_cast_1[%c0, %5, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %65 = affine.apply #map12()[%thread_id_y, %thread_id_x]
        vector.store %36, %reinterpret_cast_1[%c0, %5, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %66 = affine.apply #map13()[%thread_id_y, %thread_id_x]
        vector.store %38, %reinterpret_cast_1[%c0, %5, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %40, %reinterpret_cast_1[%c0, %10, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %42, %reinterpret_cast_1[%c0, %10, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %44, %reinterpret_cast_1[%c0, %10, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %46, %reinterpret_cast_1[%c0, %10, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %48, %reinterpret_cast_1[%c0, %13, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %50, %reinterpret_cast_1[%c0, %13, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %52, %reinterpret_cast_1[%c0, %13, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %54, %reinterpret_cast_1[%c0, %13, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %56, %reinterpret_cast_1[%c0, %16, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %58, %reinterpret_cast_1[%c0, %16, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %60, %reinterpret_cast_1[%c0, %16, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %62, %reinterpret_cast_1[%c0, %16, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %67 = vector.load %reinterpret_cast_1[%c0, %5, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %68 = vector.load %reinterpret_cast_1[%c0, %5, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %69 = vector.load %reinterpret_cast_1[%c0, %5, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %70 = vector.load %reinterpret_cast_1[%c0, %5, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %71 = vector.load %reinterpret_cast_1[%c0, %10, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %72 = vector.load %reinterpret_cast_1[%c0, %10, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %73 = vector.load %reinterpret_cast_1[%c0, %10, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %74 = vector.load %reinterpret_cast_1[%c0, %10, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %75 = vector.load %reinterpret_cast_1[%c0, %13, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %76 = vector.load %reinterpret_cast_1[%c0, %13, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %77 = vector.load %reinterpret_cast_1[%c0, %13, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %78 = vector.load %reinterpret_cast_1[%c0, %13, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %79 = vector.load %reinterpret_cast_1[%c0, %16, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %80 = vector.load %reinterpret_cast_1[%c0, %16, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %81 = vector.load %reinterpret_cast_1[%c0, %16, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        %82 = vector.load %reinterpret_cast_1[%c0, %16, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %67, %reinterpret_cast_3[%5, %63] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %68, %reinterpret_cast_3[%5, %64] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %69, %reinterpret_cast_3[%5, %65] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %70, %reinterpret_cast_3[%5, %66] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %71, %reinterpret_cast_3[%10, %63] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %72, %reinterpret_cast_3[%10, %64] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %73, %reinterpret_cast_3[%10, %65] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %74, %reinterpret_cast_3[%10, %66] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %75, %reinterpret_cast_3[%13, %63] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %76, %reinterpret_cast_3[%13, %64] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %77, %reinterpret_cast_3[%13, %65] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %78, %reinterpret_cast_3[%13, %66] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %79, %reinterpret_cast_3[%16, %63] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %80, %reinterpret_cast_3[%16, %64] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %81, %reinterpret_cast_3[%16, %65] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %82, %reinterpret_cast_3[%16, %66] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>
        vector.store %67, %reinterpret_cast_2[%c0, %5, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %68, %reinterpret_cast_2[%c0, %5, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %69, %reinterpret_cast_2[%c0, %5, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %70, %reinterpret_cast_2[%c0, %5, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %71, %reinterpret_cast_2[%c0, %10, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %72, %reinterpret_cast_2[%c0, %10, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %73, %reinterpret_cast_2[%c0, %10, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %74, %reinterpret_cast_2[%c0, %10, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %75, %reinterpret_cast_2[%c0, %13, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %76, %reinterpret_cast_2[%c0, %13, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %77, %reinterpret_cast_2[%c0, %13, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %78, %reinterpret_cast_2[%c0, %13, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %79, %reinterpret_cast_2[%c0, %16, %63] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %80, %reinterpret_cast_2[%c0, %16, %64] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %81, %reinterpret_cast_2[%c0, %16, %65] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        vector.store %82, %reinterpret_cast_2[%c0, %16, %66] : memref<1x128x128xf32, strided<[16384, 128, 1], offset: ?>>, vector<4xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) {
    %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<128x32xf16>
    %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<128x32xf16>
    %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<1x128x128xf32>
    %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<1x128x128xf32>
    %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<128x128xf32>
    %5:3 = flow.dispatch @wave_kernel::@wave_kernel(%0, %1, %2, %3, %4) : (tensor<128x32xf16>, tensor<128x32xf16>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>) -> (%2, %3, %4)
    %6:3 = hal.tensor.barrier join(%5#0, %5#1, %5#2 : tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>) => %arg6 : !hal.fence
    %7 = hal.tensor.export %6#0 : tensor<1x128x128xf32> -> !hal.buffer_view
    %8 = hal.tensor.export %6#1 : tensor<1x128x128xf32> -> !hal.buffer_view
    %9 = hal.tensor.export %6#2 : tensor<128x128xf32> -> !hal.buffer_view
    return %7, %8, %9 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
}