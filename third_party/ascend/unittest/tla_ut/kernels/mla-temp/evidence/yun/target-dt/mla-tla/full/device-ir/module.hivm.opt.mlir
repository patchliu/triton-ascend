module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.mode = "tla", hacc.target = #hacc.target<"Ascend950DT_9581">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.aic.c310.O2.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.aiv.c310.O2.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/host.O2.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.mix.aic.c310.O2.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.mix.aiv.c310.O2.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @attention_scope_0(%arg0: memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>, %arg2: memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(16 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(128 : index) : i64
    %9 = llvm.mlir.constant(192 : index) : i64
    %10 = llvm.mlir.constant(256 : index) : i64
    %11 = llvm.mlir.constant(320 : index) : i64
    %12 = llvm.mlir.constant(384 : index) : i64
    %13 = llvm.mlir.constant(448 : index) : i64
    cf.br ^bb1(%3 : i32)
  ^bb1(%14: i32):  // 2 preds: ^bb0, ^bb2
    %15 = arith.cmpi slt, %14, %4 : i32
    cf.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.sext %14 : i32 to i64
    %17 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %20 = llvm.getelementptr %19[%16] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(3 : i32) : i32
    %24 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%20, %21, %23, %22) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %25 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = llvm.mlir.constant(512 : index) : i64
    %29 = llvm.mul %16, %28 : i64
    %30 = llvm.add %29, %6 : i64
    %31 = llvm.getelementptr %27[%30] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %32 = llvm.mlir.constant(0 : i32) : i32
    %33 = llvm.mlir.constant(0 : i32) : i32
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%31, %32, %34, %33) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%36, %37) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %39 = builtin.unrealized_conversion_cast %38 : vector<256xi1> to vector<64xi1>
    %40 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%35, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %41 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.getelementptr %41[%42] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %44 = llvm.mlir.constant(512 : index) : i64
    %45 = llvm.mul %16, %44 : i64
    %46 = llvm.add %45, %7 : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.mlir.constant(0 : i32) : i32
    %51 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%47, %48, %50, %49) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %52 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%51, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = llvm.mlir.constant(1 : i32) : i32
    %55 = llvm.mlir.constant(1 : i32) : i32
    %56 = llvm.mlir.constant(0 : i32) : i32
    %57 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%40, %38, %54, %55, %56) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %58 = llvm.mlir.constant(0 : i32) : i32
    %59 = llvm.mlir.constant(1 : i32) : i32
    %60 = llvm.mlir.constant(1 : i32) : i32
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%52, %38, %59, %60, %61) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %63 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%57, %62) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %64 = llvm.extractvalue %63[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %65 = llvm.extractvalue %63[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %66 = llvm.mlir.constant(0 : i32) : i32
    %67 = llvm.mlir.constant(0 : i32) : i32
    %68 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%66, %67) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %69 = builtin.unrealized_conversion_cast %68 : vector<256xi1> to vector<128xi1>
    %70 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.getelementptr %70[%71] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %73 = llvm.mlir.constant(512 : index) : i64
    %74 = llvm.mul %16, %73 : i64
    %75 = llvm.add %74, %6 : i64
    %76 = llvm.getelementptr %72[%75] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %77 = llvm.mlir.constant(0 : i32) : i32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(1 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%64, %76, %77, %79, %78, %68) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %80 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.getelementptr %80[%81] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %83 = llvm.mlir.constant(512 : index) : i64
    %84 = llvm.mul %16, %83 : i64
    %85 = llvm.add %84, %8 : i64
    %86 = llvm.getelementptr %82[%85] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%86, %87, %89, %88) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %91 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%90, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %92 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.getelementptr %92[%93] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %95 = llvm.mlir.constant(512 : index) : i64
    %96 = llvm.mul %16, %95 : i64
    %97 = llvm.add %96, %9 : i64
    %98 = llvm.getelementptr %94[%97] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.mlir.constant(0 : i32) : i32
    %101 = llvm.mlir.constant(0 : i32) : i32
    %102 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%98, %99, %101, %100) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %103 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%102, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %104 = llvm.mlir.constant(0 : i32) : i32
    %105 = llvm.mlir.constant(1 : i32) : i32
    %106 = llvm.mlir.constant(1 : i32) : i32
    %107 = llvm.mlir.constant(0 : i32) : i32
    %108 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%91, %38, %105, %106, %107) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %109 = llvm.mlir.constant(0 : i32) : i32
    %110 = llvm.mlir.constant(1 : i32) : i32
    %111 = llvm.mlir.constant(1 : i32) : i32
    %112 = llvm.mlir.constant(0 : i32) : i32
    %113 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%103, %38, %110, %111, %112) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %114 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%108, %113) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %115 = llvm.extractvalue %114[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %116 = llvm.extractvalue %114[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %117 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.getelementptr %117[%118] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %120 = llvm.mlir.constant(512 : index) : i64
    %121 = llvm.mul %16, %120 : i64
    %122 = llvm.add %121, %8 : i64
    %123 = llvm.getelementptr %119[%122] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %124 = llvm.mlir.constant(0 : i32) : i32
    %125 = llvm.mlir.constant(0 : i32) : i32
    %126 = llvm.mlir.constant(1 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%115, %123, %124, %126, %125, %68) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %127 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.getelementptr %127[%128] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %130 = llvm.mlir.constant(512 : index) : i64
    %131 = llvm.mul %16, %130 : i64
    %132 = llvm.add %131, %10 : i64
    %133 = llvm.getelementptr %129[%132] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %134 = llvm.mlir.constant(0 : i32) : i32
    %135 = llvm.mlir.constant(0 : i32) : i32
    %136 = llvm.mlir.constant(0 : i32) : i32
    %137 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%133, %134, %136, %135) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %138 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%137, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %139 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.getelementptr %139[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %142 = llvm.mlir.constant(512 : index) : i64
    %143 = llvm.mul %16, %142 : i64
    %144 = llvm.add %143, %11 : i64
    %145 = llvm.getelementptr %141[%144] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %146 = llvm.mlir.constant(0 : i32) : i32
    %147 = llvm.mlir.constant(0 : i32) : i32
    %148 = llvm.mlir.constant(0 : i32) : i32
    %149 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%145, %146, %148, %147) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %150 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%149, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %151 = llvm.mlir.constant(0 : i32) : i32
    %152 = llvm.mlir.constant(1 : i32) : i32
    %153 = llvm.mlir.constant(1 : i32) : i32
    %154 = llvm.mlir.constant(0 : i32) : i32
    %155 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%138, %38, %152, %153, %154) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %156 = llvm.mlir.constant(0 : i32) : i32
    %157 = llvm.mlir.constant(1 : i32) : i32
    %158 = llvm.mlir.constant(1 : i32) : i32
    %159 = llvm.mlir.constant(0 : i32) : i32
    %160 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%150, %38, %157, %158, %159) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %161 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%155, %160) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %162 = llvm.extractvalue %161[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %163 = llvm.extractvalue %161[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %164 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.getelementptr %164[%165] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %167 = llvm.mlir.constant(512 : index) : i64
    %168 = llvm.mul %16, %167 : i64
    %169 = llvm.add %168, %10 : i64
    %170 = llvm.getelementptr %166[%169] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %171 = llvm.mlir.constant(0 : i32) : i32
    %172 = llvm.mlir.constant(0 : i32) : i32
    %173 = llvm.mlir.constant(1 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%162, %170, %171, %173, %172, %68) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %174 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.getelementptr %174[%175] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %177 = llvm.mlir.constant(512 : index) : i64
    %178 = llvm.mul %16, %177 : i64
    %179 = llvm.add %178, %12 : i64
    %180 = llvm.getelementptr %176[%179] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %181 = llvm.mlir.constant(0 : i32) : i32
    %182 = llvm.mlir.constant(0 : i32) : i32
    %183 = llvm.mlir.constant(0 : i32) : i32
    %184 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%180, %181, %183, %182) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %185 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%184, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %186 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %188 = llvm.getelementptr %186[%187] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %189 = llvm.mlir.constant(512 : index) : i64
    %190 = llvm.mul %16, %189 : i64
    %191 = llvm.add %190, %13 : i64
    %192 = llvm.getelementptr %188[%191] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %193 = llvm.mlir.constant(0 : i32) : i32
    %194 = llvm.mlir.constant(0 : i32) : i32
    %195 = llvm.mlir.constant(0 : i32) : i32
    %196 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%192, %193, %195, %194) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %197 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%196, %24, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %198 = llvm.mlir.constant(0 : i32) : i32
    %199 = llvm.mlir.constant(1 : i32) : i32
    %200 = llvm.mlir.constant(1 : i32) : i32
    %201 = llvm.mlir.constant(0 : i32) : i32
    %202 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%185, %38, %199, %200, %201) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %203 = llvm.mlir.constant(0 : i32) : i32
    %204 = llvm.mlir.constant(1 : i32) : i32
    %205 = llvm.mlir.constant(1 : i32) : i32
    %206 = llvm.mlir.constant(0 : i32) : i32
    %207 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%197, %38, %204, %205, %206) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %208 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%202, %207) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %209 = llvm.extractvalue %208[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %210 = llvm.extractvalue %208[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %211 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %212 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %213 = llvm.getelementptr %211[%212] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %214 = llvm.mlir.constant(512 : index) : i64
    %215 = llvm.mul %16, %214 : i64
    %216 = llvm.add %215, %12 : i64
    %217 = llvm.getelementptr %213[%216] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %218 = llvm.mlir.constant(0 : i32) : i32
    %219 = llvm.mlir.constant(0 : i32) : i32
    %220 = llvm.mlir.constant(1 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%209, %217, %218, %220, %219, %68) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %221 = arith.addi %14, %5 overflow<nsw> : i32
    cf.br ^bb1(%221 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_1(%arg0: i32, %arg1: memref<5x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg2: i32, %arg3: memref<2x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>>, %arg4: memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg4 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg3 : memref<2x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg1 : memref<5x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(16 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(128 : index) : i64
    %9 = llvm.mlir.constant(192 : index) : i64
    %10 = llvm.mlir.constant(256 : index) : i64
    %11 = llvm.mlir.constant(320 : index) : i64
    %12 = llvm.mlir.constant(384 : index) : i64
    %13 = llvm.mlir.constant(448 : index) : i64
    cf.br ^bb1(%3 : i32)
  ^bb1(%14: i32):  // 2 preds: ^bb0, ^bb2
    %15 = arith.cmpi slt, %14, %4 : i32
    cf.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.sext %arg0 : i32 to i64
    %17 = llvm.sext %14 : i32 to i64
    %18 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.getelementptr %18[%19] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %21 = llvm.mlir.constant(16 : index) : i64
    %22 = llvm.mul %16, %21 : i64
    %23 = llvm.add %22, %17 : i64
    %24 = llvm.getelementptr %20[%23] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.mlir.constant(3 : i32) : i32
    %28 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%24, %25, %27, %26) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %29 = llvm.sext %arg2 : i32 to i64
    %30 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.getelementptr %30[%31] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %33 = llvm.mlir.constant(8192 : index) : i64
    %34 = llvm.mul %29, %33 : i64
    %35 = llvm.mlir.constant(512 : index) : i64
    %36 = llvm.mul %17, %35 : i64
    %37 = llvm.add %34, %36 : i64
    %38 = llvm.add %37, %6 : i64
    %39 = llvm.getelementptr %32[%38] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %40 = llvm.mlir.constant(0 : i32) : i32
    %41 = llvm.mlir.constant(0 : i32) : i32
    %42 = llvm.mlir.constant(0 : i32) : i32
    %43 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%39, %40, %42, %41) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %44 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.getelementptr %44[%45] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %47 = llvm.mlir.constant(512 : index) : i64
    %48 = llvm.mul %17, %47 : i64
    %49 = llvm.add %48, %6 : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %51 = llvm.mlir.constant(0 : i32) : i32
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%50, %51, %53, %52) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(0 : i32) : i32
    %57 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%55, %56) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %58 = builtin.unrealized_conversion_cast %57 : vector<256xi1> to vector<64xi1>
    %59 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%54, %43, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %60 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %63 = llvm.mlir.constant(512 : index) : i64
    %64 = llvm.mul %17, %63 : i64
    %65 = llvm.add %64, %6 : i64
    %66 = llvm.getelementptr %62[%65] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %67 = llvm.mlir.constant(0 : i32) : i32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%59, %66, %67, %69, %68, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %70 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %72 = llvm.getelementptr %70[%71] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %73 = llvm.mlir.constant(8192 : index) : i64
    %74 = llvm.mul %29, %73 : i64
    %75 = llvm.mlir.constant(512 : index) : i64
    %76 = llvm.mul %17, %75 : i64
    %77 = llvm.add %74, %76 : i64
    %78 = llvm.add %77, %7 : i64
    %79 = llvm.getelementptr %72[%78] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %80 = llvm.mlir.constant(0 : i32) : i32
    %81 = llvm.mlir.constant(0 : i32) : i32
    %82 = llvm.mlir.constant(0 : i32) : i32
    %83 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%79, %80, %82, %81) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %84 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.getelementptr %84[%85] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %87 = llvm.mlir.constant(512 : index) : i64
    %88 = llvm.mul %17, %87 : i64
    %89 = llvm.add %88, %7 : i64
    %90 = llvm.getelementptr %86[%89] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %91 = llvm.mlir.constant(0 : i32) : i32
    %92 = llvm.mlir.constant(0 : i32) : i32
    %93 = llvm.mlir.constant(0 : i32) : i32
    %94 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%90, %91, %93, %92) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %95 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%94, %83, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %96 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.getelementptr %96[%97] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %99 = llvm.mlir.constant(512 : index) : i64
    %100 = llvm.mul %17, %99 : i64
    %101 = llvm.add %100, %7 : i64
    %102 = llvm.getelementptr %98[%101] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %103 = llvm.mlir.constant(0 : i32) : i32
    %104 = llvm.mlir.constant(0 : i32) : i32
    %105 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%95, %102, %103, %105, %104, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %106 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %107 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %108 = llvm.getelementptr %106[%107] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %109 = llvm.mlir.constant(8192 : index) : i64
    %110 = llvm.mul %29, %109 : i64
    %111 = llvm.mlir.constant(512 : index) : i64
    %112 = llvm.mul %17, %111 : i64
    %113 = llvm.add %110, %112 : i64
    %114 = llvm.add %113, %8 : i64
    %115 = llvm.getelementptr %108[%114] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %116 = llvm.mlir.constant(0 : i32) : i32
    %117 = llvm.mlir.constant(0 : i32) : i32
    %118 = llvm.mlir.constant(0 : i32) : i32
    %119 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%115, %116, %118, %117) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %120 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.getelementptr %120[%121] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %123 = llvm.mlir.constant(512 : index) : i64
    %124 = llvm.mul %17, %123 : i64
    %125 = llvm.add %124, %8 : i64
    %126 = llvm.getelementptr %122[%125] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %127 = llvm.mlir.constant(0 : i32) : i32
    %128 = llvm.mlir.constant(0 : i32) : i32
    %129 = llvm.mlir.constant(0 : i32) : i32
    %130 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%126, %127, %129, %128) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %131 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%130, %119, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %132 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.getelementptr %132[%133] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %135 = llvm.mlir.constant(512 : index) : i64
    %136 = llvm.mul %17, %135 : i64
    %137 = llvm.add %136, %8 : i64
    %138 = llvm.getelementptr %134[%137] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %139 = llvm.mlir.constant(0 : i32) : i32
    %140 = llvm.mlir.constant(0 : i32) : i32
    %141 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%131, %138, %139, %141, %140, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %142 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %143 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %144 = llvm.getelementptr %142[%143] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %145 = llvm.mlir.constant(8192 : index) : i64
    %146 = llvm.mul %29, %145 : i64
    %147 = llvm.mlir.constant(512 : index) : i64
    %148 = llvm.mul %17, %147 : i64
    %149 = llvm.add %146, %148 : i64
    %150 = llvm.add %149, %9 : i64
    %151 = llvm.getelementptr %144[%150] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %152 = llvm.mlir.constant(0 : i32) : i32
    %153 = llvm.mlir.constant(0 : i32) : i32
    %154 = llvm.mlir.constant(0 : i32) : i32
    %155 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%151, %152, %154, %153) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %156 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.getelementptr %156[%157] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %159 = llvm.mlir.constant(512 : index) : i64
    %160 = llvm.mul %17, %159 : i64
    %161 = llvm.add %160, %9 : i64
    %162 = llvm.getelementptr %158[%161] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %163 = llvm.mlir.constant(0 : i32) : i32
    %164 = llvm.mlir.constant(0 : i32) : i32
    %165 = llvm.mlir.constant(0 : i32) : i32
    %166 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%162, %163, %165, %164) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %167 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%166, %155, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %168 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %170 = llvm.getelementptr %168[%169] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %171 = llvm.mlir.constant(512 : index) : i64
    %172 = llvm.mul %17, %171 : i64
    %173 = llvm.add %172, %9 : i64
    %174 = llvm.getelementptr %170[%173] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %175 = llvm.mlir.constant(0 : i32) : i32
    %176 = llvm.mlir.constant(0 : i32) : i32
    %177 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%167, %174, %175, %177, %176, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %178 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %179 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %180 = llvm.getelementptr %178[%179] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %181 = llvm.mlir.constant(8192 : index) : i64
    %182 = llvm.mul %29, %181 : i64
    %183 = llvm.mlir.constant(512 : index) : i64
    %184 = llvm.mul %17, %183 : i64
    %185 = llvm.add %182, %184 : i64
    %186 = llvm.add %185, %10 : i64
    %187 = llvm.getelementptr %180[%186] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %188 = llvm.mlir.constant(0 : i32) : i32
    %189 = llvm.mlir.constant(0 : i32) : i32
    %190 = llvm.mlir.constant(0 : i32) : i32
    %191 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%187, %188, %190, %189) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %192 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %193 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %194 = llvm.getelementptr %192[%193] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %195 = llvm.mlir.constant(512 : index) : i64
    %196 = llvm.mul %17, %195 : i64
    %197 = llvm.add %196, %10 : i64
    %198 = llvm.getelementptr %194[%197] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %199 = llvm.mlir.constant(0 : i32) : i32
    %200 = llvm.mlir.constant(0 : i32) : i32
    %201 = llvm.mlir.constant(0 : i32) : i32
    %202 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%198, %199, %201, %200) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %203 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%202, %191, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %204 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %205 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.getelementptr %204[%205] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %207 = llvm.mlir.constant(512 : index) : i64
    %208 = llvm.mul %17, %207 : i64
    %209 = llvm.add %208, %10 : i64
    %210 = llvm.getelementptr %206[%209] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %211 = llvm.mlir.constant(0 : i32) : i32
    %212 = llvm.mlir.constant(0 : i32) : i32
    %213 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%203, %210, %211, %213, %212, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %214 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %215 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %216 = llvm.getelementptr %214[%215] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %217 = llvm.mlir.constant(8192 : index) : i64
    %218 = llvm.mul %29, %217 : i64
    %219 = llvm.mlir.constant(512 : index) : i64
    %220 = llvm.mul %17, %219 : i64
    %221 = llvm.add %218, %220 : i64
    %222 = llvm.add %221, %11 : i64
    %223 = llvm.getelementptr %216[%222] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %224 = llvm.mlir.constant(0 : i32) : i32
    %225 = llvm.mlir.constant(0 : i32) : i32
    %226 = llvm.mlir.constant(0 : i32) : i32
    %227 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%223, %224, %226, %225) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %228 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.getelementptr %228[%229] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %231 = llvm.mlir.constant(512 : index) : i64
    %232 = llvm.mul %17, %231 : i64
    %233 = llvm.add %232, %11 : i64
    %234 = llvm.getelementptr %230[%233] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %235 = llvm.mlir.constant(0 : i32) : i32
    %236 = llvm.mlir.constant(0 : i32) : i32
    %237 = llvm.mlir.constant(0 : i32) : i32
    %238 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%234, %235, %237, %236) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %239 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%238, %227, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %240 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %241 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.getelementptr %240[%241] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %243 = llvm.mlir.constant(512 : index) : i64
    %244 = llvm.mul %17, %243 : i64
    %245 = llvm.add %244, %11 : i64
    %246 = llvm.getelementptr %242[%245] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %247 = llvm.mlir.constant(0 : i32) : i32
    %248 = llvm.mlir.constant(0 : i32) : i32
    %249 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%239, %246, %247, %249, %248, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %250 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %251 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %252 = llvm.getelementptr %250[%251] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %253 = llvm.mlir.constant(8192 : index) : i64
    %254 = llvm.mul %29, %253 : i64
    %255 = llvm.mlir.constant(512 : index) : i64
    %256 = llvm.mul %17, %255 : i64
    %257 = llvm.add %254, %256 : i64
    %258 = llvm.add %257, %12 : i64
    %259 = llvm.getelementptr %252[%258] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %260 = llvm.mlir.constant(0 : i32) : i32
    %261 = llvm.mlir.constant(0 : i32) : i32
    %262 = llvm.mlir.constant(0 : i32) : i32
    %263 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%259, %260, %262, %261) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %264 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %265 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %266 = llvm.getelementptr %264[%265] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %267 = llvm.mlir.constant(512 : index) : i64
    %268 = llvm.mul %17, %267 : i64
    %269 = llvm.add %268, %12 : i64
    %270 = llvm.getelementptr %266[%269] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %271 = llvm.mlir.constant(0 : i32) : i32
    %272 = llvm.mlir.constant(0 : i32) : i32
    %273 = llvm.mlir.constant(0 : i32) : i32
    %274 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%270, %271, %273, %272) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %275 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%274, %263, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %276 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %277 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %278 = llvm.getelementptr %276[%277] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %279 = llvm.mlir.constant(512 : index) : i64
    %280 = llvm.mul %17, %279 : i64
    %281 = llvm.add %280, %12 : i64
    %282 = llvm.getelementptr %278[%281] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %283 = llvm.mlir.constant(0 : i32) : i32
    %284 = llvm.mlir.constant(0 : i32) : i32
    %285 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%275, %282, %283, %285, %284, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %286 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %287 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<3 x i64>, array<3 x i64>)> 
    %288 = llvm.getelementptr %286[%287] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %289 = llvm.mlir.constant(8192 : index) : i64
    %290 = llvm.mul %29, %289 : i64
    %291 = llvm.mlir.constant(512 : index) : i64
    %292 = llvm.mul %17, %291 : i64
    %293 = llvm.add %290, %292 : i64
    %294 = llvm.add %293, %13 : i64
    %295 = llvm.getelementptr %288[%294] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %296 = llvm.mlir.constant(0 : i32) : i32
    %297 = llvm.mlir.constant(0 : i32) : i32
    %298 = llvm.mlir.constant(0 : i32) : i32
    %299 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%295, %296, %298, %297) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %300 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %301 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %302 = llvm.getelementptr %300[%301] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %303 = llvm.mlir.constant(512 : index) : i64
    %304 = llvm.mul %17, %303 : i64
    %305 = llvm.add %304, %13 : i64
    %306 = llvm.getelementptr %302[%305] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %307 = llvm.mlir.constant(0 : i32) : i32
    %308 = llvm.mlir.constant(0 : i32) : i32
    %309 = llvm.mlir.constant(0 : i32) : i32
    %310 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%306, %307, %309, %308) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %311 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%310, %299, %28, %57) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %312 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %313 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %314 = llvm.getelementptr %312[%313] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %315 = llvm.mlir.constant(512 : index) : i64
    %316 = llvm.mul %17, %315 : i64
    %317 = llvm.add %316, %13 : i64
    %318 = llvm.getelementptr %314[%317] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %319 = llvm.mlir.constant(0 : i32) : i32
    %320 = llvm.mlir.constant(0 : i32) : i32
    %321 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%311, %318, %319, %321, %320, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %322 = arith.addi %14, %5 overflow<nsw> : i32
    cf.br ^bb1(%322 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_2(%arg0: memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(16 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mlir.constant(192 : index) : i64
    %9 = llvm.mlir.constant(256 : index) : i64
    %10 = llvm.mlir.constant(320 : index) : i64
    %11 = llvm.mlir.constant(384 : index) : i64
    %12 = llvm.mlir.constant(448 : index) : i64
    %13 = "hivm_regbaseintrins.intr.hivm.vbr"(%1) : (f32) -> vector<64xf32>
    cf.br ^bb1(%2 : i32)
  ^bb1(%14: i32):  // 2 preds: ^bb0, ^bb2
    %15 = arith.cmpi slt, %14, %3 : i32
    cf.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%16, %17) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %19 = builtin.unrealized_conversion_cast %18 : vector<256xi1> to vector<64xi1>
    %20 = llvm.sext %14 : i32 to i64
    %21 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %24 = llvm.mlir.constant(512 : index) : i64
    %25 = llvm.mul %20, %24 : i64
    %26 = llvm.add %25, %5 : i64
    %27 = llvm.getelementptr %23[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %27, %28, %30, %29, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %31 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.getelementptr %31[%32] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %34 = llvm.mlir.constant(512 : index) : i64
    %35 = llvm.mul %20, %34 : i64
    %36 = llvm.add %35, %6 : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %37, %38, %40, %39, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %41 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.getelementptr %41[%42] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %44 = llvm.mlir.constant(512 : index) : i64
    %45 = llvm.mul %20, %44 : i64
    %46 = llvm.add %45, %7 : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %47, %48, %50, %49, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %51 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %54 = llvm.mlir.constant(512 : index) : i64
    %55 = llvm.mul %20, %54 : i64
    %56 = llvm.add %55, %8 : i64
    %57 = llvm.getelementptr %53[%56] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %58 = llvm.mlir.constant(0 : i32) : i32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %57, %58, %60, %59, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %61 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.getelementptr %61[%62] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %64 = llvm.mlir.constant(512 : index) : i64
    %65 = llvm.mul %20, %64 : i64
    %66 = llvm.add %65, %9 : i64
    %67 = llvm.getelementptr %63[%66] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %67, %68, %70, %69, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %71 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.getelementptr %71[%72] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %74 = llvm.mlir.constant(512 : index) : i64
    %75 = llvm.mul %20, %74 : i64
    %76 = llvm.add %75, %10 : i64
    %77 = llvm.getelementptr %73[%76] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(0 : i32) : i32
    %80 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %77, %78, %80, %79, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %81 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.getelementptr %81[%82] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %84 = llvm.mlir.constant(512 : index) : i64
    %85 = llvm.mul %20, %84 : i64
    %86 = llvm.add %85, %11 : i64
    %87 = llvm.getelementptr %83[%86] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %87, %88, %90, %89, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %91 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.getelementptr %91[%92] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %94 = llvm.mlir.constant(512 : index) : i64
    %95 = llvm.mul %20, %94 : i64
    %96 = llvm.add %95, %12 : i64
    %97 = llvm.getelementptr %93[%96] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %98 = llvm.mlir.constant(0 : i32) : i32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %97, %98, %100, %99, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %101 = arith.addi %14, %4 overflow<nsw> : i32
    cf.br ^bb1(%101 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_3(%arg0: i32, %arg1: memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg2: memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg3: memref<5x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg4: memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg5: memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg5 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg4 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg3 : memref<5x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = builtin.unrealized_conversion_cast %arg2 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = builtin.unrealized_conversion_cast %arg1 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(16 : i32) : i32
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = "hivm_regbaseintrins.intr.hivm.vbr"(%5) : (f32) -> vector<64xf32>
    %13 = "hivm_regbaseintrins.intr.hivm.vbr"(%6) : (f32) -> vector<64xf32>
    %14 = llvm.mlir.constant(6 : i32) : i32
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%14, %15) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %17 = builtin.unrealized_conversion_cast %16 : vector<256xi1> to vector<64xi1>
    %18 = "hivm_regbaseintrins.intr.hivm.vbr"(%7) : (f32) -> vector<64xf32>
    cf.br ^bb1(%8, %18 : i32, vector<64xf32>)
  ^bb1(%19: i32, %20: vector<64xf32>):  // 2 preds: ^bb0, ^bb2
    %21 = arith.cmpi slt, %19, %arg0 : i32
    cf.cond_br %21, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %22 = llvm.mul %19, %10 : i32
    %23 = llvm.sext %22 : i32 to i64
    %24 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.getelementptr %24[%25] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %27 = llvm.getelementptr %26[%23] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%27, %28, %30, %29) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %32 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%20, %31, %16) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %33 = arith.addi %19, %9 overflow<nsw> : i32
    cf.br ^bb1(%33, %32 : i32, vector<64xf32>)
  ^bb3:  // pred: ^bb1
    cf.br ^bb4(%8, %12 : i32, vector<64xf32>)
  ^bb4(%34: i32, %35: vector<64xf32>):  // 2 preds: ^bb3, ^bb5
    %36 = arith.cmpi slt, %34, %arg0 : i32
    cf.cond_br %36, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %37 = llvm.mul %34, %10 : i32
    %38 = llvm.sext %37 : i32 to i64
    %39 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %39[%40] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %42 = llvm.getelementptr %41[%38] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %43 = llvm.mlir.constant(0 : i32) : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.mlir.constant(0 : i32) : i32
    %46 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%42, %43, %45, %44) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %47 = llvm.mlir.constant(0 : i32) : i32
    %48 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%46, %20, %16, %47) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %49 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %50 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %52 = llvm.getelementptr %51[%38] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = llvm.mlir.constant(0 : i32) : i32
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%52, %53, %55, %54) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %57 = llvm.mlir.constant(0 : i32) : i32
    %58 = llvm.mlir.constant(0 : i32) : i32
    %59 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%57, %58) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %60 = builtin.unrealized_conversion_cast %59 : vector<256xi1> to vector<64xi1>
    %61 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%35, %56, %48, %59) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %62 = llvm.sext %34 : i32 to i64
    %63 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.getelementptr %63[%64] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %66 = llvm.mlir.constant(16 : index) : i64
    %67 = llvm.mul %62, %66 : i64
    %68 = llvm.add %67, %11 : i64
    %69 = llvm.getelementptr %65[%68] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%48, %69, %70, %72, %71, %16) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %73 = arith.addi %34, %9 overflow<nsw> : i32
    cf.br ^bb4(%73, %61 : i32, vector<64xf32>)
  ^bb6:  // pred: ^bb4
    %74 = "hivm_regbaseintrins.intr.hivm.vdiv.s.x"(%13, %35, %16) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %75 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.getelementptr %75[%76] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(0 : i32) : i32
    %80 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%74, %77, %78, %80, %79, %16) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %81 = "hivm_regbaseintrins.intr.hivm.vln.x"(%35, %16) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %82 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%20, %81, %16) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %83 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.getelementptr %83[%84] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %86 = llvm.mlir.constant(0 : i32) : i32
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%82, %85, %86, %88, %87, %16) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    "hivm_regbaseintrins.intr.hivm.mem.bar.vst.vld"() : () -> ()
    return
  }
  func.func @attention_scope_4(%arg0: memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(64 : index) : i64
    %4 = "hivm_regbaseintrins.intr.hivm.vbr"(%2) : (f32) -> vector<64xf32>
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %6) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %8 = builtin.unrealized_conversion_cast %7 : vector<256xi1> to vector<64xi1>
    %9 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.getelementptr %9[%10] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %12 = llvm.getelementptr %11[64] : (!llvm.ptr<6>) -> !llvm.ptr<6>, f32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%4, %12, %13, %15, %14, %7) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %16 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.getelementptr %16[%17] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %19 = llvm.getelementptr %18[64] : (!llvm.ptr<6>) -> !llvm.ptr<6>, f32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%4, %19, %20, %22, %21, %7) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @attention_scope_5(%arg0: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, %arg2: memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(48 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(256 : index) : i64
    %8 = llvm.mlir.constant(64 : index) : i64
    %9 = llvm.mlir.constant(320 : index) : i64
    %10 = llvm.mlir.constant(128 : index) : i64
    %11 = llvm.mlir.constant(384 : index) : i64
    %12 = llvm.mlir.constant(192 : index) : i64
    %13 = llvm.mlir.constant(448 : index) : i64
    cf.br ^bb1(%3 : i32)
  ^bb1(%14: i32):  // 2 preds: ^bb0, ^bb2
    %15 = arith.cmpi slt, %14, %4 : i32
    cf.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.sext %14 : i32 to i64
    %17 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %20 = llvm.getelementptr %19[%16] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(3 : i32) : i32
    %24 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%20, %21, %23, %22) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %25 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = llvm.mlir.constant(256 : index) : i64
    %29 = llvm.mul %16, %28 : i64
    %30 = llvm.add %29, %6 : i64
    %31 = llvm.getelementptr %27[%30] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %32 = llvm.mlir.constant(0 : i32) : i32
    %33 = llvm.mlir.constant(0 : i32) : i32
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%31, %32, %34, %33) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %36 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.getelementptr %36[%37] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %39 = llvm.mlir.constant(512 : index) : i64
    %40 = llvm.mul %16, %39 : i64
    %41 = llvm.add %40, %7 : i64
    %42 = llvm.getelementptr %38[%41] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %43 = llvm.mlir.constant(0 : i32) : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.mlir.constant(0 : i32) : i32
    %46 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%42, %43, %45, %44) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %47 = llvm.mlir.constant(0 : i32) : i32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%47, %48) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %50 = builtin.unrealized_conversion_cast %49 : vector<256xi1> to vector<64xi1>
    %51 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%35, %46, %24, %49) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %52 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.getelementptr %52[%53] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %55 = llvm.mlir.constant(512 : index) : i64
    %56 = llvm.mul %16, %55 : i64
    %57 = llvm.add %56, %7 : i64
    %58 = llvm.getelementptr %54[%57] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.mlir.constant(0 : i32) : i32
    %61 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%51, %58, %59, %61, %60, %49) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %62 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.getelementptr %62[%63] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %65 = llvm.mlir.constant(256 : index) : i64
    %66 = llvm.mul %16, %65 : i64
    %67 = llvm.add %66, %8 : i64
    %68 = llvm.getelementptr %64[%67] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%68, %69, %71, %70) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %73 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.getelementptr %73[%74] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %76 = llvm.mlir.constant(512 : index) : i64
    %77 = llvm.mul %16, %76 : i64
    %78 = llvm.add %77, %9 : i64
    %79 = llvm.getelementptr %75[%78] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %80 = llvm.mlir.constant(0 : i32) : i32
    %81 = llvm.mlir.constant(0 : i32) : i32
    %82 = llvm.mlir.constant(0 : i32) : i32
    %83 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%79, %80, %82, %81) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %84 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%72, %83, %24, %49) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %85 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.getelementptr %85[%86] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %88 = llvm.mlir.constant(512 : index) : i64
    %89 = llvm.mul %16, %88 : i64
    %90 = llvm.add %89, %9 : i64
    %91 = llvm.getelementptr %87[%90] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %92 = llvm.mlir.constant(0 : i32) : i32
    %93 = llvm.mlir.constant(0 : i32) : i32
    %94 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%84, %91, %92, %94, %93, %49) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %95 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.getelementptr %95[%96] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %98 = llvm.mlir.constant(256 : index) : i64
    %99 = llvm.mul %16, %98 : i64
    %100 = llvm.add %99, %10 : i64
    %101 = llvm.getelementptr %97[%100] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.mlir.constant(0 : i32) : i32
    %104 = llvm.mlir.constant(0 : i32) : i32
    %105 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%101, %102, %104, %103) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %106 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.getelementptr %106[%107] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %109 = llvm.mlir.constant(512 : index) : i64
    %110 = llvm.mul %16, %109 : i64
    %111 = llvm.add %110, %11 : i64
    %112 = llvm.getelementptr %108[%111] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %113 = llvm.mlir.constant(0 : i32) : i32
    %114 = llvm.mlir.constant(0 : i32) : i32
    %115 = llvm.mlir.constant(0 : i32) : i32
    %116 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%112, %113, %115, %114) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %117 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%105, %116, %24, %49) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %118 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.getelementptr %118[%119] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %121 = llvm.mlir.constant(512 : index) : i64
    %122 = llvm.mul %16, %121 : i64
    %123 = llvm.add %122, %11 : i64
    %124 = llvm.getelementptr %120[%123] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %125 = llvm.mlir.constant(0 : i32) : i32
    %126 = llvm.mlir.constant(0 : i32) : i32
    %127 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%117, %124, %125, %127, %126, %49) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %128 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.getelementptr %128[%129] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %131 = llvm.mlir.constant(256 : index) : i64
    %132 = llvm.mul %16, %131 : i64
    %133 = llvm.add %132, %12 : i64
    %134 = llvm.getelementptr %130[%133] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %135 = llvm.mlir.constant(0 : i32) : i32
    %136 = llvm.mlir.constant(0 : i32) : i32
    %137 = llvm.mlir.constant(0 : i32) : i32
    %138 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%134, %135, %137, %136) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %139 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.getelementptr %139[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %142 = llvm.mlir.constant(512 : index) : i64
    %143 = llvm.mul %16, %142 : i64
    %144 = llvm.add %143, %13 : i64
    %145 = llvm.getelementptr %141[%144] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %146 = llvm.mlir.constant(0 : i32) : i32
    %147 = llvm.mlir.constant(0 : i32) : i32
    %148 = llvm.mlir.constant(0 : i32) : i32
    %149 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%145, %146, %148, %147) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %150 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%138, %149, %24, %49) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %151 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.getelementptr %151[%152] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %154 = llvm.mlir.constant(512 : index) : i64
    %155 = llvm.mul %16, %154 : i64
    %156 = llvm.add %155, %13 : i64
    %157 = llvm.getelementptr %153[%156] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %158 = llvm.mlir.constant(0 : i32) : i32
    %159 = llvm.mlir.constant(0 : i32) : i32
    %160 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%150, %157, %158, %160, %159, %49) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %161 = arith.addi %14, %5 overflow<nsw> : i32
    cf.br ^bb1(%161 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_6(%arg0: memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(48 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(256 : index) : i64
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(320 : index) : i64
    %9 = llvm.mlir.constant(128 : index) : i64
    %10 = llvm.mlir.constant(384 : index) : i64
    %11 = llvm.mlir.constant(192 : index) : i64
    %12 = llvm.mlir.constant(448 : index) : i64
    cf.br ^bb1(%2 : i32)
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb2
    %14 = arith.cmpi slt, %13, %3 : i32
    cf.cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %15 = llvm.sext %13 : i32 to i64
    %16 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.getelementptr %16[%17] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %19 = llvm.mlir.constant(256 : index) : i64
    %20 = llvm.mul %15, %19 : i64
    %21 = llvm.add %20, %5 : i64
    %22 = llvm.getelementptr %18[%21] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%22, %23, %25, %24) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%27, %28) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %30 = builtin.unrealized_conversion_cast %29 : vector<256xi1> to vector<64xi1>
    %31 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.getelementptr %31[%32] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %34 = llvm.mlir.constant(512 : index) : i64
    %35 = llvm.mul %15, %34 : i64
    %36 = llvm.add %35, %6 : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%26, %37, %38, %40, %39, %29) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %41 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.getelementptr %41[%42] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %44 = llvm.mlir.constant(256 : index) : i64
    %45 = llvm.mul %15, %44 : i64
    %46 = llvm.add %45, %7 : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.mlir.constant(0 : i32) : i32
    %51 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%47, %48, %50, %49) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %52 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.getelementptr %52[%53] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %55 = llvm.mlir.constant(512 : index) : i64
    %56 = llvm.mul %15, %55 : i64
    %57 = llvm.add %56, %8 : i64
    %58 = llvm.getelementptr %54[%57] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.mlir.constant(0 : i32) : i32
    %61 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%51, %58, %59, %61, %60, %29) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %62 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.getelementptr %62[%63] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %65 = llvm.mlir.constant(256 : index) : i64
    %66 = llvm.mul %15, %65 : i64
    %67 = llvm.add %66, %9 : i64
    %68 = llvm.getelementptr %64[%67] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%68, %69, %71, %70) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %73 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.getelementptr %73[%74] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %76 = llvm.mlir.constant(512 : index) : i64
    %77 = llvm.mul %15, %76 : i64
    %78 = llvm.add %77, %10 : i64
    %79 = llvm.getelementptr %75[%78] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %80 = llvm.mlir.constant(0 : i32) : i32
    %81 = llvm.mlir.constant(0 : i32) : i32
    %82 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%72, %79, %80, %82, %81, %29) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %83 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.getelementptr %83[%84] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %86 = llvm.mlir.constant(256 : index) : i64
    %87 = llvm.mul %15, %86 : i64
    %88 = llvm.add %87, %11 : i64
    %89 = llvm.getelementptr %85[%88] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %90 = llvm.mlir.constant(0 : i32) : i32
    %91 = llvm.mlir.constant(0 : i32) : i32
    %92 = llvm.mlir.constant(0 : i32) : i32
    %93 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%89, %90, %92, %91) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %94 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.getelementptr %94[%95] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %97 = llvm.mlir.constant(512 : index) : i64
    %98 = llvm.mul %15, %97 : i64
    %99 = llvm.add %98, %12 : i64
    %100 = llvm.getelementptr %96[%99] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %101 = llvm.mlir.constant(0 : i32) : i32
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%93, %100, %101, %103, %102, %29) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %104 = arith.addi %13, %4 overflow<nsw> : i32
    cf.br ^bb1(%104 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_7(%arg0: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, %arg2: memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(48 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(128 : index) : i64
    %9 = llvm.mlir.constant(192 : index) : i64
    cf.br ^bb1(%3 : i32)
  ^bb1(%10: i32):  // 2 preds: ^bb0, ^bb2
    %11 = arith.cmpi slt, %10, %4 : i32
    cf.cond_br %11, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %12 = llvm.sext %10 : i32 to i64
    %13 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %13[%14] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %16 = llvm.getelementptr %15[%12] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(3 : i32) : i32
    %20 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%16, %17, %19, %18) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %21 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %24 = llvm.mlir.constant(256 : index) : i64
    %25 = llvm.mul %12, %24 : i64
    %26 = llvm.add %25, %6 : i64
    %27 = llvm.getelementptr %23[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%27, %28, %30, %29) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %32 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %35 = llvm.mlir.constant(512 : index) : i64
    %36 = llvm.mul %12, %35 : i64
    %37 = llvm.add %36, %6 : i64
    %38 = llvm.getelementptr %34[%37] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.mlir.constant(0 : i32) : i32
    %41 = llvm.mlir.constant(0 : i32) : i32
    %42 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%38, %39, %41, %40) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %43 = llvm.mlir.constant(0 : i32) : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%43, %44) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %46 = builtin.unrealized_conversion_cast %45 : vector<256xi1> to vector<64xi1>
    %47 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%31, %42, %20, %45) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %48 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.getelementptr %48[%49] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %51 = llvm.mlir.constant(512 : index) : i64
    %52 = llvm.mul %12, %51 : i64
    %53 = llvm.add %52, %6 : i64
    %54 = llvm.getelementptr %50[%53] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(0 : i32) : i32
    %57 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%47, %54, %55, %57, %56, %45) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %58 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.getelementptr %58[%59] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %61 = llvm.mlir.constant(256 : index) : i64
    %62 = llvm.mul %12, %61 : i64
    %63 = llvm.add %62, %7 : i64
    %64 = llvm.getelementptr %60[%63] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %65 = llvm.mlir.constant(0 : i32) : i32
    %66 = llvm.mlir.constant(0 : i32) : i32
    %67 = llvm.mlir.constant(0 : i32) : i32
    %68 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%64, %65, %67, %66) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %69 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.getelementptr %69[%70] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %72 = llvm.mlir.constant(512 : index) : i64
    %73 = llvm.mul %12, %72 : i64
    %74 = llvm.add %73, %7 : i64
    %75 = llvm.getelementptr %71[%74] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %76 = llvm.mlir.constant(0 : i32) : i32
    %77 = llvm.mlir.constant(0 : i32) : i32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%75, %76, %78, %77) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %80 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%68, %79, %20, %45) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %81 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.getelementptr %81[%82] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %84 = llvm.mlir.constant(512 : index) : i64
    %85 = llvm.mul %12, %84 : i64
    %86 = llvm.add %85, %7 : i64
    %87 = llvm.getelementptr %83[%86] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%80, %87, %88, %90, %89, %45) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %91 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.getelementptr %91[%92] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %94 = llvm.mlir.constant(256 : index) : i64
    %95 = llvm.mul %12, %94 : i64
    %96 = llvm.add %95, %8 : i64
    %97 = llvm.getelementptr %93[%96] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %98 = llvm.mlir.constant(0 : i32) : i32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.mlir.constant(0 : i32) : i32
    %101 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%97, %98, %100, %99) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %102 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.getelementptr %102[%103] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %105 = llvm.mlir.constant(512 : index) : i64
    %106 = llvm.mul %12, %105 : i64
    %107 = llvm.add %106, %8 : i64
    %108 = llvm.getelementptr %104[%107] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %109 = llvm.mlir.constant(0 : i32) : i32
    %110 = llvm.mlir.constant(0 : i32) : i32
    %111 = llvm.mlir.constant(0 : i32) : i32
    %112 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%108, %109, %111, %110) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %113 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%101, %112, %20, %45) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %114 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.getelementptr %114[%115] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %117 = llvm.mlir.constant(512 : index) : i64
    %118 = llvm.mul %12, %117 : i64
    %119 = llvm.add %118, %8 : i64
    %120 = llvm.getelementptr %116[%119] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %121 = llvm.mlir.constant(0 : i32) : i32
    %122 = llvm.mlir.constant(0 : i32) : i32
    %123 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%113, %120, %121, %123, %122, %45) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %124 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.getelementptr %124[%125] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %127 = llvm.mlir.constant(256 : index) : i64
    %128 = llvm.mul %12, %127 : i64
    %129 = llvm.add %128, %9 : i64
    %130 = llvm.getelementptr %126[%129] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %131 = llvm.mlir.constant(0 : i32) : i32
    %132 = llvm.mlir.constant(0 : i32) : i32
    %133 = llvm.mlir.constant(0 : i32) : i32
    %134 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%130, %131, %133, %132) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %135 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.getelementptr %135[%136] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %138 = llvm.mlir.constant(512 : index) : i64
    %139 = llvm.mul %12, %138 : i64
    %140 = llvm.add %139, %9 : i64
    %141 = llvm.getelementptr %137[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %142 = llvm.mlir.constant(0 : i32) : i32
    %143 = llvm.mlir.constant(0 : i32) : i32
    %144 = llvm.mlir.constant(0 : i32) : i32
    %145 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%141, %142, %144, %143) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %146 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%134, %145, %20, %45) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %147 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.getelementptr %147[%148] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %150 = llvm.mlir.constant(512 : index) : i64
    %151 = llvm.mul %12, %150 : i64
    %152 = llvm.add %151, %9 : i64
    %153 = llvm.getelementptr %149[%152] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %154 = llvm.mlir.constant(0 : i32) : i32
    %155 = llvm.mlir.constant(0 : i32) : i32
    %156 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%146, %153, %154, %156, %155, %45) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %157 = arith.addi %10, %5 overflow<nsw> : i32
    cf.br ^bb1(%157 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_8(%arg0: memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(48 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mlir.constant(192 : index) : i64
    cf.br ^bb1(%2 : i32)
  ^bb1(%9: i32):  // 2 preds: ^bb0, ^bb2
    %10 = arith.cmpi slt, %9, %3 : i32
    cf.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = llvm.sext %9 : i32 to i64
    %12 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.getelementptr %12[%13] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %15 = llvm.mlir.constant(256 : index) : i64
    %16 = llvm.mul %11, %15 : i64
    %17 = llvm.add %16, %5 : i64
    %18 = llvm.getelementptr %14[%17] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%18, %19, %21, %20) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%23, %24) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %26 = builtin.unrealized_conversion_cast %25 : vector<256xi1> to vector<64xi1>
    %27 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %30 = llvm.mlir.constant(512 : index) : i64
    %31 = llvm.mul %11, %30 : i64
    %32 = llvm.add %31, %5 : i64
    %33 = llvm.getelementptr %29[%32] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = llvm.mlir.constant(0 : i32) : i32
    %36 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%22, %33, %34, %36, %35, %25) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %37 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.getelementptr %37[%38] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %40 = llvm.mlir.constant(256 : index) : i64
    %41 = llvm.mul %11, %40 : i64
    %42 = llvm.add %41, %6 : i64
    %43 = llvm.getelementptr %39[%42] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.mlir.constant(0 : i32) : i32
    %46 = llvm.mlir.constant(0 : i32) : i32
    %47 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%43, %44, %46, %45) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %48 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.getelementptr %48[%49] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %51 = llvm.mlir.constant(512 : index) : i64
    %52 = llvm.mul %11, %51 : i64
    %53 = llvm.add %52, %6 : i64
    %54 = llvm.getelementptr %50[%53] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(0 : i32) : i32
    %57 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%47, %54, %55, %57, %56, %25) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %58 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.getelementptr %58[%59] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %61 = llvm.mlir.constant(256 : index) : i64
    %62 = llvm.mul %11, %61 : i64
    %63 = llvm.add %62, %7 : i64
    %64 = llvm.getelementptr %60[%63] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %65 = llvm.mlir.constant(0 : i32) : i32
    %66 = llvm.mlir.constant(0 : i32) : i32
    %67 = llvm.mlir.constant(0 : i32) : i32
    %68 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%64, %65, %67, %66) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %69 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.getelementptr %69[%70] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %72 = llvm.mlir.constant(512 : index) : i64
    %73 = llvm.mul %11, %72 : i64
    %74 = llvm.add %73, %7 : i64
    %75 = llvm.getelementptr %71[%74] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %76 = llvm.mlir.constant(0 : i32) : i32
    %77 = llvm.mlir.constant(0 : i32) : i32
    %78 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%68, %75, %76, %78, %77, %25) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %79 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.getelementptr %79[%80] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %82 = llvm.mlir.constant(256 : index) : i64
    %83 = llvm.mul %11, %82 : i64
    %84 = llvm.add %83, %8 : i64
    %85 = llvm.getelementptr %81[%84] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %86 = llvm.mlir.constant(0 : i32) : i32
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%85, %86, %88, %87) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %90 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %93 = llvm.mlir.constant(512 : index) : i64
    %94 = llvm.mul %11, %93 : i64
    %95 = llvm.add %94, %8 : i64
    %96 = llvm.getelementptr %92[%95] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %97 = llvm.mlir.constant(0 : i32) : i32
    %98 = llvm.mlir.constant(0 : i32) : i32
    %99 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%89, %96, %97, %99, %98, %25) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %100 = arith.addi %9, %4 overflow<nsw> : i32
    cf.br ^bb1(%100 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_9(%arg0: i32, %arg1: memref<48x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, %arg2: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg3: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg4: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg5: memref<8x784xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>, %arg6: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg6 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg5 : memref<8x784xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = builtin.unrealized_conversion_cast %arg3 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = builtin.unrealized_conversion_cast %arg2 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = builtin.unrealized_conversion_cast %arg1 : memref<48x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(112 : i32) : i32
    %9 = llvm.mlir.constant(64 : i32) : i32
    %10 = llvm.mlir.constant(24 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(4.000000e-02 : f32) : f32
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.mlir.constant(16 : i32) : i32
    %16 = llvm.mlir.constant(784 : index) : i64
    %17 = llvm.mlir.constant(48 : i32) : i32
    %18 = "hivm_regbaseintrins.intr.hivm.vbr"(%6) : (f32) -> vector<64xf32>
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%19, %20) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %22 = builtin.unrealized_conversion_cast %21 : vector<256xi1> to vector<64xi1>
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%arg0, %21, %23) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = "hivm_regbaseintrins.intr.hivm.vci"(%25, %26) : (i32, i32) -> vector<64xi32>
    %28 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%27, %24, %21) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %29 = builtin.unrealized_conversion_cast %28 : vector<256xi1> to vector<64xi1>
    %30 = llvm.intr.smin(%arg0, %8)  : (i32, i32) -> i32
    %31 = llvm.mlir.constant(1 : i32) : i32
    %32 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%30, %21, %31) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %33 = llvm.mlir.constant(64 : i32) : i32
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = "hivm_regbaseintrins.intr.hivm.vci"(%33, %34) : (i32, i32) -> vector<64xi32>
    %36 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%35, %32, %21) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %37 = builtin.unrealized_conversion_cast %36 : vector<256xi1> to vector<64xi1>
    cf.br ^bb1(%7 : i32)
  ^bb1(%38: i32):  // 2 preds: ^bb0, ^bb2
    %39 = arith.cmpi slt, %38, %10 : i32
    cf.cond_br %39, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %40 = llvm.sext %38 : i32 to i64
    %41 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.getelementptr %41[%42] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %44 = llvm.mlir.constant(128 : index) : i64
    %45 = llvm.mul %40, %44 : i64
    %46 = llvm.add %45, %12 : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.mlir.constant(0 : i32) : i32
    %51 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%47, %48, %50, %49) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %52 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%51, %13, %21) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %53 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.getelementptr %53[%54] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %56 = llvm.mlir.constant(128 : index) : i64
    %57 = llvm.mul %40, %56 : i64
    %58 = llvm.add %57, %14 : i64
    %59 = llvm.getelementptr %55[%58] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %60 = llvm.mlir.constant(0 : i32) : i32
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.mlir.constant(0 : i32) : i32
    %63 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%59, %60, %62, %61) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %64 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%63, %13, %21) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %65 = "hivm_regbaseintrins.intr.hivm.vsel"(%52, %18, %28) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %66 = "hivm_regbaseintrins.intr.hivm.vsel"(%64, %18, %36) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %67 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.getelementptr %67[%68] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %70 = llvm.getelementptr %69[%40] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = llvm.mlir.constant(0 : i32) : i32
    %73 = llvm.mlir.constant(3 : i32) : i32
    %74 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%70, %71, %73, %72) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %75 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%65, %66, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %76 = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%75, %21) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %77 = llvm.mlir.constant(1 : i32) : i32
    %78 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%76, %21, %77) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %79 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%78, %74, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %80 = llvm.mlir.constant(0 : i32) : i32
    %81 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%74, %79, %21, %80) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %82 = llvm.mlir.constant(0 : i32) : i32
    %83 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%65, %79, %21, %82) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %84 = llvm.mlir.constant(0 : i32) : i32
    %85 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%66, %79, %21, %84) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %86 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%83, %85, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %87 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%86, %21) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %88 = llvm.mlir.constant(1 : i32) : i32
    %89 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%87, %21, %88) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %90 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %91 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %93 = llvm.getelementptr %92[%40] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %94 = llvm.mlir.constant(0 : i32) : i32
    %95 = llvm.mlir.constant(0 : i32) : i32
    %96 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%79, %93, %94, %96, %95, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %97 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.getelementptr %97[%98] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %100 = llvm.getelementptr %99[%40] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %101 = llvm.mlir.constant(0 : i32) : i32
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%89, %100, %101, %103, %102, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %104 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.getelementptr %104[%105] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %107 = llvm.getelementptr %106[%40] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %108 = llvm.mlir.constant(0 : i32) : i32
    %109 = llvm.mlir.constant(0 : i32) : i32
    %110 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%81, %107, %108, %110, %109, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %111 = llvm.mlir.constant(0 : i32) : i32
    %112 = llvm.mlir.constant(1 : i32) : i32
    %113 = llvm.mlir.constant(1 : i32) : i32
    %114 = llvm.mlir.constant(0 : i32) : i32
    %115 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%83, %21, %112, %113, %114) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %116 = llvm.mlir.constant(0 : i32) : i32
    %117 = llvm.mlir.constant(1 : i32) : i32
    %118 = llvm.mlir.constant(1 : i32) : i32
    %119 = llvm.mlir.constant(0 : i32) : i32
    %120 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%85, %21, %117, %118, %119) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %121 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%115, %120) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %122 = llvm.extractvalue %121[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %123 = llvm.extractvalue %121[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %124 = llvm.mul %38, %15 : i32
    %125 = llvm.mlir.constant(0 : i32) : i32
    %126 = llvm.mlir.constant(0 : i32) : i32
    %127 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%125, %126) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %128 = builtin.unrealized_conversion_cast %127 : vector<256xi1> to vector<128xi1>
    %129 = llvm.sext %124 : i32 to i64
    %130 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.getelementptr %130[%131] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %133 = llvm.mlir.constant(784 : index) : i64
    %134 = llvm.mul %12, %133 : i64
    %135 = llvm.add %134, %129 : i64
    %136 = llvm.getelementptr %132[%135] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %137 = llvm.mlir.constant(3211264 : i32) : i32
    %138 = llvm.mlir.constant(0 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"(%122, %136, %137, %138, %127) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>) -> ()
    %139 = arith.addi %38, %11 overflow<nsw> : i32
    cf.br ^bb1(%139 : i32)
  ^bb3:  // pred: ^bb1
    %140 = llvm.add %arg0, %11 : i32
    %141 = llvm.mlir.constant(1 : i32) : i32
    %142 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%140, %21, %141) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %143 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%27, %142, %21) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %144 = builtin.unrealized_conversion_cast %143 : vector<256xi1> to vector<64xi1>
    %145 = llvm.intr.smin(%140, %8)  : (i32, i32) -> i32
    %146 = llvm.mlir.constant(1 : i32) : i32
    %147 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%145, %21, %146) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %148 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%35, %147, %21) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %149 = builtin.unrealized_conversion_cast %148 : vector<256xi1> to vector<64xi1>
    cf.br ^bb4(%7 : i32)
  ^bb4(%150: i32):  // 2 preds: ^bb3, ^bb5
    %151 = arith.cmpi slt, %150, %10 : i32
    cf.cond_br %151, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %152 = llvm.add %150, %10 : i32
    %153 = llvm.sext %152 : i32 to i64
    %154 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.getelementptr %154[%155] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %157 = llvm.mlir.constant(128 : index) : i64
    %158 = llvm.mul %153, %157 : i64
    %159 = llvm.add %158, %12 : i64
    %160 = llvm.getelementptr %156[%159] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %161 = llvm.mlir.constant(0 : i32) : i32
    %162 = llvm.mlir.constant(0 : i32) : i32
    %163 = llvm.mlir.constant(0 : i32) : i32
    %164 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%160, %161, %163, %162) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %165 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%164, %13, %21) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %166 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.getelementptr %166[%167] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %169 = llvm.mlir.constant(128 : index) : i64
    %170 = llvm.mul %153, %169 : i64
    %171 = llvm.add %170, %14 : i64
    %172 = llvm.getelementptr %168[%171] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %173 = llvm.mlir.constant(0 : i32) : i32
    %174 = llvm.mlir.constant(0 : i32) : i32
    %175 = llvm.mlir.constant(0 : i32) : i32
    %176 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%172, %173, %175, %174) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %177 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%176, %13, %21) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %178 = "hivm_regbaseintrins.intr.hivm.vsel"(%165, %18, %143) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %179 = "hivm_regbaseintrins.intr.hivm.vsel"(%177, %18, %148) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %180 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %181 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %182 = llvm.getelementptr %180[%181] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %183 = llvm.getelementptr %182[%153] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %184 = llvm.mlir.constant(0 : i32) : i32
    %185 = llvm.mlir.constant(0 : i32) : i32
    %186 = llvm.mlir.constant(3 : i32) : i32
    %187 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%183, %184, %186, %185) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %188 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%178, %179, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %189 = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%188, %21) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %190 = llvm.mlir.constant(1 : i32) : i32
    %191 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%189, %21, %190) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %192 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%191, %187, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %193 = llvm.mlir.constant(0 : i32) : i32
    %194 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%187, %192, %21, %193) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %195 = llvm.mlir.constant(0 : i32) : i32
    %196 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%178, %192, %21, %195) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %197 = llvm.mlir.constant(0 : i32) : i32
    %198 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%179, %192, %21, %197) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %199 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%196, %198, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %200 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%199, %21) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %201 = llvm.mlir.constant(1 : i32) : i32
    %202 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%200, %21, %201) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %203 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %204 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %205 = llvm.getelementptr %203[%204] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %206 = llvm.getelementptr %205[%153] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %207 = llvm.mlir.constant(0 : i32) : i32
    %208 = llvm.mlir.constant(0 : i32) : i32
    %209 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%192, %206, %207, %209, %208, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %210 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %211 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %212 = llvm.getelementptr %210[%211] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %213 = llvm.getelementptr %212[%153] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %214 = llvm.mlir.constant(0 : i32) : i32
    %215 = llvm.mlir.constant(0 : i32) : i32
    %216 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%202, %213, %214, %216, %215, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %217 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %218 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %219 = llvm.getelementptr %217[%218] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %220 = llvm.getelementptr %219[%153] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %221 = llvm.mlir.constant(0 : i32) : i32
    %222 = llvm.mlir.constant(0 : i32) : i32
    %223 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%194, %220, %221, %223, %222, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %224 = llvm.mlir.constant(0 : i32) : i32
    %225 = llvm.mlir.constant(1 : i32) : i32
    %226 = llvm.mlir.constant(1 : i32) : i32
    %227 = llvm.mlir.constant(0 : i32) : i32
    %228 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%196, %21, %225, %226, %227) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %229 = llvm.mlir.constant(0 : i32) : i32
    %230 = llvm.mlir.constant(1 : i32) : i32
    %231 = llvm.mlir.constant(1 : i32) : i32
    %232 = llvm.mlir.constant(0 : i32) : i32
    %233 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%198, %21, %230, %231, %232) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %234 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%228, %233) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %235 = llvm.extractvalue %234[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %236 = llvm.extractvalue %234[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %237 = llvm.mul %152, %15 : i32
    %238 = llvm.mlir.constant(0 : i32) : i32
    %239 = llvm.mlir.constant(0 : i32) : i32
    %240 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%238, %239) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %241 = builtin.unrealized_conversion_cast %240 : vector<256xi1> to vector<128xi1>
    %242 = llvm.sext %237 : i32 to i64
    %243 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %244 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %245 = llvm.getelementptr %243[%244] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %246 = llvm.mlir.constant(784 : index) : i64
    %247 = llvm.mul %12, %246 : i64
    %248 = llvm.add %247, %242 : i64
    %249 = llvm.getelementptr %245[%248] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %250 = llvm.mlir.constant(3211264 : i32) : i32
    %251 = llvm.mlir.constant(0 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"(%235, %249, %250, %251, %240) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>) -> ()
    %252 = arith.addi %150, %11 overflow<nsw> : i32
    cf.br ^bb4(%252 : i32)
  ^bb6:  // pred: ^bb4
    "hivm_regbaseintrins.intr.hivm.mem.bar.vst.vld"() : () -> ()
    %253 = "hivm_regbaseintrins.intr.hivm.vbr"(%17) : (i32) -> vector<64xi32>
    %254 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%27, %253, %21) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %255 = builtin.unrealized_conversion_cast %254 : vector<256xi1> to vector<64xi1>
    %256 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %257 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %258 = llvm.getelementptr %256[%257] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %259 = llvm.mlir.constant(0 : i32) : i32
    %260 = llvm.mlir.constant(0 : i32) : i32
    %261 = llvm.mlir.constant(0 : i32) : i32
    %262 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%258, %259, %261, %260) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %263 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %264 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %265 = llvm.getelementptr %263[%264] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %266 = llvm.mlir.constant(0 : i32) : i32
    %267 = llvm.mlir.constant(0 : i32) : i32
    %268 = llvm.mlir.constant(0 : i32) : i32
    %269 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%265, %266, %268, %267) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %270 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %271 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %272 = llvm.getelementptr %270[%271] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %273 = llvm.mlir.constant(0 : i32) : i32
    %274 = llvm.mlir.constant(0 : i32) : i32
    %275 = llvm.mlir.constant(0 : i32) : i32
    %276 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%272, %273, %275, %274) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %277 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%262, %269, %276, %21) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %278 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %279 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %280 = llvm.getelementptr %278[%279] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %281 = llvm.mlir.constant(0 : i32) : i32
    %282 = llvm.mlir.constant(0 : i32) : i32
    %283 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%277, %280, %281, %283, %282, %254) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @attention_scope_10(%arg0: memref<48x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg2: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg3: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg4: memref<8x784xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>, %arg5: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg4 : memref<8x784xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg3 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = builtin.unrealized_conversion_cast %arg2 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = builtin.unrealized_conversion_cast %arg1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = builtin.unrealized_conversion_cast %arg0 : memref<48x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %7 = llvm.mlir.constant(112 : i32) : i32
    %8 = llvm.mlir.constant(64 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(4.000000e-02 : f32) : f32
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.mlir.constant(16 : i32) : i32
    %16 = llvm.mlir.constant(784 : index) : i64
    %17 = "hivm_regbaseintrins.intr.hivm.vbr"(%6) : (f32) -> vector<64xf32>
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%18, %19) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %21 = builtin.unrealized_conversion_cast %20 : vector<256xi1> to vector<64xi1>
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%7, %20, %22) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %24 = llvm.mlir.constant(64 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = "hivm_regbaseintrins.intr.hivm.vci"(%24, %25) : (i32, i32) -> vector<64xi32>
    %27 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%26, %23, %20) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %28 = builtin.unrealized_conversion_cast %27 : vector<256xi1> to vector<64xi1>
    cf.br ^bb1(%9 : i32)
  ^bb1(%29: i32):  // 2 preds: ^bb0, ^bb2
    %30 = arith.cmpi slt, %29, %10 : i32
    cf.cond_br %30, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %31 = llvm.sext %29 : i32 to i64
    %32 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %35 = llvm.mlir.constant(128 : index) : i64
    %36 = llvm.mul %31, %35 : i64
    %37 = llvm.add %36, %12 : i64
    %38 = llvm.getelementptr %34[%37] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.mlir.constant(0 : i32) : i32
    %41 = llvm.mlir.constant(0 : i32) : i32
    %42 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%38, %39, %41, %40) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %43 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%42, %13, %20) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %44 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.getelementptr %44[%45] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %47 = llvm.mlir.constant(128 : index) : i64
    %48 = llvm.mul %31, %47 : i64
    %49 = llvm.add %48, %14 : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %51 = llvm.mlir.constant(0 : i32) : i32
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%50, %51, %53, %52) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %55 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%54, %13, %20) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %56 = "hivm_regbaseintrins.intr.hivm.vsel"(%55, %17, %27) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %57 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.getelementptr %57[%58] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %60 = llvm.getelementptr %59[%31] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.mlir.constant(0 : i32) : i32
    %63 = llvm.mlir.constant(3 : i32) : i32
    %64 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%60, %61, %63, %62) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %65 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%43, %56, %20) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %66 = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%65, %20) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %67 = llvm.mlir.constant(1 : i32) : i32
    %68 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%66, %20, %67) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %69 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%68, %64, %20) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%64, %69, %20, %70) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %72 = llvm.mlir.constant(0 : i32) : i32
    %73 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%43, %69, %20, %72) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %74 = llvm.mlir.constant(0 : i32) : i32
    %75 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%56, %69, %20, %74) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %76 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%73, %75, %20) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %77 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%76, %20) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %78 = llvm.mlir.constant(1 : i32) : i32
    %79 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%77, %20, %78) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %80 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.getelementptr %80[%81] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %83 = llvm.getelementptr %82[%31] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %84 = llvm.mlir.constant(0 : i32) : i32
    %85 = llvm.mlir.constant(0 : i32) : i32
    %86 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%69, %83, %84, %86, %85, %20) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %87 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.getelementptr %87[%88] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %90 = llvm.getelementptr %89[%31] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %91 = llvm.mlir.constant(0 : i32) : i32
    %92 = llvm.mlir.constant(0 : i32) : i32
    %93 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%79, %90, %91, %93, %92, %20) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %94 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.getelementptr %94[%95] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %97 = llvm.getelementptr %96[%31] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %98 = llvm.mlir.constant(0 : i32) : i32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%71, %97, %98, %100, %99, %20) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %101 = llvm.mlir.constant(0 : i32) : i32
    %102 = llvm.mlir.constant(1 : i32) : i32
    %103 = llvm.mlir.constant(1 : i32) : i32
    %104 = llvm.mlir.constant(0 : i32) : i32
    %105 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%73, %20, %102, %103, %104) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %106 = llvm.mlir.constant(0 : i32) : i32
    %107 = llvm.mlir.constant(1 : i32) : i32
    %108 = llvm.mlir.constant(1 : i32) : i32
    %109 = llvm.mlir.constant(0 : i32) : i32
    %110 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%75, %20, %107, %108, %109) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %111 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%105, %110) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %112 = llvm.extractvalue %111[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %113 = llvm.extractvalue %111[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %114 = llvm.mul %29, %15 : i32
    %115 = llvm.mlir.constant(0 : i32) : i32
    %116 = llvm.mlir.constant(0 : i32) : i32
    %117 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%115, %116) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %118 = builtin.unrealized_conversion_cast %117 : vector<256xi1> to vector<128xi1>
    %119 = llvm.sext %114 : i32 to i64
    %120 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.getelementptr %120[%121] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %123 = llvm.mlir.constant(784 : index) : i64
    %124 = llvm.mul %12, %123 : i64
    %125 = llvm.add %124, %119 : i64
    %126 = llvm.getelementptr %122[%125] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %127 = llvm.mlir.constant(3211264 : i32) : i32
    %128 = llvm.mlir.constant(0 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"(%112, %126, %127, %128, %117) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>) -> ()
    %129 = arith.addi %29, %11 overflow<nsw> : i32
    cf.br ^bb1(%129 : i32)
  ^bb3:  // pred: ^bb1
    "hivm_regbaseintrins.intr.hivm.mem.bar.vst.vld"() : () -> ()
    %130 = llvm.mlir.constant(0 : i32) : i32
    %131 = llvm.mlir.constant(0 : i32) : i32
    %132 = "hivm_regbaseintrins.intr.hivm.vci"(%130, %131) : (i32, i32) -> vector<64xi32>
    %133 = "hivm_regbaseintrins.intr.hivm.vbr"(%10) : (i32) -> vector<64xi32>
    %134 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%132, %133, %20) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %135 = builtin.unrealized_conversion_cast %134 : vector<256xi1> to vector<64xi1>
    %136 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %138 = llvm.getelementptr %136[%137] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %139 = llvm.mlir.constant(0 : i32) : i32
    %140 = llvm.mlir.constant(0 : i32) : i32
    %141 = llvm.mlir.constant(0 : i32) : i32
    %142 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%138, %139, %141, %140) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %143 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.getelementptr %143[%144] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %146 = llvm.mlir.constant(0 : i32) : i32
    %147 = llvm.mlir.constant(0 : i32) : i32
    %148 = llvm.mlir.constant(0 : i32) : i32
    %149 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%145, %146, %148, %147) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %150 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %151 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %152 = llvm.getelementptr %150[%151] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %153 = llvm.mlir.constant(0 : i32) : i32
    %154 = llvm.mlir.constant(0 : i32) : i32
    %155 = llvm.mlir.constant(0 : i32) : i32
    %156 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%152, %153, %155, %154) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %157 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%142, %149, %156, %20) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %158 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %159 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %160 = llvm.getelementptr %158[%159] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %161 = llvm.mlir.constant(0 : i32) : i32
    %162 = llvm.mlir.constant(0 : i32) : i32
    %163 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%157, %160, %161, %163, %162, %134) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @attention_scope_11(%arg0: memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(48 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mlir.constant(192 : index) : i64
    %9 = llvm.mlir.constant(256 : index) : i64
    %10 = llvm.mlir.constant(320 : index) : i64
    %11 = llvm.mlir.constant(384 : index) : i64
    %12 = llvm.mlir.constant(448 : index) : i64
    %13 = "hivm_regbaseintrins.intr.hivm.vbr"(%1) : (f32) -> vector<64xf32>
    cf.br ^bb1(%2 : i32)
  ^bb1(%14: i32):  // 2 preds: ^bb0, ^bb2
    %15 = arith.cmpi slt, %14, %3 : i32
    cf.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%16, %17) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %19 = builtin.unrealized_conversion_cast %18 : vector<256xi1> to vector<64xi1>
    %20 = llvm.sext %14 : i32 to i64
    %21 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %24 = llvm.mlir.constant(512 : index) : i64
    %25 = llvm.mul %20, %24 : i64
    %26 = llvm.add %25, %5 : i64
    %27 = llvm.getelementptr %23[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %27, %28, %30, %29, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %31 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.getelementptr %31[%32] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %34 = llvm.mlir.constant(512 : index) : i64
    %35 = llvm.mul %20, %34 : i64
    %36 = llvm.add %35, %6 : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %37, %38, %40, %39, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %41 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.getelementptr %41[%42] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %44 = llvm.mlir.constant(512 : index) : i64
    %45 = llvm.mul %20, %44 : i64
    %46 = llvm.add %45, %7 : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %47, %48, %50, %49, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %51 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %54 = llvm.mlir.constant(512 : index) : i64
    %55 = llvm.mul %20, %54 : i64
    %56 = llvm.add %55, %8 : i64
    %57 = llvm.getelementptr %53[%56] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %58 = llvm.mlir.constant(0 : i32) : i32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %57, %58, %60, %59, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %61 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.getelementptr %61[%62] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %64 = llvm.mlir.constant(512 : index) : i64
    %65 = llvm.mul %20, %64 : i64
    %66 = llvm.add %65, %9 : i64
    %67 = llvm.getelementptr %63[%66] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %67, %68, %70, %69, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %71 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.getelementptr %71[%72] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %74 = llvm.mlir.constant(512 : index) : i64
    %75 = llvm.mul %20, %74 : i64
    %76 = llvm.add %75, %10 : i64
    %77 = llvm.getelementptr %73[%76] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(0 : i32) : i32
    %80 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %77, %78, %80, %79, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %81 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.getelementptr %81[%82] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %84 = llvm.mlir.constant(512 : index) : i64
    %85 = llvm.mul %20, %84 : i64
    %86 = llvm.add %85, %11 : i64
    %87 = llvm.getelementptr %83[%86] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %87, %88, %90, %89, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %91 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.getelementptr %91[%92] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %94 = llvm.mlir.constant(512 : index) : i64
    %95 = llvm.mul %20, %94 : i64
    %96 = llvm.add %95, %12 : i64
    %97 = llvm.getelementptr %93[%96] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %98 = llvm.mlir.constant(0 : i32) : i32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%13, %97, %98, %100, %99, %18) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %101 = arith.addi %14, %4 overflow<nsw> : i32
    cf.br ^bb1(%101 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_scope_12(%arg0: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %arg1: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = "hivm_regbaseintrins.intr.hivm.vbr"(%2) : (f32) -> vector<64xf32>
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%6, %7) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %9 = builtin.unrealized_conversion_cast %8 : vector<256xi1> to vector<64xi1>
    %10 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %1[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.getelementptr %10[%11] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%5, %12, %13, %15, %14, %8) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %16 = "hivm_regbaseintrins.intr.hivm.vbr"(%4) : (f32) -> vector<64xf32>
    %17 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%16, %19, %20, %22, %21, %8) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @attention_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: memref<?xi32, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xbf16, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(18432 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(6144 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(48 : i64) : i64
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.constant(64 : i64) : i64
    %8 = llvm.mlir.constant(1 : i8) : i8
    %9 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %10 = llvm.mlir.constant(0 : i8) : i8
    %11 = llvm.mlir.constant(128 : i64) : i64
    %12 = llvm.mlir.constant(112 : i64) : i64
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.mlir.constant(6 : i64) : i64
    %15 = llvm.mlir.constant(96 : i64) : i64
    %16 = llvm.mlir.constant(43008 : i64) : i64
    %17 = llvm.mlir.constant(24576 : i64) : i64
    %18 = llvm.mlir.constant(0 : i64) : i64
    %19 = llvm.mlir.constant(408576 : i64) : i64
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(320 : i32) : i32
    %22 = llvm.mlir.constant(10240 : i32) : i32
    %23 = llvm.mlir.constant(1280 : i32) : i32
    %24 = llvm.mlir.constant(4 : i32) : i32
    %25 = llvm.mlir.constant(128 : i32) : i32
    %26 = llvm.mlir.constant(112 : i32) : i32
    %27 = llvm.mlir.constant(3 : i32) : i32
    %28 = llvm.mlir.constant(2 : i32) : i32
    %29 = llvm.mlir.constant(68096 : i32) : i32
    %30 = llvm.mlir.constant(1120 : i32) : i32
    %31 = llvm.mlir.constant(16 : index) : i64
    %32 = llvm.mlir.constant(32768 : i32) : i32
    %33 = llvm.mlir.constant(16384 : i32) : i32
    %34 = llvm.mlir.constant(true) : i1
    %35 = llvm.mlir.constant(false) : i1
    %36 = llvm.mlir.constant(57344 : i32) : i32
    %37 = llvm.mlir.constant(10 : i32) : i32
    %38 = llvm.mlir.constant(-1 : i32) : i32
    %39 = llvm.mlir.constant(55296 : i32) : i32
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = builtin.unrealized_conversion_cast %40 : i64 to index
    %42 = llvm.mlir.constant(73728 : i64) : i64
    %43 = llvm.mlir.constant(1279 : i32) : i32
    %44 = llvm.mlir.constant(111 : i32) : i32
    %45 = llvm.mlir.constant(1 : i32) : i32
    %46 = hivm.hir.get_block_idx -> i64
    %47 = llvm.trunc %46 : i64 to i32
    %48 = llvm.srem %47, %arg11  : i32
    %49 = hivm.hir.pointer_cast(%19) : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>>
    %50 = hivm.hir.pointer_cast(%18) : memref<204288xbf16, #hivm.address_space<cbuf>>
    %51 = hivm.hir.pointer_cast(%17) : memref<6x6x16x16xbf16, #hivm.address_space<ca>>
    %52 = hivm.hir.pointer_cast(%16) : memref<7x6x16x16xbf16, #hivm.address_space<ca>>
    %53 = hivm.hir.pointer_cast(%18) : memref<8x6x16x16xbf16, #hivm.address_space<ca>>
    %54 = hivm.hir.pointer_cast(%18) : memref<32768xbf16, #hivm.address_space<cb>>
    %55 = hivm.hir.pointer_cast(%18) : memref<65536xf32, #hivm.address_space<cc>>
    %56 = hivm.hir.pointer_cast(%18) : memref<55552xf32, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %56 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %56 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %57 = llvm.mul %48, %21 : i32
    %58 = llvm.intr.smin(%57, %22)  : (i32, i32) -> i32
    %59 = llvm.add %58, %21 : i32
    %60 = llvm.intr.smin(%59, %22)  : (i32, i32) -> i32
    %61 = llvm.sdiv %58, %23  : i32
    %62 = llvm.icmp "slt" %58, %60 : i32
    %63 = llvm.add %60, %43 : i32
    %64 = llvm.sdiv %63, %23  : i32
    %65 = llvm.select %62, %64, %61 : i1, i32
    cf.br ^bb1(%61 : i32)
  ^bb1(%66: i32):  // 2 preds: ^bb0, ^bb19
    %67 = arith.cmpi slt, %66, %65 : i32
    cf.cond_br %67, ^bb2, ^bb20
  ^bb2:  // pred: ^bb1
    %68 = llvm.mul %66, %23 : i32
    %69 = llvm.sext %66 : i32 to i64
    %70 = builtin.unrealized_conversion_cast %69 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%70], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %71 = memref.load %reinterpret_cast[%41] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %72 = llvm.add %71, %38 : i32
    %73 = llvm.sdiv %72, %25  : i32
    %74 = llvm.add %71, %44 : i32
    %75 = llvm.sdiv %74, %26  : i32
    %76 = llvm.intr.smax(%68, %58)  : (i32, i32) -> i32
    %77 = llvm.add %68, %75 : i32
    %78 = llvm.intr.smin(%77, %60)  : (i32, i32) -> i32
    %79 = llvm.sub %76, %68 : i32
    %80 = llvm.mul %79, %26 : i32
    %81 = llvm.sub %78, %76 : i32
    %82 = llvm.intr.smax(%81, %20)  : (i32, i32) -> i32
    %83 = llvm.mul %66, %39 : i32
    %84 = llvm.sext %83 : i32 to i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%85], sizes: [96, 576], strides: [576, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<96x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>>
    %cast = memref.cast %reinterpret_cast_4 : memref<96x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_5 = memref.cast %49 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast, %cast_5, %20) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    %cast_6 = memref.cast %49 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_7 = memref.cast %51 : memref<6x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    call @l12l0_cbuf_to_ca_bfloat16_t(%cast_6, %cast_7, %18, %15, %15, %14, %15, %15) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) -> ()
    %86 = llvm.add %82, %45 : i32
    cf.br ^bb3(%38 : i32)
  ^bb3(%87: i32):  // 2 preds: ^bb2, ^bb18
    %88 = arith.cmpi slt, %87, %86 : i32
    cf.cond_br %88, ^bb4, ^bb19
  ^bb4:  // pred: ^bb3
    %89 = llvm.icmp "sge" %87, %20 : i32
    %90 = llvm.icmp "slt" %87, %82 : i32
    %91 = llvm.and %89, %90  : i1
    %92 = llvm.icmp "sgt" %87, %20 : i32
    %93 = llvm.srem %87, %27  : i32
    %94 = llvm.add %87, %45 : i32
    %95 = llvm.icmp "slt" %94, %82 : i32
    cf.cond_br %95, ^bb5, ^bb8
  ^bb5:  // pred: ^bb4
    %96 = llvm.mul %94, %26 : i32
    %97 = llvm.add %80, %96 : i32
    %98 = llvm.sdiv %97, %25  : i32
    %99 = llvm.srem %97, %25  : i32
    %100 = llvm.sub %25, %99 : i32
    %101 = llvm.intr.smin(%100, %26)  : (i32, i32) -> i32
    %102 = llvm.srem %94, %27  : i32
    %103 = llvm.mul %102, %29 : i32
    %104 = llvm.sext %103 : i32 to i64
    %105 = builtin.unrealized_conversion_cast %104 : i64 to index
    %base_buffer_8, %offset_9, %sizes_10, %strides_11 = memref.extract_strided_metadata %50 : memref<204288xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index
    %106 = llvm.mul %66, %30 : i32
    %107 = llvm.intr.smin(%98, %73)  : (i32, i32) -> i32
    %108 = llvm.sext %106 : i32 to i64
    %109 = llvm.sext %107 : i32 to i64
    %110 = llvm.add %108, %109 : i64
    %111 = builtin.unrealized_conversion_cast %110 : i64 to index
    %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%111], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %112 = memref.load %reinterpret_cast_12[%41] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %113 = llvm.sext %112 : i32 to i64
    %114 = llvm.sext %102 : i32 to i64
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %114]
    %115 = llvm.mul %113, %42 : i64
    %116 = builtin.unrealized_conversion_cast %115 : i64 to index
    %117 = llvm.sext %101 : i32 to i64
    %118 = builtin.unrealized_conversion_cast %117 : i64 to index
    %119 = llvm.sext %99 : i32 to i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    %121 = llvm.udiv %117, %31  : i64
    %122 = builtin.unrealized_conversion_cast %121 : i64 to index
    %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %arg3 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %c576 = arith.constant 576 : index
    %123 = arith.muli %120, %c576 : index
    %124 = arith.addi %116, %123 : index
    %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [%124], sizes: [%118, 576], strides: [576, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %base_buffer_8 to offset: [%105], sizes: [36, %122, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<36x?x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_19 = memref.cast %reinterpret_cast_17 : memref<?x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_20 = memref.cast %reinterpret_cast_18 : memref<36x?x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast_19, %cast_20, %24) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    %125 = llvm.icmp "slt" %101, %26 : i32
    cf.cond_br %125, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %126 = llvm.add %98, %45 : i32
    %127 = llvm.intr.smin(%126, %73)  : (i32, i32) -> i32
    %128 = llvm.sext %127 : i32 to i64
    %129 = llvm.add %108, %128 : i64
    %130 = builtin.unrealized_conversion_cast %129 : i64 to index
    %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%130], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %131 = memref.load %reinterpret_cast_21[%41] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %132 = llvm.sext %131 : i32 to i64
    %133 = llvm.mul %132, %42 : i64
    %134 = builtin.unrealized_conversion_cast %133 : i64 to index
    %135 = llvm.sub %26, %101 : i32
    %136 = llvm.sext %135 : i32 to i64
    %137 = builtin.unrealized_conversion_cast %136 : i64 to index
    %138 = llvm.udiv %136, %31  : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    %base_buffer_22, %offset_23, %sizes_24, %strides_25 = memref.extract_strided_metadata %arg3 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_26 = memref.reinterpret_cast %base_buffer_22 to offset: [%134], sizes: [%137, 576], strides: [576, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>>
    %c256 = arith.constant 256 : index
    %140 = arith.muli %122, %c256 : index
    %141 = arith.addi %105, %140 : index
    %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_8 to offset: [%141], sizes: [36, %139, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<36x?x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_28 = memref.cast %reinterpret_cast_26 : memref<?x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_29 = memref.cast %reinterpret_cast_27 : memref<36x?x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast_28, %cast_29, %24) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %114]
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb4, ^bb7
    cf.cond_br %91, ^bb9, ^bb12
  ^bb9:  // pred: ^bb8
    %142 = llvm.mul %93, %29 : i32
    %143 = llvm.sext %142 : i32 to i64
    %144 = builtin.unrealized_conversion_cast %143 : i64 to index
    %base_buffer_30, %offset_31, %sizes_32, %strides_33 = memref.extract_strided_metadata %50 : memref<204288xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index
    %145 = llvm.srem %87, %28  : i32
    %146 = llvm.sext %93 : i32 to i64
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, %146]
    %147 = llvm.sext %145 : i32 to i64
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %147]
    %148 = llvm.mul %145, %32 : i32
    %149 = llvm.sext %148 : i32 to i64
    %150 = builtin.unrealized_conversion_cast %149 : i64 to index
    %base_buffer_34, %offset_35, %sizes_36, %strides_37 = memref.extract_strided_metadata %55 : memref<65536xf32, #hivm.address_space<cc>> -> memref<f32, #hivm.address_space<cc>>, index, index, index
    %reinterpret_cast_38 = memref.reinterpret_cast %base_buffer_34 to offset: [%150], sizes: [7, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<f32, #hivm.address_space<cc>> to memref<7x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %147]
    %151 = llvm.mul %145, %33 : i32
    %152 = llvm.sext %151 : i32 to i64
    %153 = builtin.unrealized_conversion_cast %152 : i64 to index
    %base_buffer_39, %offset_40, %sizes_41, %strides_42 = memref.extract_strided_metadata %54 : memref<32768xbf16, #hivm.address_space<cb>> -> memref<bf16, #hivm.address_space<cb>>, index, index, index
    %reinterpret_cast_43 = memref.reinterpret_cast %base_buffer_39 to offset: [%153], sizes: [6, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cb>> to memref<6x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>>
    %reinterpret_cast_44 = memref.reinterpret_cast %base_buffer_30 to offset: [%144], sizes: [6, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<6x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_45 = memref.cast %reinterpret_cast_44 : memref<6x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_46 = memref.cast %reinterpret_cast_43 : memref<6x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t_transpose(%cast_45, %cast_46, %18, %15, %15, %13, %15, %12) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %147]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %147]
    %cast_47 = memref.cast %51 : memref<6x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_48 = memref.cast %reinterpret_cast_43 : memref<6x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_49 = memref.cast %reinterpret_cast_38 : memref<7x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_47, %cast_48, %cast_49, %15, %15, %12, %13) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %147]
    %154 = llvm.srem %94, %28  : i32
    %155 = llvm.sext %154 : i32 to i64
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %155]
    %base_buffer_50, %offset_51, %sizes_52:4, %strides_53:4 = memref.extract_strided_metadata %49 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_54 = memref.reinterpret_cast %base_buffer_50 to offset: [9216], sizes: [7, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<7x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 9216>, #hivm.address_space<cbuf>>
    %cast_55 = memref.cast %reinterpret_cast_54 : memref<7x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 9216>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_56 = memref.cast %52 : memref<7x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    call @l12l0_cbuf_to_ca_bfloat16_t(%cast_55, %cast_56, %18, %12, %12, %13, %12, %15) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) -> ()
    %156 = llvm.mul %154, %33 : i32
    %157 = llvm.sext %156 : i32 to i64
    %158 = builtin.unrealized_conversion_cast %157 : i64 to index
    %base_buffer_57, %offset_58, %sizes_59, %strides_60 = memref.extract_strided_metadata %54 : memref<32768xbf16, #hivm.address_space<cb>> -> memref<bf16, #hivm.address_space<cb>>, index, index, index
    %reinterpret_cast_61 = memref.reinterpret_cast %base_buffer_57 to offset: [%158], sizes: [7, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cb>> to memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>>
    %c10752 = arith.constant 10752 : index
    %159 = arith.addi %144, %c10752 : index
    %reinterpret_cast_62 = memref.reinterpret_cast %base_buffer_30 to offset: [%159], sizes: [7, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_63 = memref.cast %reinterpret_cast_62 : memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_64 = memref.cast %reinterpret_cast_61 : memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t_transpose(%cast_63, %cast_64, %18, %12, %12, %13, %12, %12) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %155]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %155]
    %cast_65 = memref.cast %52 : memref<7x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_66 = memref.cast %reinterpret_cast_61 : memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_67 = memref.cast %reinterpret_cast_38 : memref<7x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_65, %cast_66, %cast_67, %15, %12, %12, %18) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %155]
    %160 = llvm.add %87, %28 : i32
    %161 = llvm.srem %160, %28  : i32
    %162 = llvm.sext %161 : i32 to i64
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %162]
    %base_buffer_68, %offset_69, %sizes_70:4, %strides_71:4 = memref.extract_strided_metadata %49 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_72 = memref.reinterpret_cast %base_buffer_68 to offset: [19968], sizes: [8, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<8x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 19968>, #hivm.address_space<cbuf>>
    %cast_73 = memref.cast %reinterpret_cast_72 : memref<8x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 19968>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_74 = memref.cast %53 : memref<8x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    call @l12l0_cbuf_to_ca_bfloat16_t(%cast_73, %cast_74, %18, %11, %11, %13, %11, %15) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) -> ()
    %163 = llvm.mul %161, %33 : i32
    %164 = llvm.sext %163 : i32 to i64
    %165 = builtin.unrealized_conversion_cast %164 : i64 to index
    %base_buffer_75, %offset_76, %sizes_77, %strides_78 = memref.extract_strided_metadata %54 : memref<32768xbf16, #hivm.address_space<cb>> -> memref<bf16, #hivm.address_space<cb>>, index, index, index
    %reinterpret_cast_79 = memref.reinterpret_cast %base_buffer_75 to offset: [%165], sizes: [8, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cb>> to memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>>
    %c23296 = arith.constant 23296 : index
    %166 = arith.addi %144, %c23296 : index
    %reinterpret_cast_80 = memref.reinterpret_cast %base_buffer_30 to offset: [%166], sizes: [8, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_81 = memref.cast %reinterpret_cast_80 : memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_82 = memref.cast %reinterpret_cast_79 : memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t_transpose(%cast_81, %cast_82, %18, %11, %11, %13, %11, %12) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %162]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %162]
    %cast_83 = memref.cast %53 : memref<8x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_84 = memref.cast %reinterpret_cast_79 : memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_85 = memref.cast %reinterpret_cast_38 : memref<7x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_83, %cast_84, %cast_85, %15, %11, %12, %18) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %162]
    %167 = llvm.add %87, %27 : i32
    %168 = llvm.srem %167, %28  : i32
    %169 = llvm.sext %168 : i32 to i64
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %169]
    %base_buffer_86, %offset_87, %sizes_88:4, %strides_89:4 = memref.extract_strided_metadata %49 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_90 = memref.reinterpret_cast %base_buffer_86 to offset: [32256], sizes: [7, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<7x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 32256>, #hivm.address_space<cbuf>>
    %cast_91 = memref.cast %reinterpret_cast_90 : memref<7x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 32256>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_92 = memref.cast %52 : memref<7x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    call @l12l0_cbuf_to_ca_bfloat16_t(%cast_91, %cast_92, %18, %12, %12, %13, %12, %15) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) -> ()
    %170 = llvm.mul %168, %33 : i32
    %171 = llvm.sext %170 : i32 to i64
    %172 = builtin.unrealized_conversion_cast %171 : i64 to index
    %base_buffer_93, %offset_94, %sizes_95, %strides_96 = memref.extract_strided_metadata %54 : memref<32768xbf16, #hivm.address_space<cb>> -> memref<bf16, #hivm.address_space<cb>>, index, index, index
    %reinterpret_cast_97 = memref.reinterpret_cast %base_buffer_93 to offset: [%172], sizes: [7, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cb>> to memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>>
    %c37632 = arith.constant 37632 : index
    %173 = arith.addi %144, %c37632 : index
    %reinterpret_cast_98 = memref.reinterpret_cast %base_buffer_30 to offset: [%173], sizes: [7, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_99 = memref.cast %reinterpret_cast_98 : memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_100 = memref.cast %reinterpret_cast_97 : memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t_transpose(%cast_99, %cast_100, %18, %12, %12, %13, %12, %12) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %169]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %169]
    %cast_101 = memref.cast %52 : memref<7x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_102 = memref.cast %reinterpret_cast_97 : memref<7x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_103 = memref.cast %reinterpret_cast_38 : memref<7x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_101, %cast_102, %cast_103, %15, %12, %12, %18) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %169]
    %174 = llvm.add %87, %24 : i32
    %175 = llvm.srem %174, %28  : i32
    %176 = llvm.sext %175 : i32 to i64
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %176]
    %base_buffer_104, %offset_105, %sizes_106:4, %strides_107:4 = memref.extract_strided_metadata %49 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_108 = memref.reinterpret_cast %base_buffer_104 to offset: [43008], sizes: [8, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<8x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 43008>, #hivm.address_space<cbuf>>
    %cast_109 = memref.cast %reinterpret_cast_108 : memref<8x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 43008>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_110 = memref.cast %53 : memref<8x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    call @l12l0_cbuf_to_ca_bfloat16_t(%cast_109, %cast_110, %18, %11, %11, %13, %11, %15) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) -> ()
    %177 = llvm.mul %175, %33 : i32
    %178 = llvm.sext %177 : i32 to i64
    %179 = builtin.unrealized_conversion_cast %178 : i64 to index
    %base_buffer_111, %offset_112, %sizes_113, %strides_114 = memref.extract_strided_metadata %54 : memref<32768xbf16, #hivm.address_space<cb>> -> memref<bf16, #hivm.address_space<cb>>, index, index, index
    %reinterpret_cast_115 = memref.reinterpret_cast %base_buffer_111 to offset: [%179], sizes: [8, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cb>> to memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>>
    %c50176 = arith.constant 50176 : index
    %180 = arith.addi %144, %c50176 : index
    %reinterpret_cast_116 = memref.reinterpret_cast %base_buffer_30 to offset: [%180], sizes: [8, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_117 = memref.cast %reinterpret_cast_116 : memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_118 = memref.cast %reinterpret_cast_115 : memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t_transpose(%cast_117, %cast_118, %18, %11, %11, %13, %11, %12) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %176]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %176]
    %cast_119 = memref.cast %53 : memref<8x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_120 = memref.cast %reinterpret_cast_115 : memref<8x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_121 = memref.cast %reinterpret_cast_38 : memref<7x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_119, %cast_120, %cast_121, %15, %11, %12, %18) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %176]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, %147]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, %147]
    cf.cond_br %92, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 1
    cf.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %reinterpret_cast_122 = memref.reinterpret_cast %base_buffer to offset: [%41], sizes: [48, 112], strides: [128, 1] : memref<f32, #hivm.address_space<ub>> to memref<48x112xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %cast_123 = memref.cast %reinterpret_cast_38 : memref<7x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_124 = memref.cast %reinterpret_cast_122 : memref<48x112xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_123, %cast_124, %18, %9, %18, %35, %34, %10, %18, %8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %147]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 0
    cf.br ^bb12
  ^bb12:  // 2 preds: ^bb8, ^bb11
    cf.cond_br %92, ^bb13, ^bb18
  ^bb13:  // pred: ^bb12
    %181 = llvm.add %87, %28 : i32
    %182 = llvm.srem %181, %27  : i32
    %183 = llvm.mul %182, %29 : i32
    %184 = llvm.sext %183 : i32 to i64
    %185 = builtin.unrealized_conversion_cast %184 : i64 to index
    %base_buffer_125, %offset_126, %sizes_127, %strides_128 = memref.extract_strided_metadata %50 : memref<204288xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index
    %186 = llvm.add %183, %36 : i32
    %187 = llvm.sext %186 : i32 to i64
    %188 = builtin.unrealized_conversion_cast %187 : i64 to index
    %base_buffer_129, %offset_130, %sizes_131, %strides_132 = memref.extract_strided_metadata %50 : memref<204288xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index
    %reinterpret_cast_133 = memref.reinterpret_cast %base_buffer_129 to offset: [%188], sizes: [7, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<7x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %189 = llvm.add %182, %37 : i32
    %190 = llvm.sext %189 : i32 to i64
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE1>, <PIPE_MTE1>] flag = %190
    %191 = llvm.icmp "sgt" %87, %45 : i32
    %192 = llvm.or %191, %91  : i1
    %193 = llvm.zext %192 : i1 to i32
    %194 = llvm.add %87, %193 : i32
    %195 = llvm.srem %194, %28  : i32
    %196 = llvm.zext %91 : i1 to i32
    %197 = llvm.add %87, %196 : i32
    %198 = llvm.srem %197, %28  : i32
    %199 = llvm.sext %198 : i32 to i64
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %199]
    %reinterpret_cast_134 = memref.reinterpret_cast %base_buffer_125 to offset: [%185], sizes: [16, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<16x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %200 = llvm.mul %198, %32 : i32
    %201 = llvm.sext %200 : i32 to i64
    %202 = builtin.unrealized_conversion_cast %201 : i64 to index
    %base_buffer_135, %offset_136, %sizes_137, %strides_138 = memref.extract_strided_metadata %55 : memref<65536xf32, #hivm.address_space<cc>> -> memref<f32, #hivm.address_space<cc>>, index, index, index
    %reinterpret_cast_139 = memref.reinterpret_cast %base_buffer_135 to offset: [%202], sizes: [16, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<f32, #hivm.address_space<cc>> to memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
    %203 = llvm.srem %195, %28  : i32
    %204 = llvm.sext %203 : i32 to i64
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %204]
    %cast_140 = memref.cast %reinterpret_cast_133 : memref<7x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_141 = memref.cast %52 : memref<7x6x16x16xbf16, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    call @l12l0_cbuf_to_ca_bfloat16_t(%cast_140, %cast_141, %18, %12, %12, %13, %12, %15) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) -> ()
    %base_buffer_142, %offset_143, %sizes_144:4, %strides_145:4 = memref.extract_strided_metadata %52 : memref<7x6x16x16xbf16, #hivm.address_space<ca>> -> memref<bf16, #hivm.address_space<ca>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_146 = memref.reinterpret_cast %base_buffer_142 to offset: [0], sizes: [4, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<bf16, #hivm.address_space<ca>> to memref<4x6x16x16xbf16, strided<[1536, 256, 16, 1]>, #hivm.address_space<ca>>
    %205 = llvm.mul %203, %33 : i32
    %206 = llvm.sext %205 : i32 to i64
    %207 = builtin.unrealized_conversion_cast %206 : i64 to index
    %base_buffer_147, %offset_148, %sizes_149, %strides_150 = memref.extract_strided_metadata %54 : memref<32768xbf16, #hivm.address_space<cb>> -> memref<bf16, #hivm.address_space<cb>>, index, index, index
    %reinterpret_cast_151 = memref.reinterpret_cast %base_buffer_147 to offset: [%207], sizes: [4, 16, 16, 16], strides: [4096, 256, 16, 1] : memref<bf16, #hivm.address_space<cb>> to memref<4x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>>
    %cast_152 = memref.cast %reinterpret_cast_134 : memref<16x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_153 = memref.cast %reinterpret_cast_151 : memref<4x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t(%cast_152, %cast_153, %18, %7, %7, %6, %7, %5) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %204]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %204]
    %cast_154 = memref.cast %reinterpret_cast_146 : memref<4x6x16x16xbf16, strided<[1536, 256, 16, 1]>, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_155 = memref.cast %reinterpret_cast_151 : memref<4x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_156 = memref.cast %reinterpret_cast_139 : memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_154, %cast_155, %cast_156, %15, %7, %5, %13) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %204]
    %208 = llvm.add %195, %45 : i32
    %209 = llvm.srem %208, %28  : i32
    %210 = llvm.sext %209 : i32 to i64
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %210]
    %base_buffer_157, %offset_158, %sizes_159:4, %strides_160:4 = memref.extract_strided_metadata %52 : memref<7x6x16x16xbf16, #hivm.address_space<ca>> -> memref<bf16, #hivm.address_space<ca>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_161 = memref.reinterpret_cast %base_buffer_157 to offset: [6144], sizes: [3, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<bf16, #hivm.address_space<ca>> to memref<3x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 6144>, #hivm.address_space<ca>>
    %211 = llvm.mul %209, %33 : i32
    %212 = llvm.sext %211 : i32 to i64
    %213 = builtin.unrealized_conversion_cast %212 : i64 to index
    %base_buffer_162, %offset_163, %sizes_164, %strides_165 = memref.extract_strided_metadata %54 : memref<32768xbf16, #hivm.address_space<cb>> -> memref<bf16, #hivm.address_space<cb>>, index, index, index
    %reinterpret_cast_166 = memref.reinterpret_cast %base_buffer_162 to offset: [%213], sizes: [3, 16, 16, 16], strides: [4096, 256, 16, 1] : memref<bf16, #hivm.address_space<cb>> to memref<3x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>>
    %cast_167 = memref.cast %reinterpret_cast_134 : memref<16x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_168 = memref.cast %reinterpret_cast_166 : memref<3x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t(%cast_167, %cast_168, %13, %7, %4, %6, %4, %5) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %210]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %210]
    %cast_169 = memref.cast %reinterpret_cast_161 : memref<3x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 6144>, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_170 = memref.cast %reinterpret_cast_166 : memref<3x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_171 = memref.cast %reinterpret_cast_139 : memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_169, %cast_170, %cast_171, %15, %4, %5, %18) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %210]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, %199]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, %199]
    cf.cond_br %191, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 5
    cf.br ^bb15
  ^bb15:  // 2 preds: ^bb13, ^bb14
    %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_0 to offset: [%3], sizes: [48, 256], strides: [256, 1] : memref<f32, #hivm.address_space<ub>> to memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %cast_173 = memref.cast %reinterpret_cast_139 : memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_174 = memref.cast %reinterpret_cast_172 : memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_173, %cast_174, %18, %9, %18, %35, %34, %10, %18, %8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %199]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 4
    %214 = llvm.add %94, %196 : i32
    %215 = llvm.srem %214, %28  : i32
    %216 = llvm.sext %215 : i32 to i64
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %216]
    %c28672 = arith.constant 28672 : index
    %217 = arith.addi %185, %c28672 : index
    %reinterpret_cast_175 = memref.reinterpret_cast %base_buffer_125 to offset: [%217], sizes: [16, 7, 16, 16], strides: [1792, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<16x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %218 = llvm.mul %215, %32 : i32
    %219 = llvm.sext %218 : i32 to i64
    %220 = builtin.unrealized_conversion_cast %219 : i64 to index
    %base_buffer_176, %offset_177, %sizes_178, %strides_179 = memref.extract_strided_metadata %55 : memref<65536xf32, #hivm.address_space<cc>> -> memref<f32, #hivm.address_space<cc>>, index, index, index
    %reinterpret_cast_180 = memref.reinterpret_cast %base_buffer_176 to offset: [%220], sizes: [16, 6, 16, 16], strides: [1536, 256, 16, 1] : memref<f32, #hivm.address_space<cc>> to memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %204]
    %cast_181 = memref.cast %reinterpret_cast_175 : memref<16x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_182 = memref.cast %reinterpret_cast_151 : memref<4x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t(%cast_181, %cast_182, %18, %7, %7, %6, %7, %5) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %204]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %204]
    %cast_183 = memref.cast %reinterpret_cast_146 : memref<4x6x16x16xbf16, strided<[1536, 256, 16, 1]>, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_184 = memref.cast %reinterpret_cast_151 : memref<4x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_185 = memref.cast %reinterpret_cast_180 : memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_183, %cast_184, %cast_185, %15, %7, %5, %13) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %204]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, %210]
    %cast_186 = memref.cast %reinterpret_cast_175 : memref<16x7x16x16xbf16, strided<[1792, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_187 = memref.cast %reinterpret_cast_166 : memref<3x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    call @l12l0_cbuf_to_cb_bfloat16_t(%cast_186, %cast_187, %13, %7, %4, %6, %4, %5) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) -> ()
    %221 = llvm.sext %182 : i32 to i64
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %221]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, %210]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, %210]
    %cast_188 = memref.cast %reinterpret_cast_161 : memref<3x6x16x16xbf16, strided<[1536, 256, 16, 1], offset: 6144>, #hivm.address_space<ca>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>
    %cast_189 = memref.cast %reinterpret_cast_166 : memref<3x16x16x16xbf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cb>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>
    %cast_190 = memref.cast %reinterpret_cast_180 : memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mmadl0_a5_bfloat16_t_to_float(%cast_188, %cast_189, %cast_190, %15, %4, %5, %18) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, %210]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, %216]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, %216]
    cf.cond_br %191, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 9
    cf.br ^bb17
  ^bb17:  // 2 preds: ^bb15, ^bb16
    %reinterpret_cast_191 = memref.reinterpret_cast %base_buffer_0 to offset: [%1], sizes: [48, 256], strides: [256, 1] : memref<f32, #hivm.address_space<ub>> to memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %cast_192 = memref.cast %reinterpret_cast_180 : memref<16x6x16x16xf32, strided<[1536, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_193 = memref.cast %reinterpret_cast_191 : memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_192, %cast_193, %18, %9, %18, %35, %34, %10, %18, %8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %216]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 8
    cf.br ^bb18
  ^bb18:  // 2 preds: ^bb12, ^bb17
    %222 = arith.addi %87, %45 overflow<nsw> : i32
    cf.br ^bb3(%222 : i32)
  ^bb19:  // pred: ^bb3
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 6
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 7
    %223 = arith.addi %66, %45 overflow<nsw> : i32
    cf.br ^bb1(%223 : i32)
  ^bb20:  // pred: ^bb1
    return
  }
  func.func private @nd2nz_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @l12l0_cbuf_to_ca_bfloat16_t(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @l12l0_cbuf_to_cb_bfloat16_t_transpose(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mmadl0_a5_bfloat16_t_to_float(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @l12l0_cbuf_to_cb_bfloat16_t(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_cbuf_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_3d_float(memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float_nocache(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_3d_float_nocache(memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @attention_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: memref<?xi32, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xbf16, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(8192 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(18432 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(24832 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(16384 : index) : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = llvm.mlir.constant(55360 : index) : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = llvm.mlir.constant(55296 : index) : i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %12 = llvm.mlir.constant(30720 : index) : i64
    %13 = builtin.unrealized_conversion_cast %12 : i64 to index
    %14 = llvm.mlir.constant(6144 : index) : i64
    %15 = builtin.unrealized_conversion_cast %14 : i64 to index
    %16 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %17 = llvm.mlir.constant(238592 : i64) : i64
    %18 = llvm.mlir.constant(222208 : i64) : i64
    %19 = llvm.mlir.constant(0 : i64) : i64
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(320 : i32) : i32
    %22 = llvm.mlir.constant(10240 : i32) : i32
    %23 = llvm.mlir.constant(1280 : i32) : i32
    %24 = llvm.mlir.constant(96 : i32) : i32
    %25 = llvm.mlir.constant(112 : i32) : i32
    %26 = llvm.mlir.constant(48 : i32) : i32
    %27 = llvm.mlir.constant(64 : i32) : i32
    %28 = llvm.mlir.constant(3 : i32) : i32
    %29 = llvm.mlir.constant(2 : i32) : i32
    %30 = llvm.mlir.constant(68096 : i32) : i32
    %31 = llvm.mlir.constant(57344 : i32) : i32
    %32 = llvm.mlir.constant(10 : i32) : i32
    %33 = llvm.mlir.constant(16 : i32) : i32
    %34 = llvm.mlir.constant(48 : i64) : i64
    %35 = llvm.mlir.constant(24 : i64) : i64
    %36 = llvm.mlir.constant(16 : i64) : i64
    %37 = llvm.mlir.constant(5 : i64) : i64
    %38 = llvm.mlir.constant(512 : i32) : i32
    %39 = llvm.mlir.constant(8192 : i64) : i64
    %40 = llvm.mlir.constant(-1 : i32) : i32
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = builtin.unrealized_conversion_cast %41 : i64 to index
    %43 = llvm.mlir.constant(-3 : i32) : i32
    %44 = llvm.mlir.constant(768 : i64) : i64
    %45 = llvm.mlir.constant(1279 : i32) : i32
    %46 = llvm.mlir.constant(80 : i32) : i32
    %47 = llvm.mlir.constant(40960 : i32) : i32
    %48 = llvm.mlir.constant(8192 : i32) : i32
    %49 = llvm.mlir.constant(111 : i32) : i32
    %50 = llvm.mlir.constant(1 : i32) : i32
    %51 = hivm.hir.get_block_idx -> i64
    %52 = llvm.trunc %51 : i64 to i32
    %53 = llvm.srem %52, %arg11  : i32
    %54 = hivm.hir.pointer_cast(%19) : memref<204288xbf16, #hivm.address_space<cbuf>>
    %55 = hivm.hir.pointer_cast(%19) : memref<55552xf32, #hivm.address_space<ub>>
    %56 = hivm.hir.pointer_cast(%18) : memref<8192xbf16, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%42], sizes: [48, 128], strides: [128, 1] : memref<f32, #hivm.address_space<ub>> to memref<48x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %base_buffer_4, %offset_5, %sizes_6, %strides_7 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_8 = memref.reinterpret_cast %base_buffer_4 to offset: [%13], sizes: [48, 512], strides: [512, 1] : memref<f32, #hivm.address_space<ub>> to memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_9, %offset_10, %sizes_11, %strides_12 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_13 = memref.reinterpret_cast %base_buffer_9 to offset: [55296], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 55296>, #hivm.address_space<ub>>
    %cast = memref.cast %reinterpret_cast_13 : memref<64xf32, strided<[1], offset: 55296>, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_14, %offset_15, %sizes_16, %strides_17 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_18 = memref.reinterpret_cast %base_buffer_14 to offset: [55360], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 55360>, #hivm.address_space<ub>>
    %cast_19 = memref.cast %reinterpret_cast_18 : memref<64xf32, strided<[1], offset: 55360>, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %56 : memref<8192xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_24 = memref.reinterpret_cast %base_buffer_20 to offset: [%42], sizes: [8, 784], strides: [784, 1] : memref<bf16, #hivm.address_space<ub>> to memref<8x784xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>
    %57 = hivm.hir.pointer_cast(%17) : memref<128xf32, #hivm.address_space<ub>>
    %58 = hivm.hir.get_sub_block_idx -> i64
    %59 = llvm.mul %53, %21 : i32
    %60 = llvm.intr.smin(%59, %22)  : (i32, i32) -> i32
    %61 = llvm.add %60, %21 : i32
    %62 = llvm.intr.smin(%61, %22)  : (i32, i32) -> i32
    %63 = llvm.sdiv %60, %23  : i32
    %64 = llvm.icmp "slt" %60, %62 : i32
    %65 = llvm.add %62, %45 : i32
    %66 = llvm.sdiv %65, %23  : i32
    %67 = llvm.select %64, %66, %63 : i1, i32
    cf.br ^bb1(%63 : i32)
  ^bb1(%68: i32):  // 2 preds: ^bb0, ^bb27
    %69 = arith.cmpi slt, %68, %67 : i32
    cf.cond_br %69, ^bb2, ^bb28
  ^bb2:  // pred: ^bb1
    %70 = llvm.mul %68, %23 : i32
    %71 = llvm.sdiv %70, %21  : i32
    %72 = llvm.sub %53, %71 : i32
    %73 = llvm.mul %68, %24 : i32
    %74 = llvm.sext %68 : i32 to i64
    %75 = builtin.unrealized_conversion_cast %74 : i64 to index
    %reinterpret_cast_25 = memref.reinterpret_cast %arg5 to offset: [%75], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %76 = memref.load %reinterpret_cast_25[%42] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %77 = llvm.add %76, %49 : i32
    %78 = llvm.sdiv %77, %25  : i32
    %79 = llvm.intr.smax(%70, %60)  : (i32, i32) -> i32
    %80 = llvm.add %70, %78 : i32
    %81 = llvm.intr.smin(%80, %62)  : (i32, i32) -> i32
    %82 = llvm.sub %79, %70 : i32
    %83 = llvm.mul %82, %25 : i32
    %84 = llvm.sub %81, %79 : i32
    %85 = llvm.intr.smax(%84, %20)  : (i32, i32) -> i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    call @attention_scope_12(%cast, %cast_19) {hivm.vector_function} : (memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> ()
    %86 = llvm.icmp "eq" %85, %20 : i32
    cf.cond_br %86, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    call @attention_scope_11(%reinterpret_cast_8) {hivm.vector_function} : (memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %87 = llvm.add %85, %50 : i32
    cf.br ^bb5(%40 : i32)
  ^bb5(%88: i32):  // 2 preds: ^bb4, ^bb26
    %89 = arith.cmpi slt, %88, %87 : i32
    cf.cond_br %89, ^bb6, ^bb27
  ^bb6:  // pred: ^bb5
    %90 = llvm.icmp "sge" %88, %20 : i32
    %91 = llvm.icmp "slt" %88, %85 : i32
    %92 = llvm.and %90, %91  : i1
    %93 = llvm.icmp "sgt" %88, %20 : i32
    %94 = llvm.srem %88, %28  : i32
    %95 = llvm.srem %88, %29  : i32
    cf.cond_br %92, ^bb7, ^bb14
  ^bb7:  // pred: ^bb6
    %96 = llvm.mul %58, %34 : i64
    %97 = llvm.add %76, %43 : i32
    %98 = llvm.sub %97, %83 : i32
    %99 = llvm.mul %88, %25 : i32
    %100 = llvm.sub %98, %99 : i32
    %101 = llvm.sdiv %96, %35  : i64
    %102 = llvm.sext %100 : i32 to i64
    %103 = llvm.add %102, %101 : i64
    %104 = llvm.trunc %103 : i64 to i32
    %105 = llvm.mul %95, %27 : i32
    %106 = llvm.sext %105 : i32 to i64
    %107 = builtin.unrealized_conversion_cast %106 : i64 to index
    %base_buffer_26, %offset_27, %sizes_28, %strides_29 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %c55424 = arith.constant 55424 : index
    %108 = arith.addi %107, %c55424 : index
    %reinterpret_cast_30 = memref.reinterpret_cast %base_buffer_26 to offset: [%108], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_31, %offset_32, %sizes_33, %strides_34 = memref.extract_strided_metadata %57 : memref<128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_35 = memref.reinterpret_cast %base_buffer_31 to offset: [%107], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 0
    %109 = llvm.icmp "slt" %104, %25 : i32
    %110 = llvm.zext %109 : i1 to i32
    %111 = llvm.icmp "eq" %110, %20 : i32
    cf.cond_br %111, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    call @attention_scope_10(%reinterpret_cast, %cast, %reinterpret_cast_35, %reinterpret_cast_30, %reinterpret_cast_24, %cast_19) {hivm.vector_function} : (memref<48x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8x784xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    %112 = llvm.icmp "eq" %110, %50 : i32
    cf.cond_br %112, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    call @attention_scope_9(%104, %reinterpret_cast, %cast, %reinterpret_cast_35, %reinterpret_cast_30, %reinterpret_cast_24, %cast_19) {hivm.vector_function} : (i32, memref<48x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8x784xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %113 = llvm.add %88, %50 : i32
    %114 = llvm.icmp "slt" %113, %85 : i32
    cf.cond_br %114, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 1
    cf.br ^bb13
  ^bb13:  // 2 preds: ^bb11, ^bb12
    %115 = llvm.mul %94, %30 : i32
    %116 = llvm.add %115, %31 : i32
    %117 = llvm.sext %116 : i32 to i64
    %118 = builtin.unrealized_conversion_cast %117 : i64 to index
    %base_buffer_36, %offset_37, %sizes_38, %strides_39 = memref.extract_strided_metadata %54 : memref<204288xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_40 = memref.reinterpret_cast %base_buffer_20 to offset: [%42], sizes: [7, 768], strides: [784, 1] : memref<bf16, #hivm.address_space<ub>> to memref<7x768xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>
    %119 = llvm.mul %58, %44 : i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    %121 = arith.addi %118, %120 : index
    %reinterpret_cast_41 = memref.reinterpret_cast %base_buffer_36 to offset: [%121], sizes: [7, 768], strides: [1536, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<7x768xbf16, strided<[1536, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_42 = memref.cast %reinterpret_cast_40 : memref<7x768xbf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_43 = memref.cast %reinterpret_cast_41 : memref<7x768xbf16, strided<[1536, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_42, %cast_43) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %122 = llvm.add %94, %32 : i32
    %123 = llvm.sext %122 : i32 to i64
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = %123
    cf.br ^bb14
  ^bb14:  // 2 preds: ^bb6, ^bb13
    cf.cond_br %93, ^bb15, ^bb26
  ^bb15:  // pred: ^bb14
    %124 = llvm.sub %50, %95 : i32
    %125 = llvm.mul %124, %27 : i32
    %126 = llvm.sext %125 : i32 to i64
    %127 = builtin.unrealized_conversion_cast %126 : i64 to index
    %base_buffer_44, %offset_45, %sizes_46, %strides_47 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %c55424_48 = arith.constant 55424 : index
    %128 = arith.addi %127, %c55424_48 : index
    %reinterpret_cast_49 = memref.reinterpret_cast %base_buffer_44 to offset: [%128], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 4
    %reinterpret_cast_50 = memref.reinterpret_cast %base_buffer_0 to offset: [%15], sizes: [48, 256], strides: [256, 1] : memref<f32, #hivm.address_space<ub>> to memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %129 = llvm.icmp "eq" %88, %50 : i32
    cf.cond_br %129, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    call @attention_scope_8(%reinterpret_cast_50, %reinterpret_cast_8) {hivm.vector_function} : (memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb18
  ^bb17:  // pred: ^bb15
    call @attention_scope_7(%reinterpret_cast_49, %reinterpret_cast_50, %reinterpret_cast_8) {hivm.vector_function} : (memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb18
  ^bb18:  // 2 preds: ^bb16, ^bb17
    cf.cond_br %92, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 5
    cf.br ^bb20
  ^bb20:  // 2 preds: ^bb18, ^bb19
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 8
    %reinterpret_cast_51 = memref.reinterpret_cast %base_buffer_0 to offset: [%3], sizes: [48, 256], strides: [256, 1] : memref<f32, #hivm.address_space<ub>> to memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    cf.cond_br %129, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    call @attention_scope_6(%reinterpret_cast_51, %reinterpret_cast_8) {hivm.vector_function} : (memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb23
  ^bb22:  // pred: ^bb20
    call @attention_scope_5(%reinterpret_cast_49, %reinterpret_cast_51, %reinterpret_cast_8) {hivm.vector_function} : (memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<48x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb23
  ^bb23:  // 2 preds: ^bb21, ^bb22
    cf.cond_br %92, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 9
    cf.br ^bb25
  ^bb25:  // 2 preds: ^bb23, ^bb24
    cf.br ^bb26
  ^bb26:  // 2 preds: ^bb14, ^bb25
    %130 = arith.addi %88, %50 overflow<nsw> : i32
    cf.br ^bb5(%130 : i32)
  ^bb27:  // pred: ^bb5
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %131 = llvm.mul %58, %34 : i64
    %132 = llvm.sext %73 : i32 to i64
    %133 = llvm.add %132, %131 : i64
    %134 = llvm.sdiv %133, %36  : i64
    %135 = llvm.mul %134, %37 : i64
    %136 = llvm.sext %72 : i32 to i64
    %137 = llvm.add %135, %136 : i64
    %138 = llvm.mul %137, %36 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    %base_buffer_52, %offset_53, %sizes_54, %strides_55 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_56 = memref.reinterpret_cast %base_buffer_52 to offset: [%13], sizes: [3, 16, 512], strides: [8192, 512, 1] : memref<f32, #hivm.address_space<ub>> to memref<3x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>>
    %140 = llvm.mul %137, %39 : i64
    %141 = builtin.unrealized_conversion_cast %140 : i64 to index
    %reinterpret_cast_57 = memref.reinterpret_cast %arg6 to offset: [%141], sizes: [3, 16, 512], strides: [40960, 512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x16x512xf32, strided<[40960, 512, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_58 = memref.cast %reinterpret_cast_56 : memref<3x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_59 = memref.cast %reinterpret_cast_57 : memref<3x16x512xf32, strided<[40960, 512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_3d_float(%cast_58, %cast_59, %20) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    %base_buffer_60, %offset_61, %sizes_62, %strides_63 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_64 = memref.reinterpret_cast %base_buffer_60 to offset: [%11], sizes: [3, 16], strides: [16, 1] : memref<f32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %reinterpret_cast_65 = memref.reinterpret_cast %arg7 to offset: [%139], sizes: [3, 16], strides: [80, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x16xf32, strided<[80, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_66 = memref.cast %reinterpret_cast_64 : memref<3x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_67 = memref.cast %reinterpret_cast_65 : memref<3x16xf32, strided<[80, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_float(%cast_66, %cast_67, %20) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    %base_buffer_68, %offset_69, %sizes_70, %strides_71 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_72 = memref.reinterpret_cast %base_buffer_68 to offset: [%9], sizes: [3, 16], strides: [16, 1] : memref<f32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %reinterpret_cast_73 = memref.reinterpret_cast %arg8 to offset: [%139], sizes: [3, 16], strides: [80, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x16xf32, strided<[80, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_74 = memref.cast %reinterpret_cast_72 : memref<3x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_75 = memref.cast %reinterpret_cast_73 : memref<3x16xf32, strided<[80, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_float(%cast_74, %cast_75, %20) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 6
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 7
    %142 = arith.addi %68, %50 overflow<nsw> : i32
    cf.br ^bb1(%142 : i32)
  ^bb28:  // pred: ^bb1
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 14 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 14 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    %143 = llvm.mul %53, %29 : i32
    %144 = llvm.trunc %58 : i64 to i32
    %145 = llvm.add %143, %144 : i32
    %base_buffer_76, %offset_77, %sizes_78, %strides_79 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_80 = memref.reinterpret_cast %base_buffer_76 to offset: [%42], sizes: [2, 16, 512], strides: [8192, 512, 1] : memref<f32, #hivm.address_space<ub>> to memref<2x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_81, %offset_82, %sizes_83, %strides_84 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_85 = memref.reinterpret_cast %base_buffer_81 to offset: [%7], sizes: [16, 512], strides: [512, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_86, %offset_87, %sizes_88, %strides_89 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_90 = memref.reinterpret_cast %base_buffer_86 to offset: [24576], sizes: [128], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<128xf32, strided<[1], offset: 24576>, #hivm.address_space<ub>>
    %cast_91 = memref.cast %reinterpret_cast_90 : memref<128xf32, strided<[1], offset: 24576>, #hivm.address_space<ub>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_92, %offset_93, %sizes_94, %strides_95 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_96 = memref.reinterpret_cast %base_buffer_92 to offset: [24704], sizes: [128], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<128xf32, strided<[1], offset: 24704>, #hivm.address_space<ub>>
    %cast_97 = memref.cast %reinterpret_cast_96 : memref<128xf32, strided<[1], offset: 24704>, #hivm.address_space<ub>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_98, %offset_99, %sizes_100, %strides_101 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_102 = memref.reinterpret_cast %base_buffer_98 to offset: [%5], sizes: [5, 16], strides: [16, 1] : memref<f32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_103, %offset_104, %sizes_105, %strides_106 = memref.extract_strided_metadata %56 : memref<8192xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_107 = memref.reinterpret_cast %base_buffer_103 to offset: [%42], sizes: [16, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<ub>> to memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_108, %offset_109, %sizes_110, %strides_111 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_112 = memref.reinterpret_cast %base_buffer_108 to offset: [24928], sizes: [16], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: 24928>, #hivm.address_space<ub>>
    %cast_113 = memref.cast %reinterpret_cast_112 : memref<16xf32, strided<[1], offset: 24928>, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    call @attention_scope_4(%cast_91, %cast_97) {hivm.vector_function} : (memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %base_buffer_114, %offset_115, %sizes_116, %strides_117 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_118 = memref.reinterpret_cast %base_buffer_114 to offset: [24912], sizes: [16], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: 24912>, #hivm.address_space<ub>>
    %cast_119 = memref.cast %reinterpret_cast_118 : memref<16xf32, strided<[1], offset: 24912>, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    cf.br ^bb29(%145 : i32)
  ^bb29(%146: i32):  // 2 preds: ^bb28, ^bb39
    %147 = arith.cmpi slt, %146, %26 : i32
    cf.cond_br %147, ^bb30, ^bb40
  ^bb30:  // pred: ^bb29
    %148 = llvm.mul %146, %33 : i32
    %149 = llvm.sdiv %148, %24  : i32
    %150 = llvm.mul %149, %23 : i32
    %151 = llvm.add %150, %45 : i32
    %152 = llvm.sdiv %151, %21  : i32
    %153 = llvm.sdiv %150, %21  : i32
    %154 = llvm.sub %152, %153 : i32
    %155 = llvm.add %154, %50 : i32
    %156 = llvm.mul %146, %46 : i32
    %157 = llvm.sext %156 : i32 to i64
    %158 = builtin.unrealized_conversion_cast %157 : i64 to index
    %reinterpret_cast_120 = memref.reinterpret_cast %arg7 to offset: [%158], sizes: [80], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<80xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_121, %offset_122, %sizes_123, %strides_124 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_125 = memref.reinterpret_cast %base_buffer_121 to offset: [24576], sizes: [80], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<80xf32, strided<[1], offset: 24576>, #hivm.address_space<ub>>
    %cast_126 = memref.cast %reinterpret_cast_120 : memref<80xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_127 = memref.cast %reinterpret_cast_125 : memref<80xf32, strided<[1], offset: 24576>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float_nocache(%cast_126, %cast_127, %20, %16, %42, %20) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    %reinterpret_cast_128 = memref.reinterpret_cast %arg8 to offset: [%158], sizes: [80], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<80xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_129, %offset_130, %sizes_131, %strides_132 = memref.extract_strided_metadata %55 : memref<55552xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_133 = memref.reinterpret_cast %base_buffer_129 to offset: [24704], sizes: [80], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<80xf32, strided<[1], offset: 24704>, #hivm.address_space<ub>>
    %cast_134 = memref.cast %reinterpret_cast_128 : memref<80xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_135 = memref.cast %reinterpret_cast_133 : memref<80xf32, strided<[1], offset: 24704>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float_nocache(%cast_134, %cast_135, %20, %16, %42, %20) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    %159 = llvm.icmp "sgt" %155, %20 : i32
    cf.cond_br %159, ^bb31, ^bb32
  ^bb31:  // pred: ^bb30
    %160 = llvm.mul %146, %47 : i32
    %161 = llvm.sext %160 : i32 to i64
    %162 = builtin.unrealized_conversion_cast %161 : i64 to index
    %reinterpret_cast_136 = memref.reinterpret_cast %arg6 to offset: [%162], sizes: [1, 16, 512], strides: [8192, 512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_137 = memref.reinterpret_cast %base_buffer_76 to offset: [%42], sizes: [1, 16, 512], strides: [8192, 512, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>>
    %cast_138 = memref.cast %reinterpret_cast_136 : memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_139 = memref.cast %reinterpret_cast_137 : memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_3d_float_nocache(%cast_138, %cast_139, %20, %16, %42, %20) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
    cf.br ^bb32
  ^bb32:  // 2 preds: ^bb30, ^bb31
    %163 = llvm.icmp "sgt" %155, %50 : i32
    cf.cond_br %163, ^bb33, ^bb34
  ^bb33:  // pred: ^bb32
    %164 = llvm.add %156, %33 : i32
    %165 = llvm.mul %164, %38 : i32
    %166 = llvm.sext %165 : i32 to i64
    %167 = builtin.unrealized_conversion_cast %166 : i64 to index
    %reinterpret_cast_140 = memref.reinterpret_cast %arg6 to offset: [%167], sizes: [1, 16, 512], strides: [8192, 512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_141 = memref.reinterpret_cast %base_buffer_76 to offset: [%1], sizes: [1, 16, 512], strides: [8192, 512, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>>
    %cast_142 = memref.cast %reinterpret_cast_140 : memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_143 = memref.cast %reinterpret_cast_141 : memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_3d_float_nocache(%cast_142, %cast_143, %20, %16, %42, %20) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID4>]
    cf.br ^bb34
  ^bb34:  // 2 preds: ^bb32, ^bb33
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    call @attention_scope_3(%155, %cast_91, %cast_97, %reinterpret_cast_102, %cast_113, %cast_119) {hivm.vector_function} : (i32, memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<5x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> ()
    call @attention_scope_2(%reinterpret_cast_85) {hivm.vector_function} : (memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    cf.br ^bb35(%20 : i32)
  ^bb35(%168: i32):  // 2 preds: ^bb34, ^bb38
    %169 = arith.cmpi slt, %168, %155 : i32
    cf.cond_br %169, ^bb36, ^bb39
  ^bb36:  // pred: ^bb35
    %170 = llvm.srem %168, %29  : i32
    %171 = llvm.add %170, %28 : i32
    %172 = llvm.sext %171 : i32 to i64
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, %172]
    call @attention_scope_1(%168, %reinterpret_cast_102, %170, %reinterpret_cast_80, %reinterpret_cast_85) {hivm.vector_function} : (i32, memref<5x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, i32, memref<2x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>>, memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    %173 = llvm.add %168, %29 : i32
    %174 = llvm.icmp "slt" %173, %155 : i32
    cf.cond_br %174, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %172]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %172]
    %175 = llvm.mul %173, %33 : i32
    %176 = llvm.add %156, %175 : i32
    %177 = llvm.mul %176, %38 : i32
    %178 = llvm.sext %177 : i32 to i64
    %179 = builtin.unrealized_conversion_cast %178 : i64 to index
    %reinterpret_cast_144 = memref.reinterpret_cast %arg6 to offset: [%179], sizes: [1, 16, 512], strides: [8192, 512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
    %180 = llvm.sext %170 : i32 to i64
    %181 = builtin.unrealized_conversion_cast %180 : i64 to index
    %c8192 = arith.constant 8192 : index
    %182 = arith.muli %181, %c8192 : index
    %183 = arith.addi %42, %182 : index
    %reinterpret_cast_145 = memref.reinterpret_cast %base_buffer_76 to offset: [%183], sizes: [1, 16, 512], strides: [8192, 512, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>>
    %cast_146 = memref.cast %reinterpret_cast_144 : memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_147 = memref.cast %reinterpret_cast_145 : memref<1x16x512xf32, strided<[8192, 512, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_3d_float_nocache(%cast_146, %cast_147, %20, %16, %42, %20) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, %172]
    cf.br ^bb38
  ^bb38:  // 2 preds: ^bb36, ^bb37
    %184 = arith.addi %168, %50 overflow<nsw> : i32
    cf.br ^bb35(%184 : i32)
  ^bb39:  // pred: ^bb35
    call @attention_scope_0(%cast_113, %reinterpret_cast_85, %reinterpret_cast_107) {hivm.vector_function} : (memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>, memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
    %185 = llvm.mul %146, %48 : i32
    %186 = llvm.sext %185 : i32 to i64
    %187 = builtin.unrealized_conversion_cast %186 : i64 to index
    %reinterpret_cast_148 = memref.reinterpret_cast %arg9 to offset: [%187], sizes: [16, 512], strides: [512, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_149 = memref.cast %reinterpret_cast_107 : memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_150 = memref.cast %reinterpret_cast_148 : memref<16x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_bfloat16_t(%cast_149, %cast_150, %20) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    %188 = llvm.sext %148 : i32 to i64
    %189 = builtin.unrealized_conversion_cast %188 : i64 to index
    %reinterpret_cast_151 = memref.reinterpret_cast %arg10 to offset: [%189], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %cast_152 = memref.cast %reinterpret_cast_118 : memref<16xf32, strided<[1], offset: 24912>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_153 = memref.cast %reinterpret_cast_151 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_152, %cast_153, %20) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %190 = arith.addi %146, %27 overflow<nsw> : i32
    cf.br ^bb29(%190 : i32)
  ^bb40:  // pred: ^bb29
    return
  }
}
