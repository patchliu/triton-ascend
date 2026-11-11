module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.mode = "tla", hacc.target = #hacc.target<"Ascend950DT_9581">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.aic.c310.O2.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.aiv.c310.O2.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/host.O2.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.mix.aic.c310.O2.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/liupengcheng/workspace/npuir/temp/20261111-mla-final.hBf0WW/npuir-package/lib/meta_op.mix.aiv.c310.O2.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @attention_splits_scope_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<48x512xf32, #hivm.address_space<ub>>, %arg2: memref<48x512xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<48x512xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<48x512xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(48 : i32) : i32
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
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(3 : i32) : i32
    %22 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%18, %19, %21, %20) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %23 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(512 : index) : i64
    %25 = llvm.mul %16, %24 : i64
    %26 = llvm.add %25, %6 : i64
    %27 = llvm.getelementptr %23[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%27, %28, %30, %29) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %32 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(512 : index) : i64
    %34 = llvm.mul %16, %33 : i64
    %35 = llvm.add %34, %6 : i64
    %36 = llvm.getelementptr %32[%35] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%36, %37, %39, %38) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %41 = llvm.mlir.constant(0 : i32) : i32
    %42 = llvm.mlir.constant(0 : i32) : i32
    %43 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%41, %42) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %44 = builtin.unrealized_conversion_cast %43 : vector<256xi1> to vector<64xi1>
    %45 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%31, %40, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %46 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.mlir.constant(512 : index) : i64
    %48 = llvm.mul %16, %47 : i64
    %49 = llvm.add %48, %6 : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %51 = llvm.mlir.constant(0 : i32) : i32
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%45, %50, %51, %53, %52, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %54 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mlir.constant(512 : index) : i64
    %56 = llvm.mul %16, %55 : i64
    %57 = llvm.add %56, %7 : i64
    %58 = llvm.getelementptr %54[%57] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.mlir.constant(0 : i32) : i32
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%58, %59, %61, %60) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %63 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mlir.constant(512 : index) : i64
    %65 = llvm.mul %16, %64 : i64
    %66 = llvm.add %65, %7 : i64
    %67 = llvm.getelementptr %63[%66] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%67, %68, %70, %69) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %72 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%62, %71, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %73 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mlir.constant(512 : index) : i64
    %75 = llvm.mul %16, %74 : i64
    %76 = llvm.add %75, %7 : i64
    %77 = llvm.getelementptr %73[%76] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(0 : i32) : i32
    %80 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%72, %77, %78, %80, %79, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %81 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(512 : index) : i64
    %83 = llvm.mul %16, %82 : i64
    %84 = llvm.add %83, %8 : i64
    %85 = llvm.getelementptr %81[%84] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %86 = llvm.mlir.constant(0 : i32) : i32
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%85, %86, %88, %87) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %90 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.mlir.constant(512 : index) : i64
    %92 = llvm.mul %16, %91 : i64
    %93 = llvm.add %92, %8 : i64
    %94 = llvm.getelementptr %90[%93] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %95 = llvm.mlir.constant(0 : i32) : i32
    %96 = llvm.mlir.constant(0 : i32) : i32
    %97 = llvm.mlir.constant(0 : i32) : i32
    %98 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%94, %95, %97, %96) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %99 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%89, %98, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %100 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mlir.constant(512 : index) : i64
    %102 = llvm.mul %16, %101 : i64
    %103 = llvm.add %102, %8 : i64
    %104 = llvm.getelementptr %100[%103] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %105 = llvm.mlir.constant(0 : i32) : i32
    %106 = llvm.mlir.constant(0 : i32) : i32
    %107 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%99, %104, %105, %107, %106, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %108 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.mlir.constant(512 : index) : i64
    %110 = llvm.mul %16, %109 : i64
    %111 = llvm.add %110, %9 : i64
    %112 = llvm.getelementptr %108[%111] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %113 = llvm.mlir.constant(0 : i32) : i32
    %114 = llvm.mlir.constant(0 : i32) : i32
    %115 = llvm.mlir.constant(0 : i32) : i32
    %116 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%112, %113, %115, %114) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %117 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.mlir.constant(512 : index) : i64
    %119 = llvm.mul %16, %118 : i64
    %120 = llvm.add %119, %9 : i64
    %121 = llvm.getelementptr %117[%120] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %122 = llvm.mlir.constant(0 : i32) : i32
    %123 = llvm.mlir.constant(0 : i32) : i32
    %124 = llvm.mlir.constant(0 : i32) : i32
    %125 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%121, %122, %124, %123) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %126 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%116, %125, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %127 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.mlir.constant(512 : index) : i64
    %129 = llvm.mul %16, %128 : i64
    %130 = llvm.add %129, %9 : i64
    %131 = llvm.getelementptr %127[%130] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %132 = llvm.mlir.constant(0 : i32) : i32
    %133 = llvm.mlir.constant(0 : i32) : i32
    %134 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%126, %131, %132, %134, %133, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %135 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(512 : index) : i64
    %137 = llvm.mul %16, %136 : i64
    %138 = llvm.add %137, %10 : i64
    %139 = llvm.getelementptr %135[%138] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %140 = llvm.mlir.constant(0 : i32) : i32
    %141 = llvm.mlir.constant(0 : i32) : i32
    %142 = llvm.mlir.constant(0 : i32) : i32
    %143 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%139, %140, %142, %141) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %144 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.mlir.constant(512 : index) : i64
    %146 = llvm.mul %16, %145 : i64
    %147 = llvm.add %146, %10 : i64
    %148 = llvm.getelementptr %144[%147] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %149 = llvm.mlir.constant(0 : i32) : i32
    %150 = llvm.mlir.constant(0 : i32) : i32
    %151 = llvm.mlir.constant(0 : i32) : i32
    %152 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%148, %149, %151, %150) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %153 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%143, %152, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %154 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.mlir.constant(512 : index) : i64
    %156 = llvm.mul %16, %155 : i64
    %157 = llvm.add %156, %10 : i64
    %158 = llvm.getelementptr %154[%157] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %159 = llvm.mlir.constant(0 : i32) : i32
    %160 = llvm.mlir.constant(0 : i32) : i32
    %161 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%153, %158, %159, %161, %160, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %162 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.mlir.constant(512 : index) : i64
    %164 = llvm.mul %16, %163 : i64
    %165 = llvm.add %164, %11 : i64
    %166 = llvm.getelementptr %162[%165] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %167 = llvm.mlir.constant(0 : i32) : i32
    %168 = llvm.mlir.constant(0 : i32) : i32
    %169 = llvm.mlir.constant(0 : i32) : i32
    %170 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%166, %167, %169, %168) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %171 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.mlir.constant(512 : index) : i64
    %173 = llvm.mul %16, %172 : i64
    %174 = llvm.add %173, %11 : i64
    %175 = llvm.getelementptr %171[%174] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %176 = llvm.mlir.constant(0 : i32) : i32
    %177 = llvm.mlir.constant(0 : i32) : i32
    %178 = llvm.mlir.constant(0 : i32) : i32
    %179 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%175, %176, %178, %177) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %180 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%170, %179, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %181 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.mlir.constant(512 : index) : i64
    %183 = llvm.mul %16, %182 : i64
    %184 = llvm.add %183, %11 : i64
    %185 = llvm.getelementptr %181[%184] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %186 = llvm.mlir.constant(0 : i32) : i32
    %187 = llvm.mlir.constant(0 : i32) : i32
    %188 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%180, %185, %186, %188, %187, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %189 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.mlir.constant(512 : index) : i64
    %191 = llvm.mul %16, %190 : i64
    %192 = llvm.add %191, %12 : i64
    %193 = llvm.getelementptr %189[%192] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %194 = llvm.mlir.constant(0 : i32) : i32
    %195 = llvm.mlir.constant(0 : i32) : i32
    %196 = llvm.mlir.constant(0 : i32) : i32
    %197 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%193, %194, %196, %195) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %198 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %199 = llvm.mlir.constant(512 : index) : i64
    %200 = llvm.mul %16, %199 : i64
    %201 = llvm.add %200, %12 : i64
    %202 = llvm.getelementptr %198[%201] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %203 = llvm.mlir.constant(0 : i32) : i32
    %204 = llvm.mlir.constant(0 : i32) : i32
    %205 = llvm.mlir.constant(0 : i32) : i32
    %206 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%202, %203, %205, %204) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %207 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%197, %206, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %208 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %209 = llvm.mlir.constant(512 : index) : i64
    %210 = llvm.mul %16, %209 : i64
    %211 = llvm.add %210, %12 : i64
    %212 = llvm.getelementptr %208[%211] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %213 = llvm.mlir.constant(0 : i32) : i32
    %214 = llvm.mlir.constant(0 : i32) : i32
    %215 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%207, %212, %213, %215, %214, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %216 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %217 = llvm.mlir.constant(512 : index) : i64
    %218 = llvm.mul %16, %217 : i64
    %219 = llvm.add %218, %13 : i64
    %220 = llvm.getelementptr %216[%219] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %221 = llvm.mlir.constant(0 : i32) : i32
    %222 = llvm.mlir.constant(0 : i32) : i32
    %223 = llvm.mlir.constant(0 : i32) : i32
    %224 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%220, %221, %223, %222) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %225 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.mlir.constant(512 : index) : i64
    %227 = llvm.mul %16, %226 : i64
    %228 = llvm.add %227, %13 : i64
    %229 = llvm.getelementptr %225[%228] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %230 = llvm.mlir.constant(0 : i32) : i32
    %231 = llvm.mlir.constant(0 : i32) : i32
    %232 = llvm.mlir.constant(0 : i32) : i32
    %233 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%229, %230, %232, %231) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %234 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%224, %233, %22, %43) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %235 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %236 = llvm.mlir.constant(512 : index) : i64
    %237 = llvm.mul %16, %236 : i64
    %238 = llvm.add %237, %13 : i64
    %239 = llvm.getelementptr %235[%238] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %240 = llvm.mlir.constant(0 : i32) : i32
    %241 = llvm.mlir.constant(0 : i32) : i32
    %242 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%234, %239, %240, %242, %241, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %243 = arith.addi %14, %5 overflow<nsw> : i32
    cf.br ^bb1(%243 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_splits_scope_1(%arg0: i64, %arg1: memref<48x128xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<8x784xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg5 : memref<8x784xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg4 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg3 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = builtin.unrealized_conversion_cast %arg2 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = builtin.unrealized_conversion_cast %arg1 : memref<48x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(64 : i32) : i32
    %8 = llvm.mlir.constant(24 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.mlir.constant(4.000000e-02 : f32) : f32
    %12 = llvm.mlir.constant(64 : index) : i64
    %13 = llvm.mlir.constant(16 : i32) : i32
    %14 = llvm.mlir.constant(784 : index) : i64
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = "hivm_regbaseintrins.intr.hivm.vbr"(%5) : (f32) -> vector<64xf32>
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%17, %18) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %20 = builtin.unrealized_conversion_cast %19 : vector<256xi1> to vector<64xi1>
    %21 = llvm.trunc %arg0 : i64 to i32
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%21, %19, %22) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = "hivm_regbaseintrins.intr.hivm.vci"(%24, %25) : (i32, i32) -> vector<64xi32>
    %27 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%26, %23, %19) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %28 = builtin.unrealized_conversion_cast %27 : vector<256xi1> to vector<64xi1>
    %29 = llvm.mlir.constant(64 : i32) : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = "hivm_regbaseintrins.intr.hivm.vci"(%29, %30) : (i32, i32) -> vector<64xi32>
    %32 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%31, %23, %19) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %33 = builtin.unrealized_conversion_cast %32 : vector<256xi1> to vector<64xi1>
    cf.br ^bb1(%6 : i32)
  ^bb1(%34: i32):  // 2 preds: ^bb0, ^bb2
    %35 = arith.cmpi slt, %34, %8 : i32
    cf.cond_br %35, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %36 = llvm.sext %34 : i32 to i64
    %37 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.mlir.constant(128 : index) : i64
    %39 = llvm.mul %36, %38 : i64
    %40 = llvm.add %39, %10 : i64
    %41 = llvm.getelementptr %37[%40] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %42 = llvm.mlir.constant(0 : i32) : i32
    %43 = llvm.mlir.constant(0 : i32) : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%41, %42, %44, %43) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %46 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%45, %11, %19) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %47 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(128 : index) : i64
    %49 = llvm.mul %36, %48 : i64
    %50 = llvm.add %49, %12 : i64
    %51 = llvm.getelementptr %47[%50] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = llvm.mlir.constant(0 : i32) : i32
    %55 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%51, %52, %54, %53) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %56 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%55, %11, %19) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %57 = "hivm_regbaseintrins.intr.hivm.vsel"(%46, %16, %27) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %58 = "hivm_regbaseintrins.intr.hivm.vsel"(%56, %16, %32) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %59 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.getelementptr %59[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.mlir.constant(0 : i32) : i32
    %63 = llvm.mlir.constant(3 : i32) : i32
    %64 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%60, %61, %63, %62) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %65 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%57, %58, %19) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %66 = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%65, %19) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %67 = llvm.mlir.constant(1 : i32) : i32
    %68 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%66, %19, %67) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %69 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%68, %64, %19) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%64, %69, %19, %70) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %72 = llvm.mlir.constant(0 : i32) : i32
    %73 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%57, %69, %19, %72) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %74 = llvm.mlir.constant(0 : i32) : i32
    %75 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%58, %69, %19, %74) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %76 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%73, %75, %19) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %77 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%76, %19) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %78 = llvm.mlir.constant(1 : i32) : i32
    %79 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%77, %19, %78) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %80 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.getelementptr %80[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %82 = llvm.mlir.constant(0 : i32) : i32
    %83 = llvm.mlir.constant(0 : i32) : i32
    %84 = llvm.mlir.constant(3 : i32) : i32
    %85 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%81, %82, %84, %83) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %86 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%79, %85, %71, %19) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %87 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.getelementptr %87[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = llvm.mlir.constant(0 : i32) : i32
    %91 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%69, %88, %89, %91, %90, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %92 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %93 = llvm.getelementptr %92[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %94 = llvm.mlir.constant(0 : i32) : i32
    %95 = llvm.mlir.constant(0 : i32) : i32
    %96 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%86, %93, %94, %96, %95, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %97 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.getelementptr %97[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.mlir.constant(0 : i32) : i32
    %101 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%71, %98, %99, %101, %100, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.mlir.constant(1 : i32) : i32
    %104 = llvm.mlir.constant(1 : i32) : i32
    %105 = llvm.mlir.constant(0 : i32) : i32
    %106 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%73, %19, %103, %104, %105) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %107 = llvm.mlir.constant(0 : i32) : i32
    %108 = llvm.mlir.constant(1 : i32) : i32
    %109 = llvm.mlir.constant(1 : i32) : i32
    %110 = llvm.mlir.constant(0 : i32) : i32
    %111 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%75, %19, %108, %109, %110) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %112 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%106, %111) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %113 = llvm.extractvalue %112[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %114 = llvm.extractvalue %112[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %115 = llvm.mul %34, %13 : i32
    %116 = llvm.mlir.constant(0 : i32) : i32
    %117 = llvm.mlir.constant(0 : i32) : i32
    %118 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%116, %117) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %119 = builtin.unrealized_conversion_cast %118 : vector<256xi1> to vector<128xi1>
    %120 = llvm.sext %115 : i32 to i64
    %121 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.mlir.constant(784 : index) : i64
    %123 = llvm.mul %10, %122 : i64
    %124 = llvm.add %123, %120 : i64
    %125 = llvm.getelementptr %121[%124] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %126 = llvm.mlir.constant(3211264 : i32) : i32
    %127 = llvm.mlir.constant(0 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"(%113, %125, %126, %127, %118) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>) -> ()
    %128 = arith.addi %34, %9 overflow<nsw> : i32
    cf.br ^bb1(%128 : i32)
  ^bb3:  // pred: ^bb1
    %129 = llvm.add %arg0, %15 : i64
    %130 = llvm.trunc %129 : i64 to i32
    %131 = llvm.mlir.constant(1 : i32) : i32
    %132 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%130, %19, %131) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %133 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%26, %132, %19) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %134 = builtin.unrealized_conversion_cast %133 : vector<256xi1> to vector<64xi1>
    %135 = "hivm_regbaseintrins.intr.hivm.vcmp.lt.s.z"(%31, %132, %19) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %136 = builtin.unrealized_conversion_cast %135 : vector<256xi1> to vector<64xi1>
    cf.br ^bb4(%6 : i32)
  ^bb4(%137: i32):  // 2 preds: ^bb3, ^bb5
    %138 = arith.cmpi slt, %137, %8 : i32
    cf.cond_br %138, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %139 = llvm.add %137, %8 : i32
    %140 = llvm.sext %139 : i32 to i64
    %141 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(128 : index) : i64
    %143 = llvm.mul %140, %142 : i64
    %144 = llvm.add %143, %10 : i64
    %145 = llvm.getelementptr %141[%144] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %146 = llvm.mlir.constant(0 : i32) : i32
    %147 = llvm.mlir.constant(0 : i32) : i32
    %148 = llvm.mlir.constant(0 : i32) : i32
    %149 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%145, %146, %148, %147) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %150 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%149, %11, %19) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %151 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.mlir.constant(128 : index) : i64
    %153 = llvm.mul %140, %152 : i64
    %154 = llvm.add %153, %12 : i64
    %155 = llvm.getelementptr %151[%154] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %156 = llvm.mlir.constant(0 : i32) : i32
    %157 = llvm.mlir.constant(0 : i32) : i32
    %158 = llvm.mlir.constant(0 : i32) : i32
    %159 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%155, %156, %158, %157) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %160 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%159, %11, %19) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %161 = "hivm_regbaseintrins.intr.hivm.vsel"(%150, %16, %133) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %162 = "hivm_regbaseintrins.intr.hivm.vsel"(%160, %16, %135) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %163 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %164 = llvm.getelementptr %163[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %165 = llvm.mlir.constant(0 : i32) : i32
    %166 = llvm.mlir.constant(0 : i32) : i32
    %167 = llvm.mlir.constant(3 : i32) : i32
    %168 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%164, %165, %167, %166) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %169 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%161, %162, %19) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %170 = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%169, %19) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %171 = llvm.mlir.constant(1 : i32) : i32
    %172 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%170, %19, %171) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %173 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%172, %168, %19) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %174 = llvm.mlir.constant(0 : i32) : i32
    %175 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%168, %173, %19, %174) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %176 = llvm.mlir.constant(0 : i32) : i32
    %177 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%161, %173, %19, %176) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %178 = llvm.mlir.constant(0 : i32) : i32
    %179 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%162, %173, %19, %178) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %180 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%177, %179, %19) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %181 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%180, %19) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %182 = llvm.mlir.constant(1 : i32) : i32
    %183 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%181, %19, %182) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %184 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %185 = llvm.getelementptr %184[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %186 = llvm.mlir.constant(0 : i32) : i32
    %187 = llvm.mlir.constant(0 : i32) : i32
    %188 = llvm.mlir.constant(3 : i32) : i32
    %189 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%185, %186, %188, %187) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %190 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%183, %189, %175, %19) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %191 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %192 = llvm.getelementptr %191[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %193 = llvm.mlir.constant(0 : i32) : i32
    %194 = llvm.mlir.constant(0 : i32) : i32
    %195 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%173, %192, %193, %195, %194, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %196 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %197 = llvm.getelementptr %196[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %198 = llvm.mlir.constant(0 : i32) : i32
    %199 = llvm.mlir.constant(0 : i32) : i32
    %200 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%190, %197, %198, %200, %199, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %201 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %202 = llvm.getelementptr %201[%140] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %203 = llvm.mlir.constant(0 : i32) : i32
    %204 = llvm.mlir.constant(0 : i32) : i32
    %205 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%175, %202, %203, %205, %204, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %206 = llvm.mlir.constant(0 : i32) : i32
    %207 = llvm.mlir.constant(1 : i32) : i32
    %208 = llvm.mlir.constant(1 : i32) : i32
    %209 = llvm.mlir.constant(0 : i32) : i32
    %210 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%177, %19, %207, %208, %209) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %211 = llvm.mlir.constant(0 : i32) : i32
    %212 = llvm.mlir.constant(1 : i32) : i32
    %213 = llvm.mlir.constant(1 : i32) : i32
    %214 = llvm.mlir.constant(0 : i32) : i32
    %215 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%179, %19, %212, %213, %214) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %216 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%210, %215) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %217 = llvm.extractvalue %216[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %218 = llvm.extractvalue %216[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %219 = llvm.mul %139, %13 : i32
    %220 = llvm.mlir.constant(0 : i32) : i32
    %221 = llvm.mlir.constant(0 : i32) : i32
    %222 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%220, %221) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %223 = builtin.unrealized_conversion_cast %222 : vector<256xi1> to vector<128xi1>
    %224 = llvm.sext %219 : i32 to i64
    %225 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.mlir.constant(784 : index) : i64
    %227 = llvm.mul %10, %226 : i64
    %228 = llvm.add %227, %224 : i64
    %229 = llvm.getelementptr %225[%228] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %230 = llvm.mlir.constant(3211264 : i32) : i32
    %231 = llvm.mlir.constant(0 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"(%217, %229, %230, %231, %222) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>) -> ()
    %232 = arith.addi %137, %9 overflow<nsw> : i32
    cf.br ^bb4(%232 : i32)
  ^bb6:  // pred: ^bb4
    return
  }
  func.func @attention_splits_scope_2(%arg0: memref<48x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<8x784xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg4 : memref<8x784xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg3 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = builtin.unrealized_conversion_cast %arg1 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = builtin.unrealized_conversion_cast %arg0 : memref<48x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.constant(4.000000e-02 : f32) : f32
    %10 = llvm.mlir.constant(64 : index) : i64
    %11 = llvm.mlir.constant(16 : i32) : i32
    %12 = llvm.mlir.constant(784 : index) : i64
    cf.br ^bb1(%5 : i32)
  ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb2
    %14 = arith.cmpi slt, %13, %6 : i32
    cf.cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %15 = llvm.sext %13 : i32 to i64
    %16 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.mlir.constant(128 : index) : i64
    %18 = llvm.mul %15, %17 : i64
    %19 = llvm.add %18, %8 : i64
    %20 = llvm.getelementptr %16[%19] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%20, %21, %23, %22) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%25, %26) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %28 = builtin.unrealized_conversion_cast %27 : vector<256xi1> to vector<64xi1>
    %29 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%24, %9, %27) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %30 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(128 : index) : i64
    %32 = llvm.mul %15, %31 : i64
    %33 = llvm.add %32, %10 : i64
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %35 = llvm.mlir.constant(0 : i32) : i32
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%34, %35, %37, %36) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %39 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%38, %9, %27) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %40 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %42 = llvm.mlir.constant(0 : i32) : i32
    %43 = llvm.mlir.constant(0 : i32) : i32
    %44 = llvm.mlir.constant(3 : i32) : i32
    %45 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%41, %42, %44, %43) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %46 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%29, %39, %27) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %47 = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%46, %27) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %48 = llvm.mlir.constant(1 : i32) : i32
    %49 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%47, %27, %48) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %50 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%49, %45, %27) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %51 = llvm.mlir.constant(0 : i32) : i32
    %52 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%45, %50, %27, %51) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%29, %50, %27, %53) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%39, %50, %27, %55) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %57 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%54, %56, %27) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %58 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%57, %27) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %59 = llvm.mlir.constant(1 : i32) : i32
    %60 = "hivm_regbaseintrins.intr.hivm.vdup.z"(%58, %27, %59) : (vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %61 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = llvm.getelementptr %61[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.constant(0 : i32) : i32
    %65 = llvm.mlir.constant(3 : i32) : i32
    %66 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%62, %63, %65, %64) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %67 = "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%60, %66, %52, %27) : (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %68 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.getelementptr %68[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%50, %69, %70, %72, %71, %27) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %73 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.getelementptr %73[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %75 = llvm.mlir.constant(0 : i32) : i32
    %76 = llvm.mlir.constant(0 : i32) : i32
    %77 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%67, %74, %75, %77, %76, %27) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %78 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.getelementptr %78[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %80 = llvm.mlir.constant(0 : i32) : i32
    %81 = llvm.mlir.constant(0 : i32) : i32
    %82 = llvm.mlir.constant(5 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%52, %79, %80, %82, %81, %27) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %83 = llvm.mlir.constant(0 : i32) : i32
    %84 = llvm.mlir.constant(1 : i32) : i32
    %85 = llvm.mlir.constant(1 : i32) : i32
    %86 = llvm.mlir.constant(0 : i32) : i32
    %87 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%54, %27, %84, %85, %86) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(1 : i32) : i32
    %90 = llvm.mlir.constant(1 : i32) : i32
    %91 = llvm.mlir.constant(0 : i32) : i32
    %92 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%56, %27, %89, %90, %91) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %93 = "hivm_regbaseintrins.intr.hivm.vdintlv"(%87, %92) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
    %94 = llvm.extractvalue %93[0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %95 = llvm.extractvalue %93[1] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)> 
    %96 = llvm.mul %13, %11 : i32
    %97 = llvm.mlir.constant(0 : i32) : i32
    %98 = llvm.mlir.constant(0 : i32) : i32
    %99 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%97, %98) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %100 = builtin.unrealized_conversion_cast %99 : vector<256xi1> to vector<128xi1>
    %101 = llvm.sext %96 : i32 to i64
    %102 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.mlir.constant(784 : index) : i64
    %104 = llvm.mul %8, %103 : i64
    %105 = llvm.add %104, %101 : i64
    %106 = llvm.getelementptr %102[%105] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %107 = llvm.mlir.constant(3211264 : i32) : i32
    %108 = llvm.mlir.constant(0 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"(%94, %106, %107, %108, %99) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>) -> ()
    %109 = arith.addi %13, %7 overflow<nsw> : i32
    cf.br ^bb1(%109 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_splits_scope_3(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<48x512xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, vector_mode = "vec"} {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<48x512xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(48 : i32) : i32
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.constant(64 : index) : i64
    %10 = llvm.mlir.constant(128 : index) : i64
    %11 = llvm.mlir.constant(192 : index) : i64
    %12 = llvm.mlir.constant(256 : index) : i64
    %13 = llvm.mlir.constant(320 : index) : i64
    %14 = llvm.mlir.constant(384 : index) : i64
    %15 = llvm.mlir.constant(448 : index) : i64
    %16 = "hivm_regbaseintrins.intr.hivm.vbr"(%3) : (f32) -> vector<64xf32>
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%17, %18) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %20 = builtin.unrealized_conversion_cast %19 : vector<256xi1> to vector<64xi1>
    %21 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%16, %21, %22, %24, %23, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %25 = "hivm_regbaseintrins.intr.hivm.vbr"(%5) : (f32) -> vector<64xf32>
    %26 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %26, %27, %29, %28, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    cf.br ^bb1(%6 : i32)
  ^bb1(%30: i32):  // 2 preds: ^bb0, ^bb2
    %31 = arith.cmpi slt, %30, %7 : i32
    cf.cond_br %31, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %32 = llvm.sext %30 : i32 to i64
    %33 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(512 : index) : i64
    %35 = llvm.mul %32, %34 : i64
    %36 = llvm.add %35, %4 : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %37, %38, %40, %39, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %41 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(512 : index) : i64
    %43 = llvm.mul %32, %42 : i64
    %44 = llvm.add %43, %9 : i64
    %45 = llvm.getelementptr %41[%44] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %46 = llvm.mlir.constant(0 : i32) : i32
    %47 = llvm.mlir.constant(0 : i32) : i32
    %48 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %45, %46, %48, %47, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %49 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mlir.constant(512 : index) : i64
    %51 = llvm.mul %32, %50 : i64
    %52 = llvm.add %51, %10 : i64
    %53 = llvm.getelementptr %49[%52] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %54 = llvm.mlir.constant(0 : i32) : i32
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %53, %54, %56, %55, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %57 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mlir.constant(512 : index) : i64
    %59 = llvm.mul %32, %58 : i64
    %60 = llvm.add %59, %11 : i64
    %61 = llvm.getelementptr %57[%60] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %62 = llvm.mlir.constant(0 : i32) : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %61, %62, %64, %63, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %65 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(512 : index) : i64
    %67 = llvm.mul %32, %66 : i64
    %68 = llvm.add %67, %12 : i64
    %69 = llvm.getelementptr %65[%68] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %69, %70, %72, %71, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %73 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mlir.constant(512 : index) : i64
    %75 = llvm.mul %32, %74 : i64
    %76 = llvm.add %75, %13 : i64
    %77 = llvm.getelementptr %73[%76] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(0 : i32) : i32
    %80 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %77, %78, %80, %79, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %81 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(512 : index) : i64
    %83 = llvm.mul %32, %82 : i64
    %84 = llvm.add %83, %14 : i64
    %85 = llvm.getelementptr %81[%84] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %86 = llvm.mlir.constant(0 : i32) : i32
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %85, %86, %88, %87, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %89 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.mlir.constant(512 : index) : i64
    %91 = llvm.mul %32, %90 : i64
    %92 = llvm.add %91, %15 : i64
    %93 = llvm.getelementptr %89[%92] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %94 = llvm.mlir.constant(0 : i32) : i32
    %95 = llvm.mlir.constant(0 : i32) : i32
    %96 = llvm.mlir.constant(2 : i32) : i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%25, %93, %94, %96, %95, %19) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %97 = arith.addi %30, %8 overflow<nsw> : i32
    cf.br ^bb1(%97 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @attention_splits_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: memref<?xi32, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(1 : i8) : i8
    %1 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(0 : i8) : i8
    %4 = llvm.mlir.constant(-1 : i64) : i64
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(258048 : i64) : i64
    %8 = llvm.mlir.constant(430080 : i64) : i64
    %9 = llvm.mlir.constant(110592 : i64) : i64
    %10 = llvm.mlir.constant(282624 : i64) : i64
    %11 = llvm.mlir.constant(123648 : i64) : i64
    %12 = llvm.mlir.constant(99072 : i64) : i64
    %13 = llvm.mlir.constant(49152 : i64) : i64
    %14 = llvm.mlir.constant(2 : index) : i64
    %15 = llvm.mlir.constant(5 : i32) : i32
    %16 = llvm.mlir.constant(2 : i32) : i32
    %17 = llvm.mlir.constant(9 : i32) : i32
    %18 = llvm.mlir.constant(128 : i32) : i32
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = builtin.unrealized_conversion_cast %19 : i64 to index
    %21 = llvm.mlir.constant(true) : i1
    %22 = llvm.mlir.constant(96 : index) : i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = llvm.mlir.constant(576 : index) : i64
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %26 = llvm.mlir.constant(128 : index) : i64
    %27 = builtin.unrealized_conversion_cast %26 : i64 to index
    %28 = llvm.mlir.constant(2 : i64) : i64
    %29 = llvm.mlir.constant(512 : index) : i64
    %30 = builtin.unrealized_conversion_cast %29 : i64 to index
    %31 = llvm.mlir.constant(20 : i32) : i32
    %32 = llvm.mlir.constant(55296 : i32) : i32
    %33 = llvm.mlir.constant(-1 : i32) : i32
    %34 = llvm.mlir.constant(73728 : i64) : i64
    %35 = llvm.mlir.constant(1 : i64) : i64
    %36 = llvm.mlir.constant(0 : index) : i64
    %37 = builtin.unrealized_conversion_cast %36 : i64 to index
    %38 = llvm.mlir.constant(0 : i64) : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %38, %alloca[%37] : memref<1xi64>
    %39 = hivm.hir.pointer_cast(%8) : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %41 = hivm.hir.pointer_cast(%7) : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>>
    %42 = builtin.unrealized_conversion_cast %41 : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %43 = hivm.hir.pointer_cast(%9) : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>>
    %44 = builtin.unrealized_conversion_cast %43 : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %45 = hivm.hir.pointer_cast(%10) : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>>
    %46 = builtin.unrealized_conversion_cast %45 : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %47 = hivm.hir.pointer_cast(%10) : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>>
    %48 = builtin.unrealized_conversion_cast %47 : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %49 = hivm.hir.pointer_cast(%9) : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>>
    %50 = builtin.unrealized_conversion_cast %49 : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %51 = hivm.hir.get_block_idx -> i64
    %52 = llvm.trunc %51 : i64 to i32
    %53 = llvm.srem %52, %arg9  : i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    cf.br ^bb1(%53 : i32)
  ^bb1(%54: i32):  // 2 preds: ^bb0, ^bb11
    %55 = arith.cmpi slt, %54, %31 : i32
    cf.cond_br %55, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %56 = llvm.sdiv %54, %15  : i32
    %57 = llvm.srem %54, %15  : i32
    %58 = llvm.mul %57, %16 : i32
    %59 = llvm.sub %17, %58 : i32
    %60 = llvm.intr.smin(%59, %16)  : (i32, i32) -> i32
    %61 = hivm.hir.pointer_cast(%38) : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>>
    %62 = llvm.mul %56, %32 : i32
    %63 = llvm.sext %62 : i32 to i64
    %64 = builtin.unrealized_conversion_cast %63 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%64], sizes: [96, 576], strides: [576, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<96x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE2>]
    %cast = memref.cast %reinterpret_cast : memref<96x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_0 = memref.cast %61 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast, %cast_0, %6) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    %65 = llvm.sext %60 : i32 to i64
    %66 = llvm.add %65, %14 : i64
    %67 = llvm.icmp "sgt" %65, %36 : i64
    %68 = llvm.select %67, %66, %36 : i1, i64
    %69 = builtin.unrealized_conversion_cast %68 : i64 to index
    cf.br ^bb3(%37 : index)
  ^bb3(%70: index):  // 2 preds: ^bb2, ^bb10
    %71 = arith.cmpi slt, %70, %69 : index
    cf.cond_br %71, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %72 = builtin.unrealized_conversion_cast %70 : index to i64
    %73 = llvm.sub %72, %19 : i64
    %74 = memref.load %alloca[%37] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %75 = llvm.urem %74, %28  : i64
    %76 = llvm.icmp "eq" %75, %35 : i64
    %77 = llvm.select %76, %46, %44 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %78 = builtin.unrealized_conversion_cast %77 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<36x8x16x16xbf16, #hivm.address_space<cbuf>>
    %79 = llvm.add %75, %19 : i64
    %80 = llvm.srem %79, %14  : i64
    %81 = llvm.add %80, %19 : i64
    %82 = llvm.sext %58 : i32 to i64
    %83 = llvm.add %82, %72 : i64
    %84 = hivm.hir.pointer_cast(%38) : memref<8x6x16x16xf32, #hivm.address_space<cc>>
    %85 = hivm.hir.pointer_cast(%13) : memref<32x6x16x16xf32, #hivm.address_space<cc>>
    %86 = hivm.hir.pointer_cast(%12) : memref<48x128xf32, #hivm.address_space<ub>>
    %87 = hivm.hir.pointer_cast(%11) : memref<48x512xf32, #hivm.address_space<ub>>
    %88 = llvm.icmp "sge" %72, %36 : i64
    %89 = llvm.icmp "slt" %72, %65 : i64
    %90 = llvm.and %88, %89  : i1
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    cf.cond_br %90, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %91 = llvm.sext %56 : i32 to i64
    %92 = builtin.unrealized_conversion_cast %91 : i64 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%92], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %93 = memref.load %reinterpret_cast_1[%37] {hivm.tcore_type = #hivm.tcore_type<CUBE>, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %94 = llvm.add %93, %33 : i32
    %95 = llvm.sdiv %94, %18  : i32
    %96 = llvm.mul %56, %17 : i32
    %97 = llvm.sext %95 : i32 to i64
    %98 = llvm.intr.smin(%83, %97)  : (i64, i64) -> i64
    %99 = llvm.sext %96 : i32 to i64
    %100 = llvm.add %99, %98 : i64
    %101 = builtin.unrealized_conversion_cast %100 : i64 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%101], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %102 = memref.load %reinterpret_cast_2[%37] {hivm.tcore_type = #hivm.tcore_type<CUBE>, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %103 = llvm.sext %102 : i32 to i64
    %104 = llvm.mul %103, %34 : i64
    %105 = builtin.unrealized_conversion_cast %104 : i64 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%105], sizes: [128, 576], strides: [576, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_4 = memref.cast %reinterpret_cast_3 : memref<128x576xbf16, strided<[576, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_5 = memref.cast %78 : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast_4, %cast_5, %5) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    cf.cond_br %90, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %cast_6 = memref.cast %61 : memref<36x6x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_7 = memref.cast %78 : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_8 = memref.cast %84 : memref<8x6x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_bfloat16_t_to_float_tb(%cast_6, %cast_7, %21, %23, %25, %27, %cast_8, %4, %4, %4, %4, %4, %4, %4, %3, %38) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %cast_9 = memref.cast %84 : memref<8x6x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_10 = memref.cast %86 : memref<48x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_9, %cast_10, %38, %1, %38, %2, %21, %3, %38, %0) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %106 = llvm.icmp "sge" %73, %36 : i64
    %107 = llvm.icmp "slt" %73, %65 : i64
    %108 = llvm.and %106, %107  : i1
    cf.cond_br %108, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %109 = llvm.select %76, %42, %40 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %110 = builtin.unrealized_conversion_cast %109 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x6x16x16xbf16, #hivm.address_space<cbuf>>
    %111 = llvm.select %76, %50, %48 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %112 = builtin.unrealized_conversion_cast %111 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<36x8x16x16xbf16, #hivm.address_space<cbuf>>
    %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %112 : memref<36x8x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 8, 16, 16], strides: [2048, 256, 16, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1]>, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %cast_12 = memref.cast %110 : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_13 = memref.cast %reinterpret_cast_11 : memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_14 = memref.cast %85 : memref<32x6x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_bfloat16_t_to_float(%cast_12, %cast_13, %21, %23, %27, %30, %cast_14, %4, %4, %4, %4, %4, %4, %4, %3, %38) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %81
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %cast_15 = memref.cast %85 : memref<32x6x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_16 = memref.cast %87 : memref<48x512xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_15, %cast_16, %38, %1, %38, %2, %21, %3, %38, %0) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    cf.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    %113 = llvm.add %74, %35 : i64
    memref.store %113, %alloca[%37] : memref<1xi64>
    %114 = arith.addi %70, %20 overflow<nsw> : index
    cf.br ^bb3(%114 : index)
  ^bb11:  // pred: ^bb3
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    %115 = arith.addi %54, %31 overflow<nsw> : i32
    cf.br ^bb1(%115 : i32)
  ^bb12:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func private @nd2nz_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_bfloat16_t_to_float_tb(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_bfloat16_t_to_float(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_cbuf_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @attention_splits_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: memref<?xi32, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(234496 : i64) : i64
    %1 = llvm.mlir.constant(98816 : i64) : i64
    %2 = llvm.mlir.constant(258048 : i64) : i64
    %3 = llvm.mlir.constant(430080 : i64) : i64
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(123648 : i64) : i64
    %6 = llvm.mlir.constant(221952 : i64) : i64
    %7 = llvm.mlir.constant(99072 : i64) : i64
    %8 = llvm.mlir.constant(512 : i64) : i64
    %9 = llvm.mlir.constant(256 : i64) : i64
    %10 = llvm.mlir.constant(5 : i32) : i32
    %11 = llvm.mlir.constant(2 : i32) : i32
    %12 = llvm.mlir.constant(9 : i32) : i32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(384 : i32) : i32
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %17 = llvm.mlir.constant(128 : i64) : i64
    %18 = llvm.mlir.constant(2 : i64) : i64
    %19 = llvm.mlir.constant(48 : i64) : i64
    %20 = llvm.mlir.constant(20 : i32) : i32
    %21 = llvm.mlir.constant(96 : i32) : i32
    %22 = llvm.mlir.constant(256 : i32) : i32
    %23 = llvm.mlir.constant(-3 : i32) : i32
    %24 = llvm.mlir.constant(768 : i64) : i64
    %25 = llvm.mlir.constant(1 : i32) : i32
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = builtin.unrealized_conversion_cast %27 : i64 to index
    %29 = llvm.mlir.constant(0 : i64) : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %29, %alloca[%28] : memref<1xi64>
    %30 = hivm.hir.pointer_cast(%1) : memref<64xf32, #hivm.address_space<ub>>
    %31 = builtin.unrealized_conversion_cast %30 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = hivm.hir.pointer_cast(%0) : memref<64xf32, #hivm.address_space<ub>>
    %33 = builtin.unrealized_conversion_cast %32 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = hivm.hir.pointer_cast(%0) : memref<64xf32, #hivm.address_space<ub>>
    %35 = builtin.unrealized_conversion_cast %34 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = hivm.hir.pointer_cast(%1) : memref<64xf32, #hivm.address_space<ub>>
    %37 = builtin.unrealized_conversion_cast %36 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = hivm.hir.pointer_cast(%2) : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>>
    %39 = builtin.unrealized_conversion_cast %38 : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %40 = hivm.hir.pointer_cast(%3) : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %42 = hivm.hir.get_block_idx -> i64
    %43 = llvm.trunc %42 : i64 to i32
    %44 = llvm.srem %43, %arg9  : i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    cf.br ^bb1(%44 : i32)
  ^bb1(%45: i32):  // 2 preds: ^bb0, ^bb15
    %46 = arith.cmpi slt, %45, %20 : i32
    cf.cond_br %46, ^bb2, ^bb16
  ^bb2:  // pred: ^bb1
    %47 = llvm.sdiv %45, %10  : i32
    %48 = llvm.srem %45, %10  : i32
    %49 = llvm.mul %48, %11 : i32
    %50 = llvm.sub %12, %49 : i32
    %51 = llvm.intr.smin(%50, %11)  : (i32, i32) -> i32
    %52 = llvm.mul %47, %21 : i32
    %53 = hivm.hir.get_sub_block_idx -> i64
    %54 = llvm.mul %48, %22 : i32
    %55 = hivm.hir.pointer_cast(%29) : memref<64xf32, #hivm.address_space<ub>>
    %56 = hivm.hir.pointer_cast(%9) : memref<64xf32, #hivm.address_space<ub>>
    %57 = hivm.hir.pointer_cast(%8) : memref<48x512xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    call @attention_splits_scope_3(%55, %56, %57) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<48x512xf32, #hivm.address_space<ub>>) -> ()
    %58 = llvm.sext %51 : i32 to i64
    %59 = llvm.add %58, %4 : i64
    %60 = llvm.icmp "sgt" %58, %27 : i64
    %61 = llvm.select %60, %59, %27 : i1, i64
    %62 = builtin.unrealized_conversion_cast %61 : i64 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    cf.br ^bb3(%28 : index)
  ^bb3(%63: index):  // 2 preds: ^bb2, ^bb14
    %64 = arith.cmpi slt, %63, %62 : index
    cf.cond_br %64, ^bb4, ^bb15
  ^bb4:  // pred: ^bb3
    %65 = builtin.unrealized_conversion_cast %63 : index to i64
    %66 = llvm.sub %65, %15 : i64
    %67 = memref.load %alloca[%28] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %68 = llvm.urem %67, %18  : i64
    %69 = llvm.icmp "eq" %68, %26 : i64
    %70 = llvm.select %69, %33, %31 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %71 = builtin.unrealized_conversion_cast %70 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<64xf32, #hivm.address_space<ub>>
    %72 = llvm.add %68, %15 : i64
    %73 = llvm.mul %65, %17 : i64
    %74 = llvm.sext %54 : i32 to i64
    %75 = llvm.add %74, %73 : i64
    %76 = hivm.hir.pointer_cast(%7) : memref<48x128xf32, #hivm.address_space<ub>>
    %77 = hivm.hir.pointer_cast(%6) : memref<8x784xbf16, #hivm.address_space<ub>>
    %78 = hivm.hir.pointer_cast(%5) : memref<48x512xf32, #hivm.address_space<ub>>
    %79 = llvm.icmp "sge" %65, %27 : i64
    %80 = llvm.icmp "slt" %65, %58 : i64
    %81 = llvm.and %79, %80  : i1
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    cf.cond_br %81, ^bb5, ^bb10
  ^bb5:  // pred: ^bb4
    %82 = llvm.sext %47 : i32 to i64
    %83 = builtin.unrealized_conversion_cast %82 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%83], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %84 = memref.load %reinterpret_cast[%28] {hivm.tcore_type = #hivm.tcore_type<VECTOR>, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %85 = llvm.add %84, %23 : i32
    %86 = llvm.sext %85 : i32 to i64
    %87 = llvm.sub %86, %75 : i64
    %88 = llvm.trunc %87 : i64 to i32
    %89 = llvm.mul %53, %18 : i64
    %90 = llvm.sext %88 : i32 to i64
    %91 = llvm.add %90, %89 : i64
    %92 = llvm.icmp "slt" %91, %17 : i64
    %93 = llvm.zext %92 : i1 to i32
    %94 = llvm.icmp "eq" %93, %13 : i32
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
    cf.cond_br %94, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    call @attention_splits_scope_2(%76, %55, %56, %71, %77) {hivm.vector_function} : (memref<48x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<8x784xbf16, #hivm.address_space<ub>>) -> ()
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %95 = llvm.icmp "eq" %93, %25 : i32
    cf.cond_br %95, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    call @attention_splits_scope_1(%91, %76, %55, %56, %71, %77) {hivm.vector_function} : (i64, memref<48x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<8x784xbf16, #hivm.address_space<ub>>) -> ()
    cf.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
    cf.br ^bb10
  ^bb10:  // 2 preds: ^bb4, ^bb9
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    cf.cond_br %81, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %96 = llvm.select %69, %41, %39 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %97 = builtin.unrealized_conversion_cast %96 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x6x16x16xbf16, #hivm.address_space<cbuf>>
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %77 : memref<8x784xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_0 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8, 768], strides: [784, 1] : memref<bf16, #hivm.address_space<ub>> to memref<8x768xbf16, strided<[784, 1]>, #hivm.address_space<ub>>
    %98 = llvm.mul %53, %24 : i64
    %99 = builtin.unrealized_conversion_cast %98 : i64 to index
    %base_buffer_1, %offset_2, %sizes_3:4, %strides_4:4 = memref.extract_strided_metadata %97 : memref<8x6x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%99], sizes: [8, 768], strides: [1536, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<8x768xbf16, strided<[1536, 1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %72
    %cast = memref.cast %reinterpret_cast_0 : memref<8x768xbf16, strided<[784, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_6 = memref.cast %reinterpret_cast_5 : memref<8x768xbf16, strided<[1536, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast, %cast_6) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    cf.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %100 = llvm.icmp "sge" %66, %27 : i64
    %101 = llvm.icmp "slt" %66, %58 : i64
    %102 = llvm.and %100, %101  : i1
    cf.cond_br %102, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %103 = llvm.select %69, %37, %35 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %104 = builtin.unrealized_conversion_cast %103 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    call @attention_splits_scope_0(%104, %78, %57) {hivm.vector_function} : (memref<64xf32, #hivm.address_space<ub>>, memref<48x512xf32, #hivm.address_space<ub>>, memref<48x512xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    cf.br ^bb14
  ^bb14:  // 2 preds: ^bb12, ^bb13
    %105 = llvm.add %67, %26 : i64
    memref.store %105, %alloca[%28] : memref<1xi64>
    %106 = arith.addi %63, %16 overflow<nsw> : index
    cf.br ^bb3(%106 : index)
  ^bb15:  // pred: ^bb3
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %107 = llvm.mul %48, %14 : i32
    %108 = llvm.add %107, %52 : i32
    %109 = llvm.mul %53, %19 : i64
    %110 = llvm.sext %108 : i32 to i64
    %111 = llvm.add %110, %109 : i64
    %112 = builtin.unrealized_conversion_cast %111 : i64 to index
    %113 = llvm.mul %111, %8 : i64
    %114 = builtin.unrealized_conversion_cast %113 : i64 to index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%114], sizes: [48, 512], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_8 = memref.cast %57 : memref<48x512xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %reinterpret_cast_7 : memref<48x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_float(%cast_8, %cast_9, %13) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %55 : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer_10 to offset: [0], sizes: [48], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg7 to offset: [%112], sizes: [48], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<48xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %cast_16 = memref.cast %reinterpret_cast_14 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_17 = memref.cast %reinterpret_cast_15 : memref<48xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_16, %cast_17, %13) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %56 : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [0], sizes: [48], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    %reinterpret_cast_23 = memref.reinterpret_cast %arg8 to offset: [%112], sizes: [48], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<48xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %cast_24 = memref.cast %reinterpret_cast_22 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_25 = memref.cast %reinterpret_cast_23 : memref<48xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_24, %cast_25, %13) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %115 = arith.addi %45, %20 overflow<nsw> : i32
    cf.br ^bb1(%115 : i32)
  ^bb16:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
