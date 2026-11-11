; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) noalias %5, ptr addrspace(6) noalias %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr addrspace(6) noalias %12, ptr addrspace(6) noalias %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18) #0 {
  br label %20

20:                                               ; preds = %23, %19
  %21 = phi i32 [ %110, %23 ], [ 0, %19 ]
  %22 = icmp slt i32 %21, 16
  br i1 %22, label %23, label %111

23:                                               ; preds = %20
  %24 = sext i32 %21 to i64
  %25 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %26 = getelementptr float, ptr addrspace(6) %25, i64 %24
  %27 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %26, i32 0, i32 3, i32 0)
  %28 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %29 = mul i64 %24, 512
  %30 = add i64 %29, 0
  %31 = getelementptr float, ptr addrspace(6) %28, i64 %30
  %32 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %31, i32 0, i32 0, i32 0)
  %33 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %34 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %32, <64 x float> %27, <256 x i1> %33)
  %35 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %36 = mul i64 %24, 512
  %37 = add i64 %36, 64
  %38 = getelementptr float, ptr addrspace(6) %35, i64 %37
  %39 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %38, i32 0, i32 0, i32 0)
  %40 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %39, <64 x float> %27, <256 x i1> %33)
  %41 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %34, <256 x i1> %33, i32 1, i32 1, i32 0)
  %42 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %40, <256 x i1> %33, i32 1, i32 1, i32 0)
  %43 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %41, <128 x bfloat> %42)
  %44 = extractvalue { <128 x bfloat>, <128 x bfloat> } %43, 0
  %45 = call <256 x i1> @llvm.hivm.pge.b16(i32 0, i32 0)
  %46 = getelementptr bfloat, ptr addrspace(6) %13, i64 %14
  %47 = mul i64 %24, 512
  %48 = add i64 %47, 0
  %49 = getelementptr bfloat, ptr addrspace(6) %46, i64 %48
  call void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat> %44, ptr addrspace(6) %49, i32 0, i32 1, i32 0, <256 x i1> %45)
  %50 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %51 = mul i64 %24, 512
  %52 = add i64 %51, 128
  %53 = getelementptr float, ptr addrspace(6) %50, i64 %52
  %54 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %53, i32 0, i32 0, i32 0)
  %55 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %54, <64 x float> %27, <256 x i1> %33)
  %56 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %57 = mul i64 %24, 512
  %58 = add i64 %57, 192
  %59 = getelementptr float, ptr addrspace(6) %56, i64 %58
  %60 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %59, i32 0, i32 0, i32 0)
  %61 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %60, <64 x float> %27, <256 x i1> %33)
  %62 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %55, <256 x i1> %33, i32 1, i32 1, i32 0)
  %63 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %61, <256 x i1> %33, i32 1, i32 1, i32 0)
  %64 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %62, <128 x bfloat> %63)
  %65 = extractvalue { <128 x bfloat>, <128 x bfloat> } %64, 0
  %66 = getelementptr bfloat, ptr addrspace(6) %13, i64 %14
  %67 = mul i64 %24, 512
  %68 = add i64 %67, 128
  %69 = getelementptr bfloat, ptr addrspace(6) %66, i64 %68
  call void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat> %65, ptr addrspace(6) %69, i32 0, i32 1, i32 0, <256 x i1> %45)
  %70 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %71 = mul i64 %24, 512
  %72 = add i64 %71, 256
  %73 = getelementptr float, ptr addrspace(6) %70, i64 %72
  %74 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %73, i32 0, i32 0, i32 0)
  %75 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %74, <64 x float> %27, <256 x i1> %33)
  %76 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %77 = mul i64 %24, 512
  %78 = add i64 %77, 320
  %79 = getelementptr float, ptr addrspace(6) %76, i64 %78
  %80 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %79, i32 0, i32 0, i32 0)
  %81 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %80, <64 x float> %27, <256 x i1> %33)
  %82 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %75, <256 x i1> %33, i32 1, i32 1, i32 0)
  %83 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %81, <256 x i1> %33, i32 1, i32 1, i32 0)
  %84 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %82, <128 x bfloat> %83)
  %85 = extractvalue { <128 x bfloat>, <128 x bfloat> } %84, 0
  %86 = getelementptr bfloat, ptr addrspace(6) %13, i64 %14
  %87 = mul i64 %24, 512
  %88 = add i64 %87, 256
  %89 = getelementptr bfloat, ptr addrspace(6) %86, i64 %88
  call void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat> %85, ptr addrspace(6) %89, i32 0, i32 1, i32 0, <256 x i1> %45)
  %90 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %91 = mul i64 %24, 512
  %92 = add i64 %91, 384
  %93 = getelementptr float, ptr addrspace(6) %90, i64 %92
  %94 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %93, i32 0, i32 0, i32 0)
  %95 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %94, <64 x float> %27, <256 x i1> %33)
  %96 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %97 = mul i64 %24, 512
  %98 = add i64 %97, 448
  %99 = getelementptr float, ptr addrspace(6) %96, i64 %98
  %100 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %99, i32 0, i32 0, i32 0)
  %101 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %100, <64 x float> %27, <256 x i1> %33)
  %102 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %95, <256 x i1> %33, i32 1, i32 1, i32 0)
  %103 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %101, <256 x i1> %33, i32 1, i32 1, i32 0)
  %104 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %102, <128 x bfloat> %103)
  %105 = extractvalue { <128 x bfloat>, <128 x bfloat> } %104, 0
  %106 = getelementptr bfloat, ptr addrspace(6) %13, i64 %14
  %107 = mul i64 %24, 512
  %108 = add i64 %107, 384
  %109 = getelementptr bfloat, ptr addrspace(6) %106, i64 %108
  call void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat> %105, ptr addrspace(6) %109, i32 0, i32 1, i32 0, <256 x i1> %45)
  %110 = add nsw i32 %21, 1
  br label %20

111:                                              ; preds = %20
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_1(i32 %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i32 %8, ptr addrspace(6) noalias %9, ptr addrspace(6) noalias %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr addrspace(6) noalias %18, ptr addrspace(6) noalias %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24) #0 {
  br label %26

26:                                               ; preds = %29, %25
  %27 = phi i32 [ %175, %29 ], [ 0, %25 ]
  %28 = icmp slt i32 %27, 16
  br i1 %28, label %29, label %176

29:                                               ; preds = %26
  %30 = sext i32 %0 to i64
  %31 = sext i32 %27 to i64
  %32 = getelementptr float, ptr addrspace(6) %2, i64 %3
  %33 = mul i64 %30, 16
  %34 = add i64 %33, %31
  %35 = getelementptr float, ptr addrspace(6) %32, i64 %34
  %36 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %35, i32 0, i32 3, i32 0)
  %37 = sext i32 %8 to i64
  %38 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %39 = mul i64 %37, 8192
  %40 = mul i64 %31, 512
  %41 = add i64 %39, %40
  %42 = add i64 %41, 0
  %43 = getelementptr float, ptr addrspace(6) %38, i64 %42
  %44 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %43, i32 0, i32 0, i32 0)
  %45 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %46 = mul i64 %31, 512
  %47 = add i64 %46, 0
  %48 = getelementptr float, ptr addrspace(6) %45, i64 %47
  %49 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %48, i32 0, i32 0, i32 0)
  %50 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %51 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %49, <64 x float> %44, <64 x float> %36, <256 x i1> %50)
  %52 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %53 = mul i64 %31, 512
  %54 = add i64 %53, 0
  %55 = getelementptr float, ptr addrspace(6) %52, i64 %54
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %51, ptr addrspace(6) %55, i32 0, i32 2, i32 0, <256 x i1> %50)
  %56 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %57 = mul i64 %37, 8192
  %58 = mul i64 %31, 512
  %59 = add i64 %57, %58
  %60 = add i64 %59, 64
  %61 = getelementptr float, ptr addrspace(6) %56, i64 %60
  %62 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %61, i32 0, i32 0, i32 0)
  %63 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %64 = mul i64 %31, 512
  %65 = add i64 %64, 64
  %66 = getelementptr float, ptr addrspace(6) %63, i64 %65
  %67 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %66, i32 0, i32 0, i32 0)
  %68 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %67, <64 x float> %62, <64 x float> %36, <256 x i1> %50)
  %69 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %70 = mul i64 %31, 512
  %71 = add i64 %70, 64
  %72 = getelementptr float, ptr addrspace(6) %69, i64 %71
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %68, ptr addrspace(6) %72, i32 0, i32 2, i32 0, <256 x i1> %50)
  %73 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %74 = mul i64 %37, 8192
  %75 = mul i64 %31, 512
  %76 = add i64 %74, %75
  %77 = add i64 %76, 128
  %78 = getelementptr float, ptr addrspace(6) %73, i64 %77
  %79 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %78, i32 0, i32 0, i32 0)
  %80 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %81 = mul i64 %31, 512
  %82 = add i64 %81, 128
  %83 = getelementptr float, ptr addrspace(6) %80, i64 %82
  %84 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %83, i32 0, i32 0, i32 0)
  %85 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %84, <64 x float> %79, <64 x float> %36, <256 x i1> %50)
  %86 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %87 = mul i64 %31, 512
  %88 = add i64 %87, 128
  %89 = getelementptr float, ptr addrspace(6) %86, i64 %88
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %85, ptr addrspace(6) %89, i32 0, i32 2, i32 0, <256 x i1> %50)
  %90 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %91 = mul i64 %37, 8192
  %92 = mul i64 %31, 512
  %93 = add i64 %91, %92
  %94 = add i64 %93, 192
  %95 = getelementptr float, ptr addrspace(6) %90, i64 %94
  %96 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %95, i32 0, i32 0, i32 0)
  %97 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %98 = mul i64 %31, 512
  %99 = add i64 %98, 192
  %100 = getelementptr float, ptr addrspace(6) %97, i64 %99
  %101 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %100, i32 0, i32 0, i32 0)
  %102 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %101, <64 x float> %96, <64 x float> %36, <256 x i1> %50)
  %103 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %104 = mul i64 %31, 512
  %105 = add i64 %104, 192
  %106 = getelementptr float, ptr addrspace(6) %103, i64 %105
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %102, ptr addrspace(6) %106, i32 0, i32 2, i32 0, <256 x i1> %50)
  %107 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %108 = mul i64 %37, 8192
  %109 = mul i64 %31, 512
  %110 = add i64 %108, %109
  %111 = add i64 %110, 256
  %112 = getelementptr float, ptr addrspace(6) %107, i64 %111
  %113 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %112, i32 0, i32 0, i32 0)
  %114 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %115 = mul i64 %31, 512
  %116 = add i64 %115, 256
  %117 = getelementptr float, ptr addrspace(6) %114, i64 %116
  %118 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %117, i32 0, i32 0, i32 0)
  %119 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %118, <64 x float> %113, <64 x float> %36, <256 x i1> %50)
  %120 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %121 = mul i64 %31, 512
  %122 = add i64 %121, 256
  %123 = getelementptr float, ptr addrspace(6) %120, i64 %122
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %119, ptr addrspace(6) %123, i32 0, i32 2, i32 0, <256 x i1> %50)
  %124 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %125 = mul i64 %37, 8192
  %126 = mul i64 %31, 512
  %127 = add i64 %125, %126
  %128 = add i64 %127, 320
  %129 = getelementptr float, ptr addrspace(6) %124, i64 %128
  %130 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %129, i32 0, i32 0, i32 0)
  %131 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %132 = mul i64 %31, 512
  %133 = add i64 %132, 320
  %134 = getelementptr float, ptr addrspace(6) %131, i64 %133
  %135 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %134, i32 0, i32 0, i32 0)
  %136 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %135, <64 x float> %130, <64 x float> %36, <256 x i1> %50)
  %137 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %138 = mul i64 %31, 512
  %139 = add i64 %138, 320
  %140 = getelementptr float, ptr addrspace(6) %137, i64 %139
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %136, ptr addrspace(6) %140, i32 0, i32 2, i32 0, <256 x i1> %50)
  %141 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %142 = mul i64 %37, 8192
  %143 = mul i64 %31, 512
  %144 = add i64 %142, %143
  %145 = add i64 %144, 384
  %146 = getelementptr float, ptr addrspace(6) %141, i64 %145
  %147 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %146, i32 0, i32 0, i32 0)
  %148 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %149 = mul i64 %31, 512
  %150 = add i64 %149, 384
  %151 = getelementptr float, ptr addrspace(6) %148, i64 %150
  %152 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %151, i32 0, i32 0, i32 0)
  %153 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %152, <64 x float> %147, <64 x float> %36, <256 x i1> %50)
  %154 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %155 = mul i64 %31, 512
  %156 = add i64 %155, 384
  %157 = getelementptr float, ptr addrspace(6) %154, i64 %156
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %153, ptr addrspace(6) %157, i32 0, i32 2, i32 0, <256 x i1> %50)
  %158 = getelementptr float, ptr addrspace(6) %10, i64 %11
  %159 = mul i64 %37, 8192
  %160 = mul i64 %31, 512
  %161 = add i64 %159, %160
  %162 = add i64 %161, 448
  %163 = getelementptr float, ptr addrspace(6) %158, i64 %162
  %164 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %163, i32 0, i32 0, i32 0)
  %165 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %166 = mul i64 %31, 512
  %167 = add i64 %166, 448
  %168 = getelementptr float, ptr addrspace(6) %165, i64 %167
  %169 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %168, i32 0, i32 0, i32 0)
  %170 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %169, <64 x float> %164, <64 x float> %36, <256 x i1> %50)
  %171 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %172 = mul i64 %31, 512
  %173 = add i64 %172, 448
  %174 = getelementptr float, ptr addrspace(6) %171, i64 %173
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %170, ptr addrspace(6) %174, i32 0, i32 2, i32 0, <256 x i1> %50)
  %175 = add nsw i32 %27, 1
  br label %26

176:                                              ; preds = %26
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_2(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) #0 {
  %8 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0.000000e+00)
  br label %9

9:                                                ; preds = %12, %7
  %10 = phi i32 [ %47, %12 ], [ 0, %7 ]
  %11 = icmp slt i32 %10, 16
  br i1 %11, label %12, label %48

12:                                               ; preds = %9
  %13 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %14 = sext i32 %10 to i64
  %15 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %16 = mul i64 %14, 512
  %17 = add i64 %16, 0
  %18 = getelementptr float, ptr addrspace(6) %15, i64 %17
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %18, i32 0, i32 2, i32 0, <256 x i1> %13)
  %19 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %20 = mul i64 %14, 512
  %21 = add i64 %20, 64
  %22 = getelementptr float, ptr addrspace(6) %19, i64 %21
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %22, i32 0, i32 2, i32 0, <256 x i1> %13)
  %23 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %24 = mul i64 %14, 512
  %25 = add i64 %24, 128
  %26 = getelementptr float, ptr addrspace(6) %23, i64 %25
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %26, i32 0, i32 2, i32 0, <256 x i1> %13)
  %27 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %28 = mul i64 %14, 512
  %29 = add i64 %28, 192
  %30 = getelementptr float, ptr addrspace(6) %27, i64 %29
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %30, i32 0, i32 2, i32 0, <256 x i1> %13)
  %31 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %32 = mul i64 %14, 512
  %33 = add i64 %32, 256
  %34 = getelementptr float, ptr addrspace(6) %31, i64 %33
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %34, i32 0, i32 2, i32 0, <256 x i1> %13)
  %35 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %36 = mul i64 %14, 512
  %37 = add i64 %36, 320
  %38 = getelementptr float, ptr addrspace(6) %35, i64 %37
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %38, i32 0, i32 2, i32 0, <256 x i1> %13)
  %39 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %40 = mul i64 %14, 512
  %41 = add i64 %40, 384
  %42 = getelementptr float, ptr addrspace(6) %39, i64 %41
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %42, i32 0, i32 2, i32 0, <256 x i1> %13)
  %43 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %44 = mul i64 %14, 512
  %45 = add i64 %44, 448
  %46 = getelementptr float, ptr addrspace(6) %43, i64 %45
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %46, i32 0, i32 2, i32 0, <256 x i1> %13)
  %47 = add nsw i32 %10, 1
  br label %9

48:                                               ; preds = %9
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_3(i32 %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2, i64 %3, i64 %4, i64 %5, ptr addrspace(6) noalias %6, ptr addrspace(6) noalias %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) noalias %11, ptr addrspace(6) noalias %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr addrspace(6) noalias %18, ptr addrspace(6) noalias %19, i64 %20, i64 %21, i64 %22, ptr addrspace(6) noalias %23, ptr addrspace(6) noalias %24, i64 %25, i64 %26, i64 %27) #0 {
  %29 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0.000000e+00)
  %30 = call <64 x float> @llvm.hivm.vbr.v64f32(float 1.000000e+00)
  %31 = call <256 x i1> @llvm.hivm.pge.b32(i32 6, i32 0)
  %32 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0xC7EFFFFFE0000000)
  br label %33

33:                                               ; preds = %37, %28
  %34 = phi i32 [ %44, %37 ], [ 0, %28 ]
  %35 = phi <64 x float> [ %43, %37 ], [ %32, %28 ]
  %36 = icmp slt i32 %34, %0
  br i1 %36, label %37, label %45

37:                                               ; preds = %33
  %38 = mul i32 %34, 16
  %39 = sext i32 %38 to i64
  %40 = getelementptr float, ptr addrspace(6) %2, i64 %3
  %41 = getelementptr float, ptr addrspace(6) %40, i64 %39
  %42 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %41, i32 0, i32 0, i32 0)
  %43 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %35, <64 x float> %42, <256 x i1> %31)
  %44 = add nsw i32 %34, 1
  br label %33

45:                                               ; preds = %33
  br label %46

46:                                               ; preds = %50, %45
  %47 = phi i32 [ %67, %50 ], [ 0, %45 ]
  %48 = phi <64 x float> [ %61, %50 ], [ %29, %45 ]
  %49 = icmp slt i32 %47, %0
  br i1 %49, label %50, label %68

50:                                               ; preds = %46
  %51 = mul i32 %47, 16
  %52 = sext i32 %51 to i64
  %53 = getelementptr float, ptr addrspace(6) %2, i64 %3
  %54 = getelementptr float, ptr addrspace(6) %53, i64 %52
  %55 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %54, i32 0, i32 0, i32 0)
  %56 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %55, <64 x float> %35, <256 x i1> %31, i32 0)
  %57 = getelementptr float, ptr addrspace(6) %7, i64 %8
  %58 = getelementptr float, ptr addrspace(6) %57, i64 %52
  %59 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %58, i32 0, i32 0, i32 0)
  %60 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %61 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %48, <64 x float> %59, <64 x float> %56, <256 x i1> %60)
  %62 = sext i32 %47 to i64
  %63 = getelementptr float, ptr addrspace(6) %12, i64 %13
  %64 = mul i64 %62, 16
  %65 = add i64 %64, 0
  %66 = getelementptr float, ptr addrspace(6) %63, i64 %65
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %56, ptr addrspace(6) %66, i32 0, i32 2, i32 0, <256 x i1> %31)
  %67 = add nsw i32 %47, 1
  br label %46

68:                                               ; preds = %46
  %69 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %30, <64 x float> %48, <256 x i1> %31)
  %70 = getelementptr float, ptr addrspace(6) %19, i64 %20
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %69, ptr addrspace(6) %70, i32 0, i32 2, i32 0, <256 x i1> %31)
  %71 = call <64 x float> @llvm.hivm.vln.x.v64f32(<64 x float> %48, <256 x i1> %31)
  %72 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %35, <64 x float> %71, <256 x i1> %31)
  %73 = getelementptr float, ptr addrspace(6) %24, i64 %25
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %72, ptr addrspace(6) %73, i32 0, i32 2, i32 0, <256 x i1> %31)
  call void @llvm.hivm.mem.bar.vst.vld()
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_4(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) noalias %5, ptr addrspace(6) noalias %6, i64 %7, i64 %8, i64 %9) #0 {
  %11 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0.000000e+00)
  %12 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %13 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %14 = getelementptr float, ptr addrspace(6) %13, i32 80
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %11, ptr addrspace(6) %14, i32 0, i32 2, i32 0, <256 x i1> %12)
  %15 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %16 = getelementptr float, ptr addrspace(6) %15, i32 80
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %11, ptr addrspace(6) %16, i32 0, i32 2, i32 0, <256 x i1> %12)
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_5(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) noalias %5, ptr addrspace(6) noalias %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr addrspace(6) noalias %12, ptr addrspace(6) noalias %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18) #0 {
  br label %20

20:                                               ; preds = %23, %19
  %21 = phi i32 [ %89, %23 ], [ 0, %19 ]
  %22 = icmp slt i32 %21, 48
  br i1 %22, label %23, label %90

23:                                               ; preds = %20
  %24 = sext i32 %21 to i64
  %25 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %26 = getelementptr float, ptr addrspace(6) %25, i64 %24
  %27 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %26, i32 0, i32 3, i32 0)
  %28 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %29 = mul i64 %24, 256
  %30 = add i64 %29, 0
  %31 = getelementptr float, ptr addrspace(6) %28, i64 %30
  %32 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %31, i32 0, i32 0, i32 0)
  %33 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %34 = mul i64 %24, 512
  %35 = add i64 %34, 256
  %36 = getelementptr float, ptr addrspace(6) %33, i64 %35
  %37 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %36, i32 0, i32 0, i32 0)
  %38 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %39 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %32, <64 x float> %37, <64 x float> %27, <256 x i1> %38)
  %40 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %41 = mul i64 %24, 512
  %42 = add i64 %41, 256
  %43 = getelementptr float, ptr addrspace(6) %40, i64 %42
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %39, ptr addrspace(6) %43, i32 0, i32 2, i32 0, <256 x i1> %38)
  %44 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %45 = mul i64 %24, 256
  %46 = add i64 %45, 64
  %47 = getelementptr float, ptr addrspace(6) %44, i64 %46
  %48 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %47, i32 0, i32 0, i32 0)
  %49 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %50 = mul i64 %24, 512
  %51 = add i64 %50, 320
  %52 = getelementptr float, ptr addrspace(6) %49, i64 %51
  %53 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %52, i32 0, i32 0, i32 0)
  %54 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %48, <64 x float> %53, <64 x float> %27, <256 x i1> %38)
  %55 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %56 = mul i64 %24, 512
  %57 = add i64 %56, 320
  %58 = getelementptr float, ptr addrspace(6) %55, i64 %57
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %54, ptr addrspace(6) %58, i32 0, i32 2, i32 0, <256 x i1> %38)
  %59 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %60 = mul i64 %24, 256
  %61 = add i64 %60, 128
  %62 = getelementptr float, ptr addrspace(6) %59, i64 %61
  %63 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %62, i32 0, i32 0, i32 0)
  %64 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %65 = mul i64 %24, 512
  %66 = add i64 %65, 384
  %67 = getelementptr float, ptr addrspace(6) %64, i64 %66
  %68 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %67, i32 0, i32 0, i32 0)
  %69 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %63, <64 x float> %68, <64 x float> %27, <256 x i1> %38)
  %70 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %71 = mul i64 %24, 512
  %72 = add i64 %71, 384
  %73 = getelementptr float, ptr addrspace(6) %70, i64 %72
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %69, ptr addrspace(6) %73, i32 0, i32 2, i32 0, <256 x i1> %38)
  %74 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %75 = mul i64 %24, 256
  %76 = add i64 %75, 192
  %77 = getelementptr float, ptr addrspace(6) %74, i64 %76
  %78 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %77, i32 0, i32 0, i32 0)
  %79 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %80 = mul i64 %24, 512
  %81 = add i64 %80, 448
  %82 = getelementptr float, ptr addrspace(6) %79, i64 %81
  %83 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %82, i32 0, i32 0, i32 0)
  %84 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %78, <64 x float> %83, <64 x float> %27, <256 x i1> %38)
  %85 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %86 = mul i64 %24, 512
  %87 = add i64 %86, 448
  %88 = getelementptr float, ptr addrspace(6) %85, i64 %87
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %84, ptr addrspace(6) %88, i32 0, i32 2, i32 0, <256 x i1> %38)
  %89 = add nsw i32 %21, 1
  br label %20

90:                                               ; preds = %20
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_6(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) noalias %7, ptr addrspace(6) noalias %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) #0 {
  br label %15

15:                                               ; preds = %18, %14
  %16 = phi i32 [ %57, %18 ], [ 0, %14 ]
  %17 = icmp slt i32 %16, 48
  br i1 %17, label %18, label %58

18:                                               ; preds = %15
  %19 = sext i32 %16 to i64
  %20 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %21 = mul i64 %19, 256
  %22 = add i64 %21, 0
  %23 = getelementptr float, ptr addrspace(6) %20, i64 %22
  %24 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %23, i32 0, i32 0, i32 0)
  %25 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %26 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %27 = mul i64 %19, 512
  %28 = add i64 %27, 256
  %29 = getelementptr float, ptr addrspace(6) %26, i64 %28
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %24, ptr addrspace(6) %29, i32 0, i32 2, i32 0, <256 x i1> %25)
  %30 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %31 = mul i64 %19, 256
  %32 = add i64 %31, 64
  %33 = getelementptr float, ptr addrspace(6) %30, i64 %32
  %34 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %33, i32 0, i32 0, i32 0)
  %35 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %36 = mul i64 %19, 512
  %37 = add i64 %36, 320
  %38 = getelementptr float, ptr addrspace(6) %35, i64 %37
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %34, ptr addrspace(6) %38, i32 0, i32 2, i32 0, <256 x i1> %25)
  %39 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %40 = mul i64 %19, 256
  %41 = add i64 %40, 128
  %42 = getelementptr float, ptr addrspace(6) %39, i64 %41
  %43 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %42, i32 0, i32 0, i32 0)
  %44 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %45 = mul i64 %19, 512
  %46 = add i64 %45, 384
  %47 = getelementptr float, ptr addrspace(6) %44, i64 %46
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %43, ptr addrspace(6) %47, i32 0, i32 2, i32 0, <256 x i1> %25)
  %48 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %49 = mul i64 %19, 256
  %50 = add i64 %49, 192
  %51 = getelementptr float, ptr addrspace(6) %48, i64 %50
  %52 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %51, i32 0, i32 0, i32 0)
  %53 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %54 = mul i64 %19, 512
  %55 = add i64 %54, 448
  %56 = getelementptr float, ptr addrspace(6) %53, i64 %55
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %52, ptr addrspace(6) %56, i32 0, i32 2, i32 0, <256 x i1> %25)
  %57 = add nsw i32 %16, 1
  br label %15

58:                                               ; preds = %15
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_7(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) noalias %5, ptr addrspace(6) noalias %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr addrspace(6) noalias %12, ptr addrspace(6) noalias %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18) #0 {
  br label %20

20:                                               ; preds = %23, %19
  %21 = phi i32 [ %89, %23 ], [ 0, %19 ]
  %22 = icmp slt i32 %21, 48
  br i1 %22, label %23, label %90

23:                                               ; preds = %20
  %24 = sext i32 %21 to i64
  %25 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %26 = getelementptr float, ptr addrspace(6) %25, i64 %24
  %27 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %26, i32 0, i32 3, i32 0)
  %28 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %29 = mul i64 %24, 256
  %30 = add i64 %29, 0
  %31 = getelementptr float, ptr addrspace(6) %28, i64 %30
  %32 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %31, i32 0, i32 0, i32 0)
  %33 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %34 = mul i64 %24, 512
  %35 = add i64 %34, 0
  %36 = getelementptr float, ptr addrspace(6) %33, i64 %35
  %37 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %36, i32 0, i32 0, i32 0)
  %38 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %39 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %32, <64 x float> %37, <64 x float> %27, <256 x i1> %38)
  %40 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %41 = mul i64 %24, 512
  %42 = add i64 %41, 0
  %43 = getelementptr float, ptr addrspace(6) %40, i64 %42
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %39, ptr addrspace(6) %43, i32 0, i32 2, i32 0, <256 x i1> %38)
  %44 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %45 = mul i64 %24, 256
  %46 = add i64 %45, 64
  %47 = getelementptr float, ptr addrspace(6) %44, i64 %46
  %48 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %47, i32 0, i32 0, i32 0)
  %49 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %50 = mul i64 %24, 512
  %51 = add i64 %50, 64
  %52 = getelementptr float, ptr addrspace(6) %49, i64 %51
  %53 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %52, i32 0, i32 0, i32 0)
  %54 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %48, <64 x float> %53, <64 x float> %27, <256 x i1> %38)
  %55 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %56 = mul i64 %24, 512
  %57 = add i64 %56, 64
  %58 = getelementptr float, ptr addrspace(6) %55, i64 %57
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %54, ptr addrspace(6) %58, i32 0, i32 2, i32 0, <256 x i1> %38)
  %59 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %60 = mul i64 %24, 256
  %61 = add i64 %60, 128
  %62 = getelementptr float, ptr addrspace(6) %59, i64 %61
  %63 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %62, i32 0, i32 0, i32 0)
  %64 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %65 = mul i64 %24, 512
  %66 = add i64 %65, 128
  %67 = getelementptr float, ptr addrspace(6) %64, i64 %66
  %68 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %67, i32 0, i32 0, i32 0)
  %69 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %63, <64 x float> %68, <64 x float> %27, <256 x i1> %38)
  %70 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %71 = mul i64 %24, 512
  %72 = add i64 %71, 128
  %73 = getelementptr float, ptr addrspace(6) %70, i64 %72
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %69, ptr addrspace(6) %73, i32 0, i32 2, i32 0, <256 x i1> %38)
  %74 = getelementptr float, ptr addrspace(6) %6, i64 %7
  %75 = mul i64 %24, 256
  %76 = add i64 %75, 192
  %77 = getelementptr float, ptr addrspace(6) %74, i64 %76
  %78 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %77, i32 0, i32 0, i32 0)
  %79 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %80 = mul i64 %24, 512
  %81 = add i64 %80, 192
  %82 = getelementptr float, ptr addrspace(6) %79, i64 %81
  %83 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %82, i32 0, i32 0, i32 0)
  %84 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %78, <64 x float> %83, <64 x float> %27, <256 x i1> %38)
  %85 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %86 = mul i64 %24, 512
  %87 = add i64 %86, 192
  %88 = getelementptr float, ptr addrspace(6) %85, i64 %87
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %84, ptr addrspace(6) %88, i32 0, i32 2, i32 0, <256 x i1> %38)
  %89 = add nsw i32 %21, 1
  br label %20

90:                                               ; preds = %20
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_8(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) noalias %7, ptr addrspace(6) noalias %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) #0 {
  br label %15

15:                                               ; preds = %18, %14
  %16 = phi i32 [ %57, %18 ], [ 0, %14 ]
  %17 = icmp slt i32 %16, 48
  br i1 %17, label %18, label %58

18:                                               ; preds = %15
  %19 = sext i32 %16 to i64
  %20 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %21 = mul i64 %19, 256
  %22 = add i64 %21, 0
  %23 = getelementptr float, ptr addrspace(6) %20, i64 %22
  %24 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %23, i32 0, i32 0, i32 0)
  %25 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %26 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %27 = mul i64 %19, 512
  %28 = add i64 %27, 0
  %29 = getelementptr float, ptr addrspace(6) %26, i64 %28
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %24, ptr addrspace(6) %29, i32 0, i32 2, i32 0, <256 x i1> %25)
  %30 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %31 = mul i64 %19, 256
  %32 = add i64 %31, 64
  %33 = getelementptr float, ptr addrspace(6) %30, i64 %32
  %34 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %33, i32 0, i32 0, i32 0)
  %35 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %36 = mul i64 %19, 512
  %37 = add i64 %36, 64
  %38 = getelementptr float, ptr addrspace(6) %35, i64 %37
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %34, ptr addrspace(6) %38, i32 0, i32 2, i32 0, <256 x i1> %25)
  %39 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %40 = mul i64 %19, 256
  %41 = add i64 %40, 128
  %42 = getelementptr float, ptr addrspace(6) %39, i64 %41
  %43 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %42, i32 0, i32 0, i32 0)
  %44 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %45 = mul i64 %19, 512
  %46 = add i64 %45, 128
  %47 = getelementptr float, ptr addrspace(6) %44, i64 %46
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %43, ptr addrspace(6) %47, i32 0, i32 2, i32 0, <256 x i1> %25)
  %48 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %49 = mul i64 %19, 256
  %50 = add i64 %49, 192
  %51 = getelementptr float, ptr addrspace(6) %48, i64 %50
  %52 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %51, i32 0, i32 0, i32 0)
  %53 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %54 = mul i64 %19, 512
  %55 = add i64 %54, 192
  %56 = getelementptr float, ptr addrspace(6) %53, i64 %55
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %52, ptr addrspace(6) %56, i32 0, i32 2, i32 0, <256 x i1> %25)
  %57 = add nsw i32 %16, 1
  br label %15

58:                                               ; preds = %15
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_9(i32 %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, ptr addrspace(6) noalias %8, ptr addrspace(6) noalias %9, i64 %10, i64 %11, i64 %12, ptr addrspace(6) noalias %13, ptr addrspace(6) noalias %14, i64 %15, i64 %16, i64 %17, ptr addrspace(6) noalias %18, ptr addrspace(6) noalias %19, i64 %20, i64 %21, i64 %22, ptr addrspace(6) noalias %23, ptr addrspace(6) noalias %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, ptr addrspace(6) noalias %30, ptr addrspace(6) noalias %31, i64 %32, i64 %33, i64 %34) #0 {
  %36 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0xFFF0000000000000)
  %37 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %38 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %0, <256 x i1> %37, i32 1)
  %39 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %40 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %39, <64 x i32> %38, <256 x i1> %37)
  %41 = call i32 @llvm.smin.i32(i32 %0, i32 112)
  %42 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %41, <256 x i1> %37, i32 1)
  %43 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 64, i32 0)
  %44 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %43, <64 x i32> %42, <256 x i1> %37)
  br label %45

45:                                               ; preds = %48, %35
  %46 = phi i32 [ %93, %48 ], [ 0, %35 ]
  %47 = icmp slt i32 %46, 24
  br i1 %47, label %48, label %94

48:                                               ; preds = %45
  %49 = sext i32 %46 to i64
  %50 = getelementptr float, ptr addrspace(6) %2, i64 %3
  %51 = mul i64 %49, 128
  %52 = add i64 %51, 0
  %53 = getelementptr float, ptr addrspace(6) %50, i64 %52
  %54 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %53, i32 0, i32 0, i32 0)
  %55 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %54, float 0x3FA47AE140000000, <256 x i1> %37)
  %56 = getelementptr float, ptr addrspace(6) %2, i64 %3
  %57 = mul i64 %49, 128
  %58 = add i64 %57, 64
  %59 = getelementptr float, ptr addrspace(6) %56, i64 %58
  %60 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %59, i32 0, i32 0, i32 0)
  %61 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %60, float 0x3FA47AE140000000, <256 x i1> %37)
  %62 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %55, <64 x float> %36, <256 x i1> %40)
  %63 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %61, <64 x float> %36, <256 x i1> %44)
  %64 = getelementptr float, ptr addrspace(6) %9, i64 %10
  %65 = getelementptr float, ptr addrspace(6) %64, i64 %49
  %66 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %65, i32 0, i32 3, i32 0)
  %67 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %62, <64 x float> %63, <256 x i1> %37)
  %68 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %67, <256 x i1> %37)
  %69 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %68, <256 x i1> %37, i32 1)
  %70 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %69, <64 x float> %66, <256 x i1> %37)
  %71 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %66, <64 x float> %70, <256 x i1> %37, i32 0)
  %72 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %62, <64 x float> %70, <256 x i1> %37, i32 0)
  %73 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %63, <64 x float> %70, <256 x i1> %37, i32 0)
  %74 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %72, <64 x float> %73, <256 x i1> %37)
  %75 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %74, <256 x i1> %37)
  %76 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %75, <256 x i1> %37, i32 1)
  %77 = getelementptr float, ptr addrspace(6) %9, i64 %10
  %78 = getelementptr float, ptr addrspace(6) %77, i64 %49
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %70, ptr addrspace(6) %78, i32 0, i32 5, i32 0, <256 x i1> %37)
  %79 = getelementptr float, ptr addrspace(6) %14, i64 %15
  %80 = getelementptr float, ptr addrspace(6) %79, i64 %49
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %76, ptr addrspace(6) %80, i32 0, i32 5, i32 0, <256 x i1> %37)
  %81 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %82 = getelementptr float, ptr addrspace(6) %81, i64 %49
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %71, ptr addrspace(6) %82, i32 0, i32 5, i32 0, <256 x i1> %37)
  %83 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %72, <256 x i1> %37, i32 1, i32 1, i32 0)
  %84 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %73, <256 x i1> %37, i32 1, i32 1, i32 0)
  %85 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %83, <128 x bfloat> %84)
  %86 = extractvalue { <128 x bfloat>, <128 x bfloat> } %85, 0
  %87 = mul i32 %46, 16
  %88 = call <256 x i1> @llvm.hivm.pge.b16(i32 0, i32 0)
  %89 = sext i32 %87 to i64
  %90 = getelementptr bfloat, ptr addrspace(6) %24, i64 %25
  %91 = add i64 0, %89
  %92 = getelementptr bfloat, ptr addrspace(6) %90, i64 %91
  call void @llvm.hivm.vsstb.v128bf16(<128 x bfloat> %86, ptr addrspace(6) %92, i32 3211264, i32 0, <256 x i1> %88)
  %93 = add nsw i32 %46, 1
  br label %45

94:                                               ; preds = %45
  %95 = add i32 %0, 1
  %96 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %95, <256 x i1> %37, i32 1)
  %97 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %39, <64 x i32> %96, <256 x i1> %37)
  %98 = call i32 @llvm.smin.i32(i32 %95, i32 112)
  %99 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %98, <256 x i1> %37, i32 1)
  %100 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %43, <64 x i32> %99, <256 x i1> %37)
  br label %101

101:                                              ; preds = %104, %94
  %102 = phi i32 [ %150, %104 ], [ 0, %94 ]
  %103 = icmp slt i32 %102, 24
  br i1 %103, label %104, label %151

104:                                              ; preds = %101
  %105 = add i32 %102, 24
  %106 = sext i32 %105 to i64
  %107 = getelementptr float, ptr addrspace(6) %2, i64 %3
  %108 = mul i64 %106, 128
  %109 = add i64 %108, 0
  %110 = getelementptr float, ptr addrspace(6) %107, i64 %109
  %111 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %110, i32 0, i32 0, i32 0)
  %112 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %111, float 0x3FA47AE140000000, <256 x i1> %37)
  %113 = getelementptr float, ptr addrspace(6) %2, i64 %3
  %114 = mul i64 %106, 128
  %115 = add i64 %114, 64
  %116 = getelementptr float, ptr addrspace(6) %113, i64 %115
  %117 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %116, i32 0, i32 0, i32 0)
  %118 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %117, float 0x3FA47AE140000000, <256 x i1> %37)
  %119 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %112, <64 x float> %36, <256 x i1> %97)
  %120 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %118, <64 x float> %36, <256 x i1> %100)
  %121 = getelementptr float, ptr addrspace(6) %9, i64 %10
  %122 = getelementptr float, ptr addrspace(6) %121, i64 %106
  %123 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %122, i32 0, i32 3, i32 0)
  %124 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %119, <64 x float> %120, <256 x i1> %37)
  %125 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %124, <256 x i1> %37)
  %126 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %125, <256 x i1> %37, i32 1)
  %127 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %126, <64 x float> %123, <256 x i1> %37)
  %128 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %123, <64 x float> %127, <256 x i1> %37, i32 0)
  %129 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %119, <64 x float> %127, <256 x i1> %37, i32 0)
  %130 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %120, <64 x float> %127, <256 x i1> %37, i32 0)
  %131 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %129, <64 x float> %130, <256 x i1> %37)
  %132 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %131, <256 x i1> %37)
  %133 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %132, <256 x i1> %37, i32 1)
  %134 = getelementptr float, ptr addrspace(6) %9, i64 %10
  %135 = getelementptr float, ptr addrspace(6) %134, i64 %106
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %127, ptr addrspace(6) %135, i32 0, i32 5, i32 0, <256 x i1> %37)
  %136 = getelementptr float, ptr addrspace(6) %14, i64 %15
  %137 = getelementptr float, ptr addrspace(6) %136, i64 %106
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %133, ptr addrspace(6) %137, i32 0, i32 5, i32 0, <256 x i1> %37)
  %138 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %139 = getelementptr float, ptr addrspace(6) %138, i64 %106
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %128, ptr addrspace(6) %139, i32 0, i32 5, i32 0, <256 x i1> %37)
  %140 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %129, <256 x i1> %37, i32 1, i32 1, i32 0)
  %141 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %130, <256 x i1> %37, i32 1, i32 1, i32 0)
  %142 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %140, <128 x bfloat> %141)
  %143 = extractvalue { <128 x bfloat>, <128 x bfloat> } %142, 0
  %144 = mul i32 %105, 16
  %145 = call <256 x i1> @llvm.hivm.pge.b16(i32 0, i32 0)
  %146 = sext i32 %144 to i64
  %147 = getelementptr bfloat, ptr addrspace(6) %24, i64 %25
  %148 = add i64 0, %146
  %149 = getelementptr bfloat, ptr addrspace(6) %147, i64 %148
  call void @llvm.hivm.vsstb.v128bf16(<128 x bfloat> %143, ptr addrspace(6) %149, i32 3211264, i32 0, <256 x i1> %145)
  %150 = add nsw i32 %102, 1
  br label %101

151:                                              ; preds = %101
  call void @llvm.hivm.mem.bar.vst.vld()
  %152 = call <64 x i32> @llvm.hivm.vbr.v64i32(i32 48)
  %153 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %39, <64 x i32> %152, <256 x i1> %37)
  %154 = getelementptr float, ptr addrspace(6) %14, i64 %15
  %155 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %154, i32 0, i32 0, i32 0)
  %156 = getelementptr float, ptr addrspace(6) %31, i64 %32
  %157 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %156, i32 0, i32 0, i32 0)
  %158 = getelementptr float, ptr addrspace(6) %19, i64 %20
  %159 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %158, i32 0, i32 0, i32 0)
  %160 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %155, <64 x float> %157, <64 x float> %159, <256 x i1> %37)
  %161 = getelementptr float, ptr addrspace(6) %31, i64 %32
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %160, ptr addrspace(6) %161, i32 0, i32 2, i32 0, <256 x i1> %153)
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_10(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) noalias %7, ptr addrspace(6) noalias %8, i64 %9, i64 %10, i64 %11, ptr addrspace(6) noalias %12, ptr addrspace(6) noalias %13, i64 %14, i64 %15, i64 %16, ptr addrspace(6) noalias %17, ptr addrspace(6) noalias %18, i64 %19, i64 %20, i64 %21, ptr addrspace(6) noalias %22, ptr addrspace(6) noalias %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, ptr addrspace(6) noalias %29, ptr addrspace(6) noalias %30, i64 %31, i64 %32, i64 %33) #0 {
  %35 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0xFFF0000000000000)
  %36 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %37 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 112, <256 x i1> %36, i32 1)
  %38 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 64, i32 0)
  %39 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %38, <64 x i32> %37, <256 x i1> %36)
  br label %40

40:                                               ; preds = %43, %34
  %41 = phi i32 [ %87, %43 ], [ 0, %34 ]
  %42 = icmp slt i32 %41, 48
  br i1 %42, label %43, label %88

43:                                               ; preds = %40
  %44 = sext i32 %41 to i64
  %45 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %46 = mul i64 %44, 128
  %47 = add i64 %46, 0
  %48 = getelementptr float, ptr addrspace(6) %45, i64 %47
  %49 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %48, i32 0, i32 0, i32 0)
  %50 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %49, float 0x3FA47AE140000000, <256 x i1> %36)
  %51 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %52 = mul i64 %44, 128
  %53 = add i64 %52, 64
  %54 = getelementptr float, ptr addrspace(6) %51, i64 %53
  %55 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %54, i32 0, i32 0, i32 0)
  %56 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %55, float 0x3FA47AE140000000, <256 x i1> %36)
  %57 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %56, <64 x float> %35, <256 x i1> %39)
  %58 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %59 = getelementptr float, ptr addrspace(6) %58, i64 %44
  %60 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %59, i32 0, i32 3, i32 0)
  %61 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %50, <64 x float> %57, <256 x i1> %36)
  %62 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %61, <256 x i1> %36)
  %63 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %62, <256 x i1> %36, i32 1)
  %64 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %63, <64 x float> %60, <256 x i1> %36)
  %65 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %60, <64 x float> %64, <256 x i1> %36, i32 0)
  %66 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %50, <64 x float> %64, <256 x i1> %36, i32 0)
  %67 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %57, <64 x float> %64, <256 x i1> %36, i32 0)
  %68 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %66, <64 x float> %67, <256 x i1> %36)
  %69 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %68, <256 x i1> %36)
  %70 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %69, <256 x i1> %36, i32 1)
  %71 = getelementptr float, ptr addrspace(6) %8, i64 %9
  %72 = getelementptr float, ptr addrspace(6) %71, i64 %44
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %64, ptr addrspace(6) %72, i32 0, i32 5, i32 0, <256 x i1> %36)
  %73 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %74 = getelementptr float, ptr addrspace(6) %73, i64 %44
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %70, ptr addrspace(6) %74, i32 0, i32 5, i32 0, <256 x i1> %36)
  %75 = getelementptr float, ptr addrspace(6) %18, i64 %19
  %76 = getelementptr float, ptr addrspace(6) %75, i64 %44
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %65, ptr addrspace(6) %76, i32 0, i32 5, i32 0, <256 x i1> %36)
  %77 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %66, <256 x i1> %36, i32 1, i32 1, i32 0)
  %78 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %67, <256 x i1> %36, i32 1, i32 1, i32 0)
  %79 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %77, <128 x bfloat> %78)
  %80 = extractvalue { <128 x bfloat>, <128 x bfloat> } %79, 0
  %81 = mul i32 %41, 16
  %82 = call <256 x i1> @llvm.hivm.pge.b16(i32 0, i32 0)
  %83 = sext i32 %81 to i64
  %84 = getelementptr bfloat, ptr addrspace(6) %23, i64 %24
  %85 = add i64 0, %83
  %86 = getelementptr bfloat, ptr addrspace(6) %84, i64 %85
  call void @llvm.hivm.vsstb.v128bf16(<128 x bfloat> %80, ptr addrspace(6) %86, i32 3211264, i32 0, <256 x i1> %82)
  %87 = add nsw i32 %41, 1
  br label %40

88:                                               ; preds = %40
  call void @llvm.hivm.mem.bar.vst.vld()
  %89 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %90 = call <64 x i32> @llvm.hivm.vbr.v64i32(i32 48)
  %91 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %89, <64 x i32> %90, <256 x i1> %36)
  %92 = getelementptr float, ptr addrspace(6) %13, i64 %14
  %93 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %92, i32 0, i32 0, i32 0)
  %94 = getelementptr float, ptr addrspace(6) %30, i64 %31
  %95 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %94, i32 0, i32 0, i32 0)
  %96 = getelementptr float, ptr addrspace(6) %18, i64 %19
  %97 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %96, i32 0, i32 0, i32 0)
  %98 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %93, <64 x float> %95, <64 x float> %97, <256 x i1> %36)
  %99 = getelementptr float, ptr addrspace(6) %30, i64 %31
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %98, ptr addrspace(6) %99, i32 0, i32 2, i32 0, <256 x i1> %91)
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_11(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) #0 {
  %8 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0.000000e+00)
  br label %9

9:                                                ; preds = %12, %7
  %10 = phi i32 [ %47, %12 ], [ 0, %7 ]
  %11 = icmp slt i32 %10, 48
  br i1 %11, label %12, label %48

12:                                               ; preds = %9
  %13 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %14 = sext i32 %10 to i64
  %15 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %16 = mul i64 %14, 512
  %17 = add i64 %16, 0
  %18 = getelementptr float, ptr addrspace(6) %15, i64 %17
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %18, i32 0, i32 2, i32 0, <256 x i1> %13)
  %19 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %20 = mul i64 %14, 512
  %21 = add i64 %20, 64
  %22 = getelementptr float, ptr addrspace(6) %19, i64 %21
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %22, i32 0, i32 2, i32 0, <256 x i1> %13)
  %23 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %24 = mul i64 %14, 512
  %25 = add i64 %24, 128
  %26 = getelementptr float, ptr addrspace(6) %23, i64 %25
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %26, i32 0, i32 2, i32 0, <256 x i1> %13)
  %27 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %28 = mul i64 %14, 512
  %29 = add i64 %28, 192
  %30 = getelementptr float, ptr addrspace(6) %27, i64 %29
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %30, i32 0, i32 2, i32 0, <256 x i1> %13)
  %31 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %32 = mul i64 %14, 512
  %33 = add i64 %32, 256
  %34 = getelementptr float, ptr addrspace(6) %31, i64 %33
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %34, i32 0, i32 2, i32 0, <256 x i1> %13)
  %35 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %36 = mul i64 %14, 512
  %37 = add i64 %36, 320
  %38 = getelementptr float, ptr addrspace(6) %35, i64 %37
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %38, i32 0, i32 2, i32 0, <256 x i1> %13)
  %39 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %40 = mul i64 %14, 512
  %41 = add i64 %40, 384
  %42 = getelementptr float, ptr addrspace(6) %39, i64 %41
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %42, i32 0, i32 2, i32 0, <256 x i1> %13)
  %43 = getelementptr float, ptr addrspace(6) %1, i64 %2
  %44 = mul i64 %14, 512
  %45 = add i64 %44, 448
  %46 = getelementptr float, ptr addrspace(6) %43, i64 %45
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %46, i32 0, i32 2, i32 0, <256 x i1> %13)
  %47 = add nsw i32 %10, 1
  br label %9

48:                                               ; preds = %9
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_scope_12(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) noalias %5, ptr addrspace(6) noalias %6, i64 %7, i64 %8, i64 %9) #0 {
  %11 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0xC7EFFFFFE0000000)
  %12 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %13 = getelementptr float, ptr addrspace(6) %1, i64 %2
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %11, ptr addrspace(6) %13, i32 0, i32 2, i32 0, <256 x i1> %12)
  %14 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0.000000e+00)
  %15 = getelementptr float, ptr addrspace(6) %6, i64 %7
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %14, ptr addrspace(6) %15, i32 0, i32 2, i32 0, <256 x i1> %12)
  ret void
}

; Function Attrs: noinline
define private void @copy_ubuf_to_cbuf_2d_bfloat16_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(2) %7, ptr addrspace(2) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) #1 {
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %15, ptr addrspace(6) %1, 1
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(2) %7, 0
  %24 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } %23, ptr addrspace(2) %8, 1
  %25 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } %24, i64 %9, 2
  %26 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } %25, i64 %10, 3, 0
  %27 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } %26, i64 %12, 4, 0
  %28 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } %27, i64 %11, 3, 1
  %29 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } %28, i64 %13, 4, 1
  %30 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [2 x i64], [2 x i64] } %29, ptr %30, align 8
  call void @_mlir_ciface_copy_ubuf_to_cbuf_2d_bfloat16_t(ptr %22, ptr %30)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_copy_ubuf_to_cbuf_2d_bfloat16_t(ptr, ptr) #1

; Function Attrs: noinline
define private void @store_ubuf_to_gm_3d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr addrspace(1) %9, ptr addrspace(1) %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i32 %18) #1 {
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %0, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %20, ptr addrspace(6) %1, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %21, i64 %2, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %22, i64 %3, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %23, i64 %6, 4, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %24, i64 %4, 3, 1
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %25, i64 %7, 4, 1
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %26, i64 %5, 3, 2
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %27, i64 %8, 4, 2
  %29 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %28, ptr %29, align 8
  %30 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(1) %9, 0
  %31 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %30, ptr addrspace(1) %10, 1
  %32 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %31, i64 %11, 2
  %33 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %32, i64 %12, 3, 0
  %34 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %33, i64 %15, 4, 0
  %35 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %34, i64 %13, 3, 1
  %36 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %35, i64 %16, 4, 1
  %37 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %36, i64 %14, 3, 2
  %38 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %37, i64 %17, 4, 2
  %39 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %38, ptr %39, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_3d_float(ptr %29, ptr %39, i32 %18)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_3d_float(ptr, ptr, i32) #1

; Function Attrs: noinline
define private void @store_ubuf_to_gm_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(1) %7, ptr addrspace(1) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i32 %14) #1 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %5, 4, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %4, 3, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %6, 4, 1
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, ptr %23, align 8
  %24 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %7, 0
  %25 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %24, ptr addrspace(1) %8, 1
  %26 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %25, i64 %9, 2
  %27 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %26, i64 %10, 3, 0
  %28 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %27, i64 %12, 4, 0
  %29 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %28, i64 %11, 3, 1
  %30 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %29, i64 %13, 4, 1
  %31 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %30, ptr %31, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_2d_float(ptr %23, ptr %31, i32 %14)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_2d_float(ptr, ptr, i32) #1

; Function Attrs: noinline
define private void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, float %11, i64 %12, i32 %13) #1 {
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, ptr addrspace(1) %1, 1
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, i64 %4, 4, 0
  %20 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, ptr %20, align 8
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %6, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_float_nocache(ptr %20, ptr %26, i32 %10, float %11, i64 %12, i32 %13)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_float_nocache(ptr, ptr, i32, float, i64, i32) #1

; Function Attrs: noinline
define private void @load_gm_to_ubuf_3d_float_nocache(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr addrspace(6) %9, ptr addrspace(6) %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i32 %18, float %19, i64 %20, i32 %21) #1 {
  %23 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(1) %0, 0
  %24 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %23, ptr addrspace(1) %1, 1
  %25 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %24, i64 %2, 2
  %26 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %25, i64 %3, 3, 0
  %27 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %26, i64 %6, 4, 0
  %28 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %27, i64 %4, 3, 1
  %29 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %28, i64 %7, 4, 1
  %30 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %29, i64 %5, 3, 2
  %31 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %30, i64 %8, 4, 2
  %32 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [3 x i64], [3 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %9, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %33, ptr addrspace(6) %10, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %34, i64 %11, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %35, i64 %12, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %36, i64 %15, 4, 0
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %37, i64 %13, 3, 1
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %38, i64 %16, 4, 1
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %39, i64 %14, 3, 2
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %40, i64 %17, 4, 2
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %41, ptr %42, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_3d_float_nocache(ptr %32, ptr %42, i32 %18, float %19, i64 %20, i32 %21)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_3d_float_nocache(ptr, ptr, i32, float, i64, i32) #1

; Function Attrs: noinline
define private void @store_ubuf_to_gm_2d_bfloat16_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(1) %7, ptr addrspace(1) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i32 %14) #1 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %5, 4, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %4, 3, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %6, 4, 1
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, ptr %23, align 8
  %24 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %7, 0
  %25 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %24, ptr addrspace(1) %8, 1
  %26 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %25, i64 %9, 2
  %27 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %26, i64 %10, 3, 0
  %28 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %27, i64 %12, 4, 0
  %29 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %28, i64 %11, 3, 1
  %30 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %29, i64 %13, 4, 1
  %31 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %30, ptr %31, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_2d_bfloat16_t(ptr %23, ptr %31, i32 %14)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_2d_bfloat16_t(ptr, ptr, i32) #1

; Function Attrs: noinline
define private void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #1 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr, ptr, i32) #1

define dso_local void @attention_mix_aiv(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, ptr addrspace(1) %9, ptr addrspace(1) %10, i32 %11, i32 %12, i32 %13) #2 {
  %15 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %16 = trunc i64 %15 to i32
  %17 = srem i32 %16, %11
  %18 = call i64 @llvm.hivm.GET.SUBBLOCKID()
  %19 = mul i32 %17, 2
  %20 = call i32 @llvm.smin.i32(i32 %19, i32 44)
  %21 = add i32 %20, 2
  %22 = call i32 @llvm.smin.i32(i32 %21, i32 44)
  %23 = sdiv i32 %20, 11
  %24 = icmp slt i32 %20, %22
  %25 = add i32 %22, 10
  %26 = sdiv i32 %25, 11
  %27 = select i1 %24, i32 %26, i32 %23
  br label %28

28:                                               ; preds = %113, %14
  %29 = phi i32 [ %123, %113 ], [ %23, %14 ]
  %30 = icmp slt i32 %29, %27
  br i1 %30, label %31, label %124

31:                                               ; preds = %28
  %32 = mul i32 %29, 11
  %33 = sdiv i32 %32, 2
  %34 = sub i32 %17, %33
  %35 = mul i32 %29, 96
  %36 = sext i32 %29 to i64
  %37 = getelementptr i32, ptr addrspace(1) %5, i64 %36
  %38 = load i32, ptr addrspace(1) %37, align 4
  %39 = add i32 %38, 111
  %40 = sdiv i32 %39, 112
  %41 = call i32 @llvm.smax.i32(i32 %32, i32 %20)
  %42 = add i32 %32, %40
  %43 = call i32 @llvm.smin.i32(i32 %42, i32 %22)
  %44 = sub i32 %41, %32
  %45 = mul i32 %44, 112
  %46 = sub i32 %43, %41
  %47 = call i32 @llvm.smax.i32(i32 %46, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call ptc_simdvf void @attention_scope_12(ptr addrspace(6) null, ptr addrspace(6) null, i64 55296, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 55360, i64 64, i64 1)
  %48 = icmp eq i32 %47, 0
  br i1 %48, label %49, label %50

49:                                               ; preds = %31
  call ptc_simdvf void @attention_scope_11(ptr addrspace(6) null, ptr addrspace(6) null, i64 30720, i64 48, i64 512, i64 512, i64 1)
  br label %50

50:                                               ; preds = %49, %31
  %51 = add i32 %47, 1
  br label %52

52:                                               ; preds = %111, %50
  %53 = phi i32 [ %112, %111 ], [ -1, %50 ]
  %54 = icmp slt i32 %53, %51
  br i1 %54, label %55, label %113

55:                                               ; preds = %52
  %56 = icmp sge i32 %53, 0
  %57 = icmp slt i32 %53, %47
  %58 = and i1 %56, %57
  %59 = icmp sgt i32 %53, 0
  %60 = srem i32 %53, 3
  %61 = srem i32 %53, 2
  br i1 %58, label %62, label %94

62:                                               ; preds = %55
  %63 = mul i64 %18, 48
  %64 = add i32 %38, -3
  %65 = sub i32 %64, %45
  %66 = mul i32 %53, 112
  %67 = sub i32 %65, %66
  %68 = sdiv i64 %63, 24
  %69 = sext i32 %67 to i64
  %70 = add i64 %69, %68
  %71 = trunc i64 %70 to i32
  %72 = mul i32 %61, 64
  %73 = sext i32 %72 to i64
  %74 = add i64 %73, 55424
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 1, i64 0)
  %75 = icmp slt i32 %71, 112
  %76 = zext i1 %75 to i32
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %78, label %79

78:                                               ; preds = %62
  call ptc_simdvf void @attention_scope_10(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 48, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 55296, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 222208 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 222208 to ptr addrspace(6)), i64 %73, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 %74, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), i64 0, i64 8, i64 784, i64 784, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 55360, i64 64, i64 1)
  br label %79

79:                                               ; preds = %78, %62
  %80 = icmp eq i32 %76, 1
  br i1 %80, label %81, label %82

81:                                               ; preds = %79
  call ptc_simdvf void @attention_scope_9(i32 %71, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 48, i64 128, i64 128, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 55296, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 222208 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 222208 to ptr addrspace(6)), i64 %73, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 %74, i64 64, i64 1, ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), i64 0, i64 8, i64 784, i64 784, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 55360, i64 64, i64 1)
  br label %82

82:                                               ; preds = %81, %79
  %83 = add i32 %53, 1
  %84 = icmp slt i32 %83, %47
  br i1 %84, label %85, label %86

85:                                               ; preds = %82
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 1, i64 1)
  br label %86

86:                                               ; preds = %85, %82
  %87 = mul i32 %60, 68096
  %88 = add i32 %87, 57344
  %89 = sext i32 %88 to i64
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  %90 = mul i64 %18, 768
  %91 = add i64 %89, %90
  call void @copy_ubuf_to_cbuf_2d_bfloat16_t(ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), i64 0, i64 7, i64 768, i64 784, i64 1, ptr addrspace(2) null, ptr addrspace(2) null, i64 %91, i64 7, i64 768, i64 1536, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  %92 = add i32 %60, 10
  %93 = sext i32 %92 to i64
  call void @llvm.hivm.SET.INTRA.BLOCK.mode(i64 5, i64 %93)
  br label %94

94:                                               ; preds = %86, %55
  br i1 %59, label %95, label %111

95:                                               ; preds = %94
  %96 = sub i32 1, %61
  %97 = mul i32 %96, 64
  %98 = sext i32 %97 to i64
  %99 = add i64 %98, 55424
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 1, i64 4)
  %100 = icmp eq i32 %53, 1
  br i1 %100, label %101, label %102

101:                                              ; preds = %95
  call ptc_simdvf void @attention_scope_8(ptr addrspace(6) null, ptr addrspace(6) null, i64 6144, i64 48, i64 256, i64 256, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 30720, i64 48, i64 512, i64 512, i64 1)
  br label %103

102:                                              ; preds = %95
  call ptc_simdvf void @attention_scope_7(ptr addrspace(6) null, ptr addrspace(6) null, i64 %99, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 6144, i64 48, i64 256, i64 256, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 30720, i64 48, i64 512, i64 512, i64 1)
  br label %103

103:                                              ; preds = %101, %102
  br i1 %58, label %104, label %105

104:                                              ; preds = %103
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 1, i64 5)
  br label %105

105:                                              ; preds = %104, %103
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 1, i64 8)
  br i1 %100, label %106, label %107

106:                                              ; preds = %105
  call ptc_simdvf void @attention_scope_6(ptr addrspace(6) null, ptr addrspace(6) null, i64 18432, i64 48, i64 256, i64 256, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 30720, i64 48, i64 512, i64 512, i64 1)
  br label %108

107:                                              ; preds = %105
  call ptc_simdvf void @attention_scope_5(ptr addrspace(6) null, ptr addrspace(6) null, i64 %99, i64 64, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 18432, i64 48, i64 256, i64 256, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 30720, i64 48, i64 512, i64 512, i64 1)
  br label %108

108:                                              ; preds = %106, %107
  br i1 %58, label %109, label %110

109:                                              ; preds = %108
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 1, i64 9)
  br label %110

110:                                              ; preds = %109, %108
  br label %111

111:                                              ; preds = %110, %94
  %112 = add nsw i32 %53, 1
  br label %52

113:                                              ; preds = %52
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  %114 = mul i64 %18, 48
  %115 = sext i32 %35 to i64
  %116 = add i64 %115, %114
  %117 = sdiv i64 %116, 16
  %118 = mul i64 %117, 6
  %119 = sext i32 %34 to i64
  %120 = add i64 %118, %119
  %121 = mul i64 %120, 16
  %122 = mul i64 %120, 8192
  call void @store_ubuf_to_gm_3d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 30720, i64 3, i64 16, i64 512, i64 8192, i64 512, i64 1, ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %122, i64 3, i64 16, i64 512, i64 49152, i64 512, i64 1, i32 0)
  call void @store_ubuf_to_gm_2d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 55296, i64 3, i64 16, i64 16, i64 1, ptr addrspace(1) %7, ptr addrspace(1) %7, i64 %121, i64 3, i64 16, i64 96, i64 1, i32 0)
  call void @store_ubuf_to_gm_2d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 55360, i64 3, i64 16, i64 16, i64 1, ptr addrspace(1) %8, ptr addrspace(1) %8, i64 %121, i64 3, i64 16, i64 96, i64 1, i32 0)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 5, i64 6)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 0, i64 7)
  %123 = add nsw i32 %29, 1
  br label %28

124:                                              ; preds = %28
  call void @llvm.hivm.BARRIER(i64 5)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 3585)
  call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 14)
  %125 = trunc i64 %18 to i32
  %126 = add i32 %19, %125
  call ptc_simdvf void @attention_scope_4(ptr addrspace(6) null, ptr addrspace(6) null, i64 24576, i64 144, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 24720, i64 144, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  br label %127

127:                                              ; preds = %171, %124
  %128 = phi i32 [ %175, %171 ], [ %126, %124 ]
  %129 = icmp slt i32 %128, 24
  br i1 %129, label %130, label %176

130:                                              ; preds = %127
  %131 = mul i32 %128, 16
  %132 = sdiv i32 %131, 96
  %133 = mul i32 %132, 11
  %134 = add i32 %133, 10
  %135 = sdiv i32 %134, 2
  %136 = sdiv i32 %133, 2
  %137 = sub i32 %135, %136
  %138 = add i32 %137, 1
  %139 = mul i32 %128, 96
  %140 = sext i32 %139 to i64
  call void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %7, ptr addrspace(1) %7, i64 %140, i64 96, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 24576, i64 96, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %8, ptr addrspace(1) %8, i64 %140, i64 96, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 24720, i64 96, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 2)
  %141 = icmp sgt i32 %138, 0
  br i1 %141, label %142, label %145

142:                                              ; preds = %130
  %143 = mul i32 %128, 49152
  %144 = sext i32 %143 to i64
  call void @load_gm_to_ubuf_3d_float_nocache(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %144, i64 1, i64 16, i64 512, i64 8192, i64 512, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1, i64 16, i64 512, i64 8192, i64 512, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 3)
  br label %145

145:                                              ; preds = %142, %130
  %146 = icmp sgt i32 %138, 1
  br i1 %146, label %147, label %151

147:                                              ; preds = %145
  %148 = add i32 %139, 16
  %149 = mul i32 %148, 512
  %150 = sext i32 %149 to i64
  call void @load_gm_to_ubuf_3d_float_nocache(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %150, i64 1, i64 16, i64 512, i64 8192, i64 512, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 8192, i64 1, i64 16, i64 512, i64 8192, i64 512, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 4)
  br label %151

151:                                              ; preds = %147, %145
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 2)
  call ptc_simdvf void @attention_scope_3(i32 %138, ptr addrspace(6) null, ptr addrspace(6) null, i64 24576, i64 144, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 24720, i64 144, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 24864, i64 6, i64 16, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 24976, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 24960, i64 16, i64 1)
  call ptc_simdvf void @attention_scope_2(ptr addrspace(6) null, ptr addrspace(6) null, i64 16384, i64 16, i64 512, i64 512, i64 1)
  br label %152

152:                                              ; preds = %169, %151
  %153 = phi i32 [ %170, %169 ], [ 0, %151 ]
  %154 = icmp slt i32 %153, %138
  br i1 %154, label %155, label %171

155:                                              ; preds = %152
  %156 = srem i32 %153, 2
  %157 = add i32 %156, 3
  %158 = sext i32 %157 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 4, i64 1, i64 %158)
  call ptc_simdvf void @attention_scope_1(i32 %153, ptr addrspace(6) null, ptr addrspace(6) null, i64 24864, i64 6, i64 16, i64 16, i64 1, i32 %156, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 2, i64 16, i64 512, i64 8192, i64 512, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 16384, i64 16, i64 512, i64 512, i64 1)
  %159 = add i32 %153, 2
  %160 = icmp slt i32 %159, %138
  br i1 %160, label %161, label %169

161:                                              ; preds = %155
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %158)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %158)
  %162 = mul i32 %159, 16
  %163 = add i32 %139, %162
  %164 = mul i32 %163, 512
  %165 = sext i32 %164 to i64
  %166 = sext i32 %156 to i64
  %167 = mul i64 %166, 8192
  %168 = add i64 %167, 0
  call void @load_gm_to_ubuf_3d_float_nocache(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %165, i64 1, i64 16, i64 512, i64 8192, i64 512, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 %168, i64 1, i64 16, i64 512, i64 8192, i64 512, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 4, i64 1, i64 %158)
  br label %169

169:                                              ; preds = %161, %155
  %170 = add nsw i32 %153, 1
  br label %152

171:                                              ; preds = %152
  call ptc_simdvf void @attention_scope_0(ptr addrspace(6) null, ptr addrspace(6) null, i64 24976, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 16384, i64 16, i64 512, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), i64 0, i64 16, i64 512, i64 512, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 2)
  %172 = mul i32 %128, 8192
  %173 = sext i32 %172 to i64
  call void @store_ubuf_to_gm_2d_bfloat16_t(ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 222720 to ptr addrspace(6)), i64 0, i64 16, i64 512, i64 512, i64 1, ptr addrspace(1) %9, ptr addrspace(1) %9, i64 %173, i64 16, i64 512, i64 512, i64 1, i32 0)
  %174 = sext i32 %131 to i64
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 24960, i64 16, i64 1, ptr addrspace(1) %10, ptr addrspace(1) %10, i64 %174, i64 16, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 6)
  %175 = add nsw i32 %128, 56
  br label %127

176:                                              ; preds = %127
  ret void
}

; Function Attrs: nounwind readonly 
declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #3

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #4

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32) #5

; Function Attrs: nounwind readnone 
declare { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat>, <128 x bfloat>) #5

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b16(i32, i32) #4

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float>, <64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vbr.v64f32(float) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vln.x.v64f32(<64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind
declare void @llvm.hivm.mem.bar.vst.vld() #4

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float>, <64 x float>, <256 x i1>, i32) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32) #5

; Function Attrs: nounwind readonly 
declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32) #3

; Function Attrs: nounwind readnone 
declare <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smin.i32(i32, i32) #7

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vbr.v64i32(i32) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float>, <256 x i1>, i32) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>) #5

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vsstb.v128bf16(<128 x bfloat>, ptr addrspace(6) nocapture readonly, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #5

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.SUBBLOCKID() #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #8

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CROSS.CORE(i64, i64) #8

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64, i64) #8

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smax.i32(i32, i32) #7

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64, i64) #8

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64, i64) #8

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.INTRA.BLOCK.mode(i64, i64) #8

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #3 = { nounwind readonly  }
attributes #4 = { nounwind }
attributes #5 = { nounwind readnone  }
attributes #6 = { nounwind writeonly  }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn readnone  }
attributes #8 = { nounwind  inaccessiblememonly }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @attention_mix_aiv, !"kernel", i32 1}

