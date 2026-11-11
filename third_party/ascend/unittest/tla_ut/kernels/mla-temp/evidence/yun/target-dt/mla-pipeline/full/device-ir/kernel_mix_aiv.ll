; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_splits_scope_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2) #0 {
  br label %4

4:                                                ; preds = %7, %3
  %5 = phi i32 [ %108, %7 ], [ 0, %3 ]
  %6 = icmp slt i32 %5, 48
  br i1 %6, label %7, label %109

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64
  %9 = getelementptr float, ptr addrspace(6) %0, i64 %8
  %10 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %9, i32 0, i32 3, i32 0)
  %11 = mul i64 %8, 512
  %12 = add i64 %11, 0
  %13 = getelementptr float, ptr addrspace(6) %1, i64 %12
  %14 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %13, i32 0, i32 0, i32 0)
  %15 = mul i64 %8, 512
  %16 = add i64 %15, 0
  %17 = getelementptr float, ptr addrspace(6) %2, i64 %16
  %18 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %17, i32 0, i32 0, i32 0)
  %19 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %20 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %14, <64 x float> %18, <64 x float> %10, <256 x i1> %19)
  %21 = mul i64 %8, 512
  %22 = add i64 %21, 0
  %23 = getelementptr float, ptr addrspace(6) %2, i64 %22
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %20, ptr addrspace(6) %23, i32 0, i32 2, i32 0, <256 x i1> %19)
  %24 = mul i64 %8, 512
  %25 = add i64 %24, 64
  %26 = getelementptr float, ptr addrspace(6) %1, i64 %25
  %27 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %26, i32 0, i32 0, i32 0)
  %28 = mul i64 %8, 512
  %29 = add i64 %28, 64
  %30 = getelementptr float, ptr addrspace(6) %2, i64 %29
  %31 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %30, i32 0, i32 0, i32 0)
  %32 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %27, <64 x float> %31, <64 x float> %10, <256 x i1> %19)
  %33 = mul i64 %8, 512
  %34 = add i64 %33, 64
  %35 = getelementptr float, ptr addrspace(6) %2, i64 %34
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %32, ptr addrspace(6) %35, i32 0, i32 2, i32 0, <256 x i1> %19)
  %36 = mul i64 %8, 512
  %37 = add i64 %36, 128
  %38 = getelementptr float, ptr addrspace(6) %1, i64 %37
  %39 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %38, i32 0, i32 0, i32 0)
  %40 = mul i64 %8, 512
  %41 = add i64 %40, 128
  %42 = getelementptr float, ptr addrspace(6) %2, i64 %41
  %43 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %42, i32 0, i32 0, i32 0)
  %44 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %39, <64 x float> %43, <64 x float> %10, <256 x i1> %19)
  %45 = mul i64 %8, 512
  %46 = add i64 %45, 128
  %47 = getelementptr float, ptr addrspace(6) %2, i64 %46
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %44, ptr addrspace(6) %47, i32 0, i32 2, i32 0, <256 x i1> %19)
  %48 = mul i64 %8, 512
  %49 = add i64 %48, 192
  %50 = getelementptr float, ptr addrspace(6) %1, i64 %49
  %51 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %50, i32 0, i32 0, i32 0)
  %52 = mul i64 %8, 512
  %53 = add i64 %52, 192
  %54 = getelementptr float, ptr addrspace(6) %2, i64 %53
  %55 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %54, i32 0, i32 0, i32 0)
  %56 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %51, <64 x float> %55, <64 x float> %10, <256 x i1> %19)
  %57 = mul i64 %8, 512
  %58 = add i64 %57, 192
  %59 = getelementptr float, ptr addrspace(6) %2, i64 %58
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %56, ptr addrspace(6) %59, i32 0, i32 2, i32 0, <256 x i1> %19)
  %60 = mul i64 %8, 512
  %61 = add i64 %60, 256
  %62 = getelementptr float, ptr addrspace(6) %1, i64 %61
  %63 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %62, i32 0, i32 0, i32 0)
  %64 = mul i64 %8, 512
  %65 = add i64 %64, 256
  %66 = getelementptr float, ptr addrspace(6) %2, i64 %65
  %67 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %66, i32 0, i32 0, i32 0)
  %68 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %63, <64 x float> %67, <64 x float> %10, <256 x i1> %19)
  %69 = mul i64 %8, 512
  %70 = add i64 %69, 256
  %71 = getelementptr float, ptr addrspace(6) %2, i64 %70
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %68, ptr addrspace(6) %71, i32 0, i32 2, i32 0, <256 x i1> %19)
  %72 = mul i64 %8, 512
  %73 = add i64 %72, 320
  %74 = getelementptr float, ptr addrspace(6) %1, i64 %73
  %75 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %74, i32 0, i32 0, i32 0)
  %76 = mul i64 %8, 512
  %77 = add i64 %76, 320
  %78 = getelementptr float, ptr addrspace(6) %2, i64 %77
  %79 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %78, i32 0, i32 0, i32 0)
  %80 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %75, <64 x float> %79, <64 x float> %10, <256 x i1> %19)
  %81 = mul i64 %8, 512
  %82 = add i64 %81, 320
  %83 = getelementptr float, ptr addrspace(6) %2, i64 %82
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %80, ptr addrspace(6) %83, i32 0, i32 2, i32 0, <256 x i1> %19)
  %84 = mul i64 %8, 512
  %85 = add i64 %84, 384
  %86 = getelementptr float, ptr addrspace(6) %1, i64 %85
  %87 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %86, i32 0, i32 0, i32 0)
  %88 = mul i64 %8, 512
  %89 = add i64 %88, 384
  %90 = getelementptr float, ptr addrspace(6) %2, i64 %89
  %91 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %90, i32 0, i32 0, i32 0)
  %92 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %87, <64 x float> %91, <64 x float> %10, <256 x i1> %19)
  %93 = mul i64 %8, 512
  %94 = add i64 %93, 384
  %95 = getelementptr float, ptr addrspace(6) %2, i64 %94
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %92, ptr addrspace(6) %95, i32 0, i32 2, i32 0, <256 x i1> %19)
  %96 = mul i64 %8, 512
  %97 = add i64 %96, 448
  %98 = getelementptr float, ptr addrspace(6) %1, i64 %97
  %99 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %98, i32 0, i32 0, i32 0)
  %100 = mul i64 %8, 512
  %101 = add i64 %100, 448
  %102 = getelementptr float, ptr addrspace(6) %2, i64 %101
  %103 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %102, i32 0, i32 0, i32 0)
  %104 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %99, <64 x float> %103, <64 x float> %10, <256 x i1> %19)
  %105 = mul i64 %8, 512
  %106 = add i64 %105, 448
  %107 = getelementptr float, ptr addrspace(6) %2, i64 %106
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %104, ptr addrspace(6) %107, i32 0, i32 2, i32 0, <256 x i1> %19)
  %108 = add nsw i32 %5, 1
  br label %4

109:                                              ; preds = %4
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_splits_scope_1(i64 %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2, ptr addrspace(6) noalias %3, ptr addrspace(6) noalias %4, ptr addrspace(6) noalias %5) #0 {
  %7 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0xFFF0000000000000)
  %8 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %9 = trunc i64 %0 to i32
  %10 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %9, <256 x i1> %8, i32 1)
  %11 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %12 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %11, <64 x i32> %10, <256 x i1> %8)
  %13 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 64, i32 0)
  %14 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %13, <64 x i32> %10, <256 x i1> %8)
  br label %15

15:                                               ; preds = %18, %6
  %16 = phi i32 [ %59, %18 ], [ 0, %6 ]
  %17 = icmp slt i32 %16, 24
  br i1 %17, label %18, label %60

18:                                               ; preds = %15
  %19 = sext i32 %16 to i64
  %20 = mul i64 %19, 128
  %21 = add i64 %20, 0
  %22 = getelementptr float, ptr addrspace(6) %1, i64 %21
  %23 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %22, i32 0, i32 0, i32 0)
  %24 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %23, float 0x3FA47AE140000000, <256 x i1> %8)
  %25 = mul i64 %19, 128
  %26 = add i64 %25, 64
  %27 = getelementptr float, ptr addrspace(6) %1, i64 %26
  %28 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %27, i32 0, i32 0, i32 0)
  %29 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %28, float 0x3FA47AE140000000, <256 x i1> %8)
  %30 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %24, <64 x float> %7, <256 x i1> %12)
  %31 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %29, <64 x float> %7, <256 x i1> %14)
  %32 = getelementptr float, ptr addrspace(6) %2, i64 %19
  %33 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %32, i32 0, i32 3, i32 0)
  %34 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %30, <64 x float> %31, <256 x i1> %8)
  %35 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %34, <256 x i1> %8)
  %36 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %35, <256 x i1> %8, i32 1)
  %37 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %36, <64 x float> %33, <256 x i1> %8)
  %38 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %33, <64 x float> %37, <256 x i1> %8, i32 0)
  %39 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %30, <64 x float> %37, <256 x i1> %8, i32 0)
  %40 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %31, <64 x float> %37, <256 x i1> %8, i32 0)
  %41 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %39, <64 x float> %40, <256 x i1> %8)
  %42 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %41, <256 x i1> %8)
  %43 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %42, <256 x i1> %8, i32 1)
  %44 = getelementptr float, ptr addrspace(6) %3, i64 %19
  %45 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %44, i32 0, i32 3, i32 0)
  %46 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %43, <64 x float> %45, <64 x float> %38, <256 x i1> %8)
  %47 = getelementptr float, ptr addrspace(6) %2, i64 %19
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %37, ptr addrspace(6) %47, i32 0, i32 5, i32 0, <256 x i1> %8)
  %48 = getelementptr float, ptr addrspace(6) %3, i64 %19
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %46, ptr addrspace(6) %48, i32 0, i32 5, i32 0, <256 x i1> %8)
  %49 = getelementptr float, ptr addrspace(6) %4, i64 %19
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %38, ptr addrspace(6) %49, i32 0, i32 5, i32 0, <256 x i1> %8)
  %50 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %39, <256 x i1> %8, i32 1, i32 1, i32 0)
  %51 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %40, <256 x i1> %8, i32 1, i32 1, i32 0)
  %52 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %50, <128 x bfloat> %51)
  %53 = extractvalue { <128 x bfloat>, <128 x bfloat> } %52, 0
  %54 = mul i32 %16, 16
  %55 = call <256 x i1> @llvm.hivm.pge.b16(i32 0, i32 0)
  %56 = sext i32 %54 to i64
  %57 = add i64 0, %56
  %58 = getelementptr bfloat, ptr addrspace(6) %5, i64 %57
  call void @llvm.hivm.vsstb.v128bf16(<128 x bfloat> %53, ptr addrspace(6) %58, i32 3211264, i32 0, <256 x i1> %55)
  %59 = add nsw i32 %16, 1
  br label %15

60:                                               ; preds = %15
  %61 = add i64 %0, 1
  %62 = trunc i64 %61 to i32
  %63 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %62, <256 x i1> %8, i32 1)
  %64 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %11, <64 x i32> %63, <256 x i1> %8)
  %65 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %13, <64 x i32> %63, <256 x i1> %8)
  br label %66

66:                                               ; preds = %69, %60
  %67 = phi i32 [ %111, %69 ], [ 0, %60 ]
  %68 = icmp slt i32 %67, 24
  br i1 %68, label %69, label %112

69:                                               ; preds = %66
  %70 = add i32 %67, 24
  %71 = sext i32 %70 to i64
  %72 = mul i64 %71, 128
  %73 = add i64 %72, 0
  %74 = getelementptr float, ptr addrspace(6) %1, i64 %73
  %75 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %74, i32 0, i32 0, i32 0)
  %76 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %75, float 0x3FA47AE140000000, <256 x i1> %8)
  %77 = mul i64 %71, 128
  %78 = add i64 %77, 64
  %79 = getelementptr float, ptr addrspace(6) %1, i64 %78
  %80 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %79, i32 0, i32 0, i32 0)
  %81 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %80, float 0x3FA47AE140000000, <256 x i1> %8)
  %82 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %76, <64 x float> %7, <256 x i1> %64)
  %83 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %81, <64 x float> %7, <256 x i1> %65)
  %84 = getelementptr float, ptr addrspace(6) %2, i64 %71
  %85 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %84, i32 0, i32 3, i32 0)
  %86 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %82, <64 x float> %83, <256 x i1> %8)
  %87 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %86, <256 x i1> %8)
  %88 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %87, <256 x i1> %8, i32 1)
  %89 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %88, <64 x float> %85, <256 x i1> %8)
  %90 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %85, <64 x float> %89, <256 x i1> %8, i32 0)
  %91 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %82, <64 x float> %89, <256 x i1> %8, i32 0)
  %92 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %83, <64 x float> %89, <256 x i1> %8, i32 0)
  %93 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %91, <64 x float> %92, <256 x i1> %8)
  %94 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %93, <256 x i1> %8)
  %95 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %94, <256 x i1> %8, i32 1)
  %96 = getelementptr float, ptr addrspace(6) %3, i64 %71
  %97 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %96, i32 0, i32 3, i32 0)
  %98 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %95, <64 x float> %97, <64 x float> %90, <256 x i1> %8)
  %99 = getelementptr float, ptr addrspace(6) %2, i64 %71
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %89, ptr addrspace(6) %99, i32 0, i32 5, i32 0, <256 x i1> %8)
  %100 = getelementptr float, ptr addrspace(6) %3, i64 %71
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %98, ptr addrspace(6) %100, i32 0, i32 5, i32 0, <256 x i1> %8)
  %101 = getelementptr float, ptr addrspace(6) %4, i64 %71
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %90, ptr addrspace(6) %101, i32 0, i32 5, i32 0, <256 x i1> %8)
  %102 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %91, <256 x i1> %8, i32 1, i32 1, i32 0)
  %103 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %92, <256 x i1> %8, i32 1, i32 1, i32 0)
  %104 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %102, <128 x bfloat> %103)
  %105 = extractvalue { <128 x bfloat>, <128 x bfloat> } %104, 0
  %106 = mul i32 %70, 16
  %107 = call <256 x i1> @llvm.hivm.pge.b16(i32 0, i32 0)
  %108 = sext i32 %106 to i64
  %109 = add i64 0, %108
  %110 = getelementptr bfloat, ptr addrspace(6) %5, i64 %109
  call void @llvm.hivm.vsstb.v128bf16(<128 x bfloat> %105, ptr addrspace(6) %110, i32 3211264, i32 0, <256 x i1> %107)
  %111 = add nsw i32 %67, 1
  br label %66

112:                                              ; preds = %66
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_splits_scope_2(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2, ptr addrspace(6) noalias %3, ptr addrspace(6) noalias %4) #0 {
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i32 [ %49, %9 ], [ 0, %5 ]
  %8 = icmp slt i32 %7, 48
  br i1 %8, label %9, label %50

9:                                                ; preds = %6
  %10 = sext i32 %7 to i64
  %11 = mul i64 %10, 128
  %12 = add i64 %11, 0
  %13 = getelementptr float, ptr addrspace(6) %0, i64 %12
  %14 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %13, i32 0, i32 0, i32 0)
  %15 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %16 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %14, float 0x3FA47AE140000000, <256 x i1> %15)
  %17 = mul i64 %10, 128
  %18 = add i64 %17, 64
  %19 = getelementptr float, ptr addrspace(6) %0, i64 %18
  %20 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %19, i32 0, i32 0, i32 0)
  %21 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %20, float 0x3FA47AE140000000, <256 x i1> %15)
  %22 = getelementptr float, ptr addrspace(6) %1, i64 %10
  %23 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %22, i32 0, i32 3, i32 0)
  %24 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %16, <64 x float> %21, <256 x i1> %15)
  %25 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %24, <256 x i1> %15)
  %26 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %25, <256 x i1> %15, i32 1)
  %27 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %26, <64 x float> %23, <256 x i1> %15)
  %28 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %23, <64 x float> %27, <256 x i1> %15, i32 0)
  %29 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %16, <64 x float> %27, <256 x i1> %15, i32 0)
  %30 = call <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float> %21, <64 x float> %27, <256 x i1> %15, i32 0)
  %31 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %29, <64 x float> %30, <256 x i1> %15)
  %32 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %31, <256 x i1> %15)
  %33 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %32, <256 x i1> %15, i32 1)
  %34 = getelementptr float, ptr addrspace(6) %2, i64 %10
  %35 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %34, i32 0, i32 3, i32 0)
  %36 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %33, <64 x float> %35, <64 x float> %28, <256 x i1> %15)
  %37 = getelementptr float, ptr addrspace(6) %1, i64 %10
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %27, ptr addrspace(6) %37, i32 0, i32 5, i32 0, <256 x i1> %15)
  %38 = getelementptr float, ptr addrspace(6) %2, i64 %10
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %36, ptr addrspace(6) %38, i32 0, i32 5, i32 0, <256 x i1> %15)
  %39 = getelementptr float, ptr addrspace(6) %3, i64 %10
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %28, ptr addrspace(6) %39, i32 0, i32 5, i32 0, <256 x i1> %15)
  %40 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %29, <256 x i1> %15, i32 1, i32 1, i32 0)
  %41 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %30, <256 x i1> %15, i32 1, i32 1, i32 0)
  %42 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat> %40, <128 x bfloat> %41)
  %43 = extractvalue { <128 x bfloat>, <128 x bfloat> } %42, 0
  %44 = mul i32 %7, 16
  %45 = call <256 x i1> @llvm.hivm.pge.b16(i32 0, i32 0)
  %46 = sext i32 %44 to i64
  %47 = add i64 0, %46
  %48 = getelementptr bfloat, ptr addrspace(6) %4, i64 %47
  call void @llvm.hivm.vsstb.v128bf16(<128 x bfloat> %43, ptr addrspace(6) %48, i32 3211264, i32 0, <256 x i1> %45)
  %49 = add nsw i32 %7, 1
  br label %6

50:                                               ; preds = %6
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @attention_splits_scope_3(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2) #0 {
  %4 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0xC7EFFFFFE0000000)
  %5 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %4, ptr addrspace(6) %0, i32 0, i32 2, i32 0, <256 x i1> %5)
  %6 = call <64 x float> @llvm.hivm.vbr.v64f32(float 0.000000e+00)
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %1, i32 0, i32 2, i32 0, <256 x i1> %5)
  br label %7

7:                                                ; preds = %10, %3
  %8 = phi i32 [ %36, %10 ], [ 0, %3 ]
  %9 = icmp slt i32 %8, 48
  br i1 %9, label %10, label %37

10:                                               ; preds = %7
  %11 = sext i32 %8 to i64
  %12 = mul i64 %11, 512
  %13 = add i64 %12, 0
  %14 = getelementptr float, ptr addrspace(6) %2, i64 %13
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %14, i32 0, i32 2, i32 0, <256 x i1> %5)
  %15 = mul i64 %11, 512
  %16 = add i64 %15, 64
  %17 = getelementptr float, ptr addrspace(6) %2, i64 %16
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %17, i32 0, i32 2, i32 0, <256 x i1> %5)
  %18 = mul i64 %11, 512
  %19 = add i64 %18, 128
  %20 = getelementptr float, ptr addrspace(6) %2, i64 %19
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %20, i32 0, i32 2, i32 0, <256 x i1> %5)
  %21 = mul i64 %11, 512
  %22 = add i64 %21, 192
  %23 = getelementptr float, ptr addrspace(6) %2, i64 %22
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %23, i32 0, i32 2, i32 0, <256 x i1> %5)
  %24 = mul i64 %11, 512
  %25 = add i64 %24, 256
  %26 = getelementptr float, ptr addrspace(6) %2, i64 %25
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %26, i32 0, i32 2, i32 0, <256 x i1> %5)
  %27 = mul i64 %11, 512
  %28 = add i64 %27, 320
  %29 = getelementptr float, ptr addrspace(6) %2, i64 %28
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %29, i32 0, i32 2, i32 0, <256 x i1> %5)
  %30 = mul i64 %11, 512
  %31 = add i64 %30, 384
  %32 = getelementptr float, ptr addrspace(6) %2, i64 %31
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %32, i32 0, i32 2, i32 0, <256 x i1> %5)
  %33 = mul i64 %11, 512
  %34 = add i64 %33, 448
  %35 = getelementptr float, ptr addrspace(6) %2, i64 %34
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %35, i32 0, i32 2, i32 0, <256 x i1> %5)
  %36 = add nsw i32 %8, 1
  br label %7

37:                                               ; preds = %7
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

define dso_local void @attention_splits_mix_aiv(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, i32 %9, i32 %10, i32 %11) #2 {
  %13 = alloca i64, i64 1, align 8
  store i64 0, ptr %13, align 4
  %14 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %15 = trunc i64 %14 to i32
  %16 = srem i32 %15, %9
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 1, i64 1)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 1, i64 2)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  br label %17

17:                                               ; preds = %86, %12
  %18 = phi i32 [ %93, %86 ], [ %16, %12 ]
  %19 = icmp slt i32 %18, 32
  br i1 %19, label %20, label %94

20:                                               ; preds = %17
  %21 = sdiv i32 %18, 4
  %22 = srem i32 %18, 4
  %23 = mul i32 %22, 280
  %24 = sub i32 1120, %23
  %25 = call i32 @llvm.smin.i32(i32 %24, i32 280)
  %26 = mul i32 %21, 96
  %27 = call i64 @llvm.hivm.GET.SUBBLOCKID()
  %28 = mul i32 %22, 35840
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call ptc_simdvf void @attention_splits_scope_3(ptr addrspace(6) null, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)))
  %29 = sext i32 %25 to i64
  %30 = add i64 %29, 2
  %31 = icmp sgt i64 %29, 0
  %32 = select i1 %31, i64 %30, i64 0
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  br label %33

33:                                               ; preds = %83, %20
  %34 = phi i64 [ %85, %83 ], [ 0, %20 ]
  %35 = icmp slt i64 %34, %32
  br i1 %35, label %36, label %86

36:                                               ; preds = %33
  %37 = sub i64 %34, 1
  %38 = load i64, ptr %13, align 4
  %39 = urem i64 %38, 2
  %40 = icmp eq i64 %39, 1
  %41 = select i1 %40, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 234496 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 234496 to ptr addrspace(6)), i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 98816 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98816 to ptr addrspace(6)), i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }
  %42 = add i64 %39, 1
  %43 = mul i64 %34, 128
  %44 = sext i32 %28 to i64
  %45 = add i64 %44, %43
  %46 = icmp sge i64 %34, 0
  %47 = icmp slt i64 %34, %29
  %48 = and i1 %46, %47
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  br i1 %48, label %49, label %70

49:                                               ; preds = %36
  %50 = sext i32 %21 to i64
  %51 = getelementptr i32, ptr addrspace(1) %5, i64 %50
  %52 = load i32, ptr addrspace(1) %51, align 4
  %53 = add i32 %52, -3
  %54 = sext i32 %53 to i64
  %55 = sub i64 %54, %45
  %56 = trunc i64 %55 to i32
  %57 = mul i64 %27, 2
  %58 = sext i32 %56 to i64
  %59 = add i64 %58, %57
  %60 = icmp slt i64 %59, 128
  %61 = zext i1 %60 to i32
  %62 = icmp eq i32 %61, 0
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 1, i64 3)
  br i1 %62, label %63, label %65

63:                                               ; preds = %49
  %64 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, 1
  call ptc_simdvf void @attention_splits_scope_2(ptr addrspace(6) inttoptr (i64 99072 to ptr addrspace(6)), ptr addrspace(6) null, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) %64, ptr addrspace(6) inttoptr (i64 221952 to ptr addrspace(6)))
  br label %65

65:                                               ; preds = %63, %49
  %66 = icmp eq i32 %61, 1
  br i1 %66, label %67, label %69

67:                                               ; preds = %65
  %68 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, 1
  call ptc_simdvf void @attention_splits_scope_1(i64 %59, ptr addrspace(6) inttoptr (i64 99072 to ptr addrspace(6)), ptr addrspace(6) null, ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) %68, ptr addrspace(6) inttoptr (i64 221952 to ptr addrspace(6)))
  br label %69

69:                                               ; preds = %67, %65
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 1, i64 2)
  br label %70

70:                                               ; preds = %69, %36
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  br i1 %48, label %71, label %76

71:                                               ; preds = %70
  %72 = select i1 %40, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 430080 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 430080 to ptr addrspace(2)), i64 0, [4 x i64] [i64 8, i64 6, i64 16, i64 16], [4 x i64] [i64 1536, i64 256, i64 16, i64 1] }, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 258048 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 258048 to ptr addrspace(2)), i64 0, [4 x i64] [i64 8, i64 6, i64 16, i64 16], [4 x i64] [i64 1536, i64 256, i64 16, i64 1] }
  %73 = mul i64 %27, 768
  %74 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %72, 0
  %75 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %72, 1
  call void @llvm.hivm.WAIT.INTRA.BLOCK.mode(i64 5, i64 %42)
  call void @copy_ubuf_to_cbuf_2d_bfloat16_t(ptr addrspace(6) inttoptr (i64 221952 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 221952 to ptr addrspace(6)), i64 0, i64 8, i64 768, i64 784, i64 1, ptr addrspace(2) %74, ptr addrspace(2) %75, i64 %73, i64 8, i64 768, i64 1536, i64 1)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 5, i64 0)
  br label %76

76:                                               ; preds = %71, %70
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  %77 = icmp sge i64 %37, 0
  %78 = icmp slt i64 %37, %29
  %79 = and i1 %77, %78
  br i1 %79, label %80, label %83

80:                                               ; preds = %76
  %81 = select i1 %40, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 98816 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 98816 to ptr addrspace(6)), i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 234496 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 234496 to ptr addrspace(6)), i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 1, i64 0)
  %82 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %81, 1
  call ptc_simdvf void @attention_splits_scope_0(ptr addrspace(6) %82, ptr addrspace(6) inttoptr (i64 123648 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)))
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 1, i64 1)
  br label %83

83:                                               ; preds = %80, %76
  %84 = add i64 %38, 1
  store i64 %84, ptr %13, align 4
  %85 = add nsw i64 %34, 1
  br label %33

86:                                               ; preds = %33
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %87 = mul i32 %22, 768
  %88 = add i32 %87, %26
  %89 = mul i64 %27, 48
  %90 = sext i32 %88 to i64
  %91 = add i64 %90, %89
  %92 = mul i64 %91, 512
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @store_ubuf_to_gm_2d_float(ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 512 to ptr addrspace(6)), i64 0, i64 48, i64 512, i64 512, i64 1, ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %92, i64 48, i64 512, i64 512, i64 1, i32 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 48, i64 1, ptr addrspace(1) %7, ptr addrspace(1) %7, i64 %91, i64 48, i64 1, i32 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 256 to ptr addrspace(6)), i64 0, i64 48, i64 1, ptr addrspace(1) %8, ptr addrspace(1) %8, i64 %91, i64 48, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  %93 = add nsw i32 %18, 32
  br label %17

94:                                               ; preds = %17
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 5, i64 1)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 5, i64 2)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

; Function Attrs: nounwind readonly 
declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #3

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #4

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float>, <64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vbr.v64f32(float) #5

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32) #5

; Function Attrs: nounwind readonly 
declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32) #3

; Function Attrs: nounwind readnone 
declare <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float>, <256 x i1>, i32) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vexpdif.v64f32(<64 x float>, <64 x float>, <256 x i1>, i32) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32) #5

; Function Attrs: nounwind readnone 
declare { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vdintlv.v128bf16(<128 x bfloat>, <128 x bfloat>) #5

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b16(i32, i32) #4

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vsstb.v128bf16(<128 x bfloat>, ptr addrspace(6) nocapture readonly, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64, i64) #7

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64, i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smin.i32(i32, i32) #8

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.SUBBLOCKID() #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.INTRA.BLOCK.mode(i64, i64) #7

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #3 = { nounwind readonly  }
attributes #4 = { nounwind }
attributes #5 = { nounwind readnone  }
attributes #6 = { nounwind writeonly  }
attributes #7 = { nounwind  inaccessiblememonly }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn readnone  }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @attention_splits_mix_aiv, !"kernel", i32 1}

