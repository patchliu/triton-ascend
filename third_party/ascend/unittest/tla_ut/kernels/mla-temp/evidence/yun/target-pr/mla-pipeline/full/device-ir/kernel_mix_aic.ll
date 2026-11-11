; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define dso_local void @attention_splits_mix_aic(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, i32 %9, i32 %10, i32 %11) #0 {
  %13 = alloca i64, i64 1, align 8
  store i64 0, ptr %13, align 4
  %14 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %15 = trunc i64 %14 to i32
  %16 = srem i32 %15, %9
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 3, i64 17)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 3, i64 1)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 3, i64 18)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 3, i64 2)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 3, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 3, i64 1)
  br label %17

17:                                               ; preds = %112, %12
  %18 = phi i32 [ %113, %112 ], [ %16, %12 ]
  %19 = icmp slt i32 %18, 56
  br i1 %19, label %20, label %114

20:                                               ; preds = %17
  %21 = sdiv i32 %18, 7
  %22 = srem i32 %18, 7
  %23 = mul i32 %22, 160
  %24 = sub i32 1120, %23
  %25 = call i32 @llvm.smin.i32(i32 %24, i32 160)
  %26 = mul i32 %21, 55296
  %27 = sext i32 %26 to i64
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 0)
  call void @llvm.hivm.BARRIER(i64 4)
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %27, i64 96, i64 576, i64 576, i64 1, ptr addrspace(2) null, ptr addrspace(2) null, i64 0, i64 36, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i32 0)
  %28 = sext i32 %25 to i64
  %29 = add i64 %28, 2
  %30 = icmp sgt i64 %28, 0
  %31 = select i1 %30, i64 %29, i64 0
  br label %32

32:                                               ; preds = %109, %20
  %33 = phi i64 [ %111, %109 ], [ 0, %20 ]
  %34 = icmp slt i64 %33, %31
  br i1 %34, label %35, label %112

35:                                               ; preds = %32
  %36 = sub i64 %33, 1
  %37 = load i64, ptr %13, align 4
  %38 = urem i64 %37, 2
  %39 = icmp eq i64 %38, 1
  %40 = select i1 %39, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 282624 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 282624 to ptr addrspace(2)), i64 0, [4 x i64] [i64 36, i64 8, i64 16, i64 16], [4 x i64] [i64 2048, i64 256, i64 16, i64 1] }, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 110592 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 110592 to ptr addrspace(2)), i64 0, [4 x i64] [i64 36, i64 8, i64 16, i64 16], [4 x i64] [i64 2048, i64 256, i64 16, i64 1] }
  %41 = add i64 %38, 1
  %42 = srem i64 %41, 2
  %43 = add i64 %42, 1
  %44 = sext i32 %23 to i64
  %45 = add i64 %44, %33
  %46 = icmp sge i64 %33, 0
  %47 = icmp slt i64 %33, %28
  %48 = and i1 %46, %47
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 1)
  br i1 %48, label %49, label %75

49:                                               ; preds = %35
  %50 = sext i32 %21 to i64
  %51 = getelementptr i32, ptr addrspace(1) %5, i64 %50
  %52 = load i32, ptr addrspace(1) %51, align 4
  %53 = add i32 %52, -1
  %54 = sdiv i32 %53, 128
  %55 = mul i32 %21, 1120
  %56 = sext i32 %54 to i64
  %57 = call i64 @llvm.smin.i64(i64 %45, i64 %56)
  %58 = sext i32 %55 to i64
  %59 = add i64 %58, %57
  %60 = getelementptr i32, ptr addrspace(1) %4, i64 %59
  %61 = load i32, ptr addrspace(1) %60, align 4
  %62 = sext i32 %61 to i64
  %63 = mul i64 %62, 73728
  %64 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 0
  %65 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 1
  %66 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 2
  %67 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 0
  %68 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 1
  %69 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 2
  %70 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 3
  %71 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 0
  %72 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 1
  %73 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 2
  %74 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 3
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %63, i64 128, i64 576, i64 576, i64 1, ptr addrspace(2) %64, ptr addrspace(2) %65, i64 %66, i64 %67, i64 %68, i64 %69, i64 %70, i64 %71, i64 %72, i64 %73, i64 %74, i32 4)
  br label %75

75:                                               ; preds = %49, %35
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 3, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 3, i64 0)
  br i1 %48, label %76, label %88

76:                                               ; preds = %75
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 0)
  %77 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 0
  %78 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 1
  %79 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 2
  %80 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 0
  %81 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 1
  %82 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 2
  %83 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 3, 3
  %84 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 0
  %85 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 1
  %86 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 2
  %87 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %40, 4, 3
  call void @mma_tile_bfloat16_t_to_float_tb(ptr addrspace(2) null, ptr addrspace(2) null, i64 0, i64 36, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(2) %77, ptr addrspace(2) %78, i64 %79, i64 %80, i64 %81, i64 %82, i64 %83, i64 %84, i64 %85, i64 %86, i64 %87, i1 true, i64 96, i64 576, i64 128, ptr addrspace(5) null, ptr addrspace(5) null, i64 0, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i8 0, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 10, i64 0)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 18)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 10, i64 0)
  call void @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr addrspace(5) null, ptr addrspace(5) null, i64 0, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 99072 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 99072 to ptr addrspace(6)), i64 0, i64 48, i64 128, i64 128, i64 1, i64 0, float 1.000000e+00, i64 0, i1 false, i1 true, i8 0, i64 0, i8 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 19)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 3)
  br label %88

88:                                               ; preds = %76, %75
  %89 = icmp sge i64 %36, 0
  %90 = icmp slt i64 %36, %28
  %91 = and i1 %89, %90
  br i1 %91, label %92, label %109

92:                                               ; preds = %88
  %93 = select i1 %39, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 258048 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 258048 to ptr addrspace(2)), i64 0, [4 x i64] [i64 8, i64 6, i64 16, i64 16], [4 x i64] [i64 1536, i64 256, i64 16, i64 1] }, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 430080 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 430080 to ptr addrspace(2)), i64 0, [4 x i64] [i64 8, i64 6, i64 16, i64 16], [4 x i64] [i64 1536, i64 256, i64 16, i64 1] }
  %94 = select i1 %39, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 110592 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 110592 to ptr addrspace(2)), i64 0, [4 x i64] [i64 36, i64 8, i64 16, i64 16], [4 x i64] [i64 2048, i64 256, i64 16, i64 1] }, { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } { ptr addrspace(2) inttoptr (i64 282624 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 282624 to ptr addrspace(2)), i64 0, [4 x i64] [i64 36, i64 8, i64 16, i64 16], [4 x i64] [i64 2048, i64 256, i64 16, i64 1] }
  %95 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %94, 0
  %96 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %94, 1
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 3, i64 16)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 3, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 1)
  %97 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 0
  %98 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 1
  %99 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 2
  %100 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 3, 0
  %101 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 3, 1
  %102 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 3, 2
  %103 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 3, 3
  %104 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 4, 0
  %105 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 4, 1
  %106 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 4, 2
  %107 = extractvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %93, 4, 3
  call void @mma_tile_bfloat16_t_to_float(ptr addrspace(2) %97, ptr addrspace(2) %98, i64 %99, i64 %100, i64 %101, i64 %102, i64 %103, i64 %104, i64 %105, i64 %106, i64 %107, ptr addrspace(2) %95, ptr addrspace(2) %96, i64 0, i64 32, i64 8, i64 16, i64 16, i64 2048, i64 256, i64 16, i64 1, i1 true, i64 96, i64 128, i64 512, ptr addrspace(5) inttoptr (i64 49152 to ptr addrspace(5)), ptr addrspace(5) inttoptr (i64 49152 to ptr addrspace(5)), i64 0, i64 32, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i8 0, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 10, i64 0)
  %108 = add i64 %43, 16
  call void @llvm.hivm.SET.INTRA.BLOCK.mode(i64 3, i64 %108)
  call void @llvm.hivm.SET.INTRA.BLOCK.mode(i64 3, i64 %43)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 17)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 10, i64 0)
  call void @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr addrspace(5) inttoptr (i64 49152 to ptr addrspace(5)), ptr addrspace(5) inttoptr (i64 49152 to ptr addrspace(5)), i64 0, i64 32, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 123648 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 123648 to ptr addrspace(6)), i64 0, i64 48, i64 512, i64 512, i64 1, i64 0, float 1.000000e+00, i64 0, i1 false, i1 true, i8 0, i64 0, i8 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 1)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 16)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 0)
  br label %109

109:                                              ; preds = %92, %88
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 1)
  %110 = add i64 %37, 1
  store i64 %110, ptr %13, align 4
  %111 = add nsw i64 %33, 1
  br label %32

112:                                              ; preds = %32
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 0)
  %113 = add nsw i32 %18, 28
  br label %17

114:                                              ; preds = %17
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 3, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 3, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 1)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 17)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 1)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 18)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 2)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

; Function Attrs: alwaysinline
define private void @nd2nz_bfloat16_t(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(2) %7, ptr addrspace(2) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i32 %18) #1 {
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %0, 0
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %20, ptr addrspace(1) %1, 1
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %21, i64 %2, 2
  %23 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %22, i64 %3, 3, 0
  %24 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %23, i64 %5, 4, 0
  %25 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %24, i64 %4, 3, 1
  %26 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %25, i64 %6, 4, 1
  %27 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %7, 0
  %29 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %28, ptr addrspace(2) %8, 1
  %30 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %29, i64 %9, 2
  %31 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %30, i64 %10, 3, 0
  %32 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %31, i64 %14, 4, 0
  %33 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %32, i64 %11, 3, 1
  %34 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %33, i64 %15, 4, 1
  %35 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %34, i64 %12, 3, 2
  %36 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %35, i64 %16, 4, 2
  %37 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %36, i64 %13, 3, 3
  %38 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %37, i64 %17, 4, 3
  %39 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %38, ptr %39, align 8
  call void @_mlir_ciface_nd2nz_bfloat16_t(ptr %27, ptr %39, i32 %18)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_nd2nz_bfloat16_t(ptr, ptr, i32) #1

; Function Attrs: alwaysinline
define private void @mma_tile_bfloat16_t_to_float_tb(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(2) %11, ptr addrspace(2) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i1 %22, i64 %23, i64 %24, i64 %25, ptr addrspace(5) %26, ptr addrspace(5) %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44, i64 %45) #1 {
  %47 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %0, 0
  %48 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %47, ptr addrspace(2) %1, 1
  %49 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %48, i64 %2, 2
  %50 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %49, i64 %3, 3, 0
  %51 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %50, i64 %7, 4, 0
  %52 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %51, i64 %4, 3, 1
  %53 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %52, i64 %8, 4, 1
  %54 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %53, i64 %5, 3, 2
  %55 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %54, i64 %9, 4, 2
  %56 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %55, i64 %6, 3, 3
  %57 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %56, i64 %10, 4, 3
  %58 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %57, ptr %58, align 8
  %59 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %11, 0
  %60 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %59, ptr addrspace(2) %12, 1
  %61 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %60, i64 %13, 2
  %62 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %61, i64 %14, 3, 0
  %63 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %62, i64 %18, 4, 0
  %64 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %63, i64 %15, 3, 1
  %65 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %64, i64 %19, 4, 1
  %66 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %65, i64 %16, 3, 2
  %67 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %66, i64 %20, 4, 2
  %68 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %67, i64 %17, 3, 3
  %69 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %68, i64 %21, 4, 3
  %70 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %69, ptr %70, align 8
  %71 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(5) %26, 0
  %72 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %71, ptr addrspace(5) %27, 1
  %73 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %72, i64 %28, 2
  %74 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %73, i64 %29, 3, 0
  %75 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %74, i64 %33, 4, 0
  %76 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %75, i64 %30, 3, 1
  %77 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %76, i64 %34, 4, 1
  %78 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %77, i64 %31, 3, 2
  %79 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %78, i64 %35, 4, 2
  %80 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %79, i64 %32, 3, 3
  %81 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %80, i64 %36, 4, 3
  %82 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %81, ptr %82, align 8
  call void @_mlir_ciface_mma_tile_bfloat16_t_to_float_tb(ptr %58, ptr %70, i1 %22, i64 %23, i64 %24, i64 %25, ptr %82, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44, i64 %45)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_mma_tile_bfloat16_t_to_float_tb(ptr, ptr, i1, i64, i64, i64, ptr, i64, i64, i64, i64, i64, i64, i64, i8, i64) #1

; Function Attrs: alwaysinline
define private void @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr addrspace(5) %0, ptr addrspace(5) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, float %19, i64 %20, i1 %21, i1 %22, i8 %23, i64 %24, i8 %25) #1 {
  %27 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(5) %0, 0
  %28 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %27, ptr addrspace(5) %1, 1
  %29 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %28, i64 %2, 2
  %30 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %29, i64 %3, 3, 0
  %31 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %30, i64 %7, 4, 0
  %32 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %31, i64 %4, 3, 1
  %33 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %32, i64 %8, 4, 1
  %34 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %33, i64 %5, 3, 2
  %35 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %34, i64 %9, 4, 2
  %36 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %35, i64 %6, 3, 3
  %37 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %36, i64 %10, 4, 3
  %38 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %11, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %39, ptr addrspace(6) %12, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %40, i64 %13, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, i64 %14, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %42, i64 %16, 4, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %43, i64 %15, 3, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %44, i64 %17, 4, 1
  %46 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, ptr %46, align 8
  call void @_mlir_ciface_fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr %38, ptr %46, i64 %18, float %19, i64 %20, i1 %21, i1 %22, i8 %23, i64 %24, i8 %25)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr, ptr, i64, float, i64, i1, i1, i8, i64, i8) #1

; Function Attrs: alwaysinline
define private void @mma_tile_bfloat16_t_to_float(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(2) %11, ptr addrspace(2) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i1 %22, i64 %23, i64 %24, i64 %25, ptr addrspace(5) %26, ptr addrspace(5) %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44, i64 %45) #1 {
  %47 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %0, 0
  %48 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %47, ptr addrspace(2) %1, 1
  %49 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %48, i64 %2, 2
  %50 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %49, i64 %3, 3, 0
  %51 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %50, i64 %7, 4, 0
  %52 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %51, i64 %4, 3, 1
  %53 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %52, i64 %8, 4, 1
  %54 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %53, i64 %5, 3, 2
  %55 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %54, i64 %9, 4, 2
  %56 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %55, i64 %6, 3, 3
  %57 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %56, i64 %10, 4, 3
  %58 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %57, ptr %58, align 8
  %59 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %11, 0
  %60 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %59, ptr addrspace(2) %12, 1
  %61 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %60, i64 %13, 2
  %62 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %61, i64 %14, 3, 0
  %63 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %62, i64 %18, 4, 0
  %64 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %63, i64 %15, 3, 1
  %65 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %64, i64 %19, 4, 1
  %66 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %65, i64 %16, 3, 2
  %67 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %66, i64 %20, 4, 2
  %68 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %67, i64 %17, 3, 3
  %69 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %68, i64 %21, 4, 3
  %70 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %69, ptr %70, align 8
  %71 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(5) %26, 0
  %72 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %71, ptr addrspace(5) %27, 1
  %73 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %72, i64 %28, 2
  %74 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %73, i64 %29, 3, 0
  %75 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %74, i64 %33, 4, 0
  %76 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %75, i64 %30, 3, 1
  %77 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %76, i64 %34, 4, 1
  %78 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %77, i64 %31, 3, 2
  %79 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %78, i64 %35, 4, 2
  %80 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %79, i64 %32, 3, 3
  %81 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %80, i64 %36, 4, 3
  %82 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %81, ptr %82, align 8
  call void @_mlir_ciface_mma_tile_bfloat16_t_to_float(ptr %58, ptr %70, i1 %22, i64 %23, i64 %24, i64 %25, ptr %82, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44, i64 %45)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_mma_tile_bfloat16_t_to_float(ptr, ptr, i1, i64, i64, i64, ptr, i64, i64, i64, i64, i64, i64, i64, i8, i64) #1

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smin.i32(i32, i32) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i64 @llvm.smin.i64(i64, i64) #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.INTRA.BLOCK.mode(i64, i64) #3

attributes #0 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #1 = { alwaysinline }
attributes #2 = { nounwind readnone  }
attributes #3 = { nounwind  inaccessiblememonly }
attributes #4 = { nounwind }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn readnone  }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @attention_splits_mix_aic, !"kernel", i32 1}

