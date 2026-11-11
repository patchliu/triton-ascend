; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define dso_local void @attention_mix_aic(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, ptr addrspace(1) %9, ptr addrspace(1) %10, i32 %11, i32 %12, i32 %13) #0 {
  %15 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %16 = trunc i64 %15 to i32
  %17 = srem i32 %16, %11
  %18 = mul i32 %17, 320
  %19 = call i32 @llvm.smin.i32(i32 %18, i32 10240)
  %20 = add i32 %19, 320
  %21 = call i32 @llvm.smin.i32(i32 %20, i32 10240)
  %22 = sdiv i32 %19, 1280
  %23 = icmp slt i32 %19, %21
  %24 = add i32 %21, 1279
  %25 = sdiv i32 %24, 1280
  %26 = select i1 %23, i32 %25, i32 %22
  br label %27

27:                                               ; preds = %181, %14
  %28 = phi i32 [ %182, %181 ], [ %22, %14 ]
  %29 = icmp slt i32 %28, %26
  br i1 %29, label %30, label %183

30:                                               ; preds = %27
  %31 = mul i32 %28, 1280
  %32 = sext i32 %28 to i64
  %33 = getelementptr i32, ptr addrspace(1) %5, i64 %32
  %34 = load i32, ptr addrspace(1) %33, align 4
  %35 = add i32 %34, -1
  %36 = sdiv i32 %35, 128
  %37 = add i32 %34, 111
  %38 = sdiv i32 %37, 112
  %39 = call i32 @llvm.smax.i32(i32 %31, i32 %19)
  %40 = add i32 %31, %38
  %41 = call i32 @llvm.smin.i32(i32 %40, i32 %21)
  %42 = sub i32 %39, %31
  %43 = mul i32 %42, 112
  %44 = sub i32 %41, %39
  %45 = call i32 @llvm.smax.i32(i32 %44, i32 0)
  %46 = mul i32 %28, 55296
  %47 = sext i32 %46 to i64
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %47, i64 96, i64 576, i64 576, i64 1, ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), i64 0, i64 36, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 3, i64 3)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 3, i64 3)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 2)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 3, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 3, i64 1)
  call void @l12l0_cbuf_to_ca_bfloat16_t(ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), i64 0, i64 36, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(3) inttoptr (i64 24576 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 24576 to ptr addrspace(3)), i64 0, i64 6, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 0, i64 96, i64 96, i64 6, i64 96, i64 96)
  %48 = add i32 %45, 1
  br label %49

49:                                               ; preds = %179, %30
  %50 = phi i32 [ %180, %179 ], [ -1, %30 ]
  %51 = icmp slt i32 %50, %48
  br i1 %51, label %52, label %181

52:                                               ; preds = %49
  %53 = icmp sge i32 %50, 0
  %54 = icmp slt i32 %50, %45
  %55 = and i1 %53, %54
  %56 = icmp sgt i32 %50, 0
  %57 = srem i32 %50, 3
  %58 = add i32 %50, 1
  %59 = icmp slt i32 %58, %45
  br i1 %59, label %60, label %101

60:                                               ; preds = %52
  %61 = mul i32 %58, 112
  %62 = add i32 %43, %61
  %63 = sdiv i32 %62, 128
  %64 = srem i32 %62, 128
  %65 = sub i32 128, %64
  %66 = call i32 @llvm.smin.i32(i32 %65, i32 112)
  %67 = srem i32 %58, 3
  %68 = mul i32 %67, 68096
  %69 = sext i32 %68 to i64
  %70 = mul i32 %28, 1120
  %71 = call i32 @llvm.smin.i32(i32 %63, i32 %36)
  %72 = sext i32 %70 to i64
  %73 = sext i32 %71 to i64
  %74 = add i64 %72, %73
  %75 = getelementptr i32, ptr addrspace(1) %4, i64 %74
  %76 = load i32, ptr addrspace(1) %75, align 4
  %77 = sext i32 %76 to i64
  %78 = sext i32 %67 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 4, i64 %78)
  %79 = mul i64 %77, 73728
  %80 = sext i32 %66 to i64
  %81 = sext i32 %64 to i64
  %82 = udiv i64 %80, 16
  %83 = mul i64 %81, 576
  %84 = add i64 %79, %83
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %84, i64 %80, i64 576, i64 576, i64 1, ptr addrspace(2) null, ptr addrspace(2) null, i64 %69, i64 36, i64 %82, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, i32 4)
  %85 = icmp slt i32 %66, 112
  br i1 %85, label %86, label %100

86:                                               ; preds = %60
  %87 = add i32 %63, 1
  %88 = call i32 @llvm.smin.i32(i32 %87, i32 %36)
  %89 = sext i32 %88 to i64
  %90 = add i64 %72, %89
  %91 = getelementptr i32, ptr addrspace(1) %4, i64 %90
  %92 = load i32, ptr addrspace(1) %91, align 4
  %93 = sext i32 %92 to i64
  %94 = mul i64 %93, 73728
  %95 = sub i32 112, %66
  %96 = sext i32 %95 to i64
  %97 = udiv i64 %96, 16
  %98 = mul i64 %82, 256
  %99 = add i64 %69, %98
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %94, i64 %96, i64 576, i64 576, i64 1, ptr addrspace(2) null, ptr addrspace(2) null, i64 %99, i64 36, i64 %97, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, i32 4)
  br label %100

100:                                              ; preds = %86, %60
  call void @llvm.hivm.SET.FLAG.REG(i64 4, i64 3, i64 %78)
  br label %101

101:                                              ; preds = %100, %52
  br i1 %55, label %102, label %137

102:                                              ; preds = %101
  %103 = mul i32 %57, 68096
  %104 = sext i32 %103 to i64
  %105 = srem i32 %50, 2
  %106 = sext i32 %57 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 4, i64 3, i64 %106)
  %107 = sext i32 %105 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 10, i64 2, i64 %107)
  %108 = mul i32 %105, 32768
  %109 = sext i32 %108 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %107)
  %110 = mul i32 %105, 16384
  %111 = sext i32 %110 to i64
  call void @l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr addrspace(2) null, ptr addrspace(2) null, i64 %104, i64 6, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %111, i64 6, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, i64 0, i64 96, i64 96, i64 1, i64 96, i64 112)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %107)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %107)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) inttoptr (i64 24576 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 24576 to ptr addrspace(3)), i64 0, i64 6, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %111, i64 6, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %109, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 96, i64 112, i64 1)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %107)
  %112 = srem i32 %58, 2
  %113 = sext i32 %112 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %113)
  call void @l12l0_cbuf_to_ca_bfloat16_t(ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), i64 9216, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 0, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 0, i64 112, i64 112, i64 1, i64 112, i64 96)
  %114 = mul i32 %112, 16384
  %115 = sext i32 %114 to i64
  %116 = add i64 %104, 10752
  call void @l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr addrspace(2) null, ptr addrspace(2) null, i64 %116, i64 7, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %115, i64 7, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, i64 0, i64 112, i64 112, i64 1, i64 112, i64 112)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %113)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %113)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 0, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %115, i64 7, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %109, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 112, i64 112, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %113)
  %117 = add i32 %50, 2
  %118 = srem i32 %117, 2
  %119 = sext i32 %118 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %119)
  call void @l12l0_cbuf_to_ca_bfloat16_t(ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), i64 19968, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(3) null, ptr addrspace(3) null, i64 0, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 0, i64 128, i64 128, i64 1, i64 128, i64 96)
  %120 = mul i32 %118, 16384
  %121 = sext i32 %120 to i64
  %122 = add i64 %104, 23296
  call void @l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr addrspace(2) null, ptr addrspace(2) null, i64 %122, i64 8, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %121, i64 8, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, i64 0, i64 128, i64 128, i64 1, i64 128, i64 112)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %119)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %119)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) null, ptr addrspace(3) null, i64 0, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %121, i64 8, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %109, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 128, i64 112, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %119)
  %123 = add i32 %50, 3
  %124 = srem i32 %123, 2
  %125 = sext i32 %124 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %125)
  call void @l12l0_cbuf_to_ca_bfloat16_t(ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), i64 32256, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 0, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 0, i64 112, i64 112, i64 1, i64 112, i64 96)
  %126 = mul i32 %124, 16384
  %127 = sext i32 %126 to i64
  %128 = add i64 %104, 37632
  call void @l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr addrspace(2) null, ptr addrspace(2) null, i64 %128, i64 7, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %127, i64 7, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, i64 0, i64 112, i64 112, i64 1, i64 112, i64 112)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %125)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %125)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 0, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %127, i64 7, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %109, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 112, i64 112, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %125)
  %129 = add i32 %50, 4
  %130 = srem i32 %129, 2
  %131 = sext i32 %130 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %131)
  call void @l12l0_cbuf_to_ca_bfloat16_t(ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 408576 to ptr addrspace(2)), i64 43008, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(3) null, ptr addrspace(3) null, i64 0, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 0, i64 128, i64 128, i64 1, i64 128, i64 96)
  %132 = mul i32 %130, 16384
  %133 = sext i32 %132 to i64
  %134 = add i64 %104, 50176
  call void @l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr addrspace(2) null, ptr addrspace(2) null, i64 %134, i64 8, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %133, i64 8, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, i64 0, i64 128, i64 128, i64 1, i64 128, i64 112)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %131)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %131)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) null, ptr addrspace(3) null, i64 0, i64 8, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %133, i64 8, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %109, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 128, i64 112, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %131)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 10, i64 %107)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 10, i64 %107)
  br i1 %56, label %135, label %136

135:                                              ; preds = %102
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 17)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 1)
  br label %136

136:                                              ; preds = %135, %102
  call void @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr addrspace(5) null, ptr addrspace(5) null, i64 %109, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 48, i64 112, i64 128, i64 1, i64 0, float 1.000000e+00, i64 0, i1 false, i1 true, i8 0, i64 0, i8 1)
  call void @llvm.hivm.SET.FLAG.REG(i64 10, i64 2, i64 %107)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 16)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 0)
  br label %137

137:                                              ; preds = %136, %101
  br i1 %56, label %138, label %179

138:                                              ; preds = %137
  %139 = add i32 %50, 2
  %140 = srem i32 %139, 3
  %141 = mul i32 %140, 68096
  %142 = sext i32 %141 to i64
  %143 = add i32 %141, 57344
  %144 = sext i32 %143 to i64
  %145 = add i32 %140, 10
  %146 = sext i32 %145 to i64
  %147 = add i64 %146, 16
  call void @llvm.hivm.WAIT.INTRA.BLOCK.mode(i64 3, i64 %147)
  call void @llvm.hivm.WAIT.INTRA.BLOCK.mode(i64 3, i64 %146)
  %148 = icmp sgt i32 %50, 1
  %149 = or i1 %148, %55
  %150 = zext i1 %149 to i32
  %151 = add i32 %50, %150
  %152 = srem i32 %151, 2
  %153 = zext i1 %55 to i32
  %154 = add i32 %50, %153
  %155 = srem i32 %154, 2
  %156 = sext i32 %155 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 10, i64 2, i64 %156)
  %157 = mul i32 %155, 32768
  %158 = sext i32 %157 to i64
  %159 = srem i32 %152, 2
  %160 = sext i32 %159 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %160)
  call void @l12l0_cbuf_to_ca_bfloat16_t(ptr addrspace(2) null, ptr addrspace(2) null, i64 %144, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 0, i64 7, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 0, i64 112, i64 112, i64 1, i64 112, i64 96)
  %161 = mul i32 %159, 16384
  %162 = sext i32 %161 to i64
  call void @l12l0_cbuf_to_cb_bfloat16_t(ptr addrspace(2) null, ptr addrspace(2) null, i64 %142, i64 16, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %162, i64 4, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, i64 0, i64 64, i64 64, i64 2, i64 64, i64 256)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %160)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %160)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 0, i64 4, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %162, i64 4, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %158, i64 16, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 64, i64 256, i64 1)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %160)
  %163 = add i32 %152, 1
  %164 = srem i32 %163, 2
  %165 = sext i32 %164 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %165)
  %166 = mul i32 %164, 16384
  %167 = sext i32 %166 to i64
  call void @l12l0_cbuf_to_cb_bfloat16_t(ptr addrspace(2) null, ptr addrspace(2) null, i64 %142, i64 16, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %167, i64 3, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, i64 1, i64 64, i64 48, i64 2, i64 48, i64 256)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %165)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %165)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 6144, i64 3, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %167, i64 3, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %158, i64 16, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 48, i64 256, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %165)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 10, i64 %156)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 10, i64 %156)
  br i1 %148, label %168, label %169

168:                                              ; preds = %138
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 21)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 5)
  br label %169

169:                                              ; preds = %168, %138
  call void @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr addrspace(5) null, ptr addrspace(5) null, i64 %158, i64 16, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 6144, i64 48, i64 256, i64 256, i64 1, i64 0, float 1.000000e+00, i64 0, i1 false, i1 true, i8 0, i64 0, i8 1)
  call void @llvm.hivm.SET.FLAG.REG(i64 10, i64 2, i64 %156)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 20)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 4)
  %170 = add i32 %58, %153
  %171 = srem i32 %170, 2
  %172 = sext i32 %171 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 10, i64 2, i64 %172)
  %173 = add i64 %142, 28672
  %174 = mul i32 %171, 32768
  %175 = sext i32 %174 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %160)
  call void @l12l0_cbuf_to_cb_bfloat16_t(ptr addrspace(2) null, ptr addrspace(2) null, i64 %173, i64 16, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %162, i64 4, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, i64 0, i64 64, i64 64, i64 2, i64 64, i64 256)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %160)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %160)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 0, i64 4, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %162, i64 4, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %175, i64 16, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 64, i64 256, i64 1)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %160)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 3, i64 %165)
  call void @l12l0_cbuf_to_cb_bfloat16_t(ptr addrspace(2) null, ptr addrspace(2) null, i64 %173, i64 16, i64 7, i64 16, i64 16, i64 1792, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %167, i64 3, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, i64 1, i64 64, i64 48, i64 2, i64 48, i64 256)
  %176 = sext i32 %140 to i64
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 4, i64 %176)
  call void @llvm.hivm.SET.FLAG.REG(i64 3, i64 2, i64 %165)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 3, i64 2, i64 %165)
  call void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), ptr addrspace(3) inttoptr (i64 43008 to ptr addrspace(3)), i64 6144, i64 3, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(4) null, ptr addrspace(4) null, i64 %167, i64 3, i64 16, i64 16, i64 16, i64 4096, i64 256, i64 16, i64 1, ptr addrspace(5) null, ptr addrspace(5) null, i64 %175, i64 16, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, i64 96, i64 48, i64 256, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 3, i64 %165)
  call void @llvm.hivm.SET.FLAG.REG(i64 2, i64 10, i64 %172)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 2, i64 10, i64 %172)
  br i1 %148, label %177, label %178

177:                                              ; preds = %169
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 25)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 10, i64 9)
  br label %178

178:                                              ; preds = %177, %169
  call void @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(ptr addrspace(5) null, ptr addrspace(5) null, i64 %175, i64 16, i64 6, i64 16, i64 16, i64 1536, i64 256, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 18432, i64 48, i64 256, i64 256, i64 1, i64 0, float 1.000000e+00, i64 0, i1 false, i1 true, i8 0, i64 0, i8 1)
  call void @llvm.hivm.SET.FLAG.REG(i64 10, i64 2, i64 %172)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 24)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 10, i64 8)
  br label %179

179:                                              ; preds = %178, %137
  %180 = add nsw i32 %50, 1
  br label %49

181:                                              ; preds = %49
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 3, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 3, i64 1)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 0, i64 22)
  call void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64 0, i64 6)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 0, i64 23)
  call void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64 0, i64 7)
  %182 = add nsw i32 %28, 1
  br label %27

183:                                              ; preds = %27
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
define private void @l12l0_cbuf_to_ca_bfloat16_t(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(3) %11, ptr addrspace(3) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27) #1 {
  %29 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %0, 0
  %30 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %29, ptr addrspace(2) %1, 1
  %31 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %30, i64 %2, 2
  %32 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %31, i64 %3, 3, 0
  %33 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %32, i64 %7, 4, 0
  %34 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %33, i64 %4, 3, 1
  %35 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %34, i64 %8, 4, 1
  %36 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %35, i64 %5, 3, 2
  %37 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %36, i64 %9, 4, 2
  %38 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %37, i64 %6, 3, 3
  %39 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %38, i64 %10, 4, 3
  %40 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %39, ptr %40, align 8
  %41 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(3) %11, 0
  %42 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %41, ptr addrspace(3) %12, 1
  %43 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %42, i64 %13, 2
  %44 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %43, i64 %14, 3, 0
  %45 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %44, i64 %18, 4, 0
  %46 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %45, i64 %15, 3, 1
  %47 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %46, i64 %19, 4, 1
  %48 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %47, i64 %16, 3, 2
  %49 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %48, i64 %20, 4, 2
  %50 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %49, i64 %17, 3, 3
  %51 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %50, i64 %21, 4, 3
  %52 = alloca { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %51, ptr %52, align 8
  call void @_mlir_ciface_l12l0_cbuf_to_ca_bfloat16_t(ptr %40, ptr %52, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_l12l0_cbuf_to_ca_bfloat16_t(ptr, ptr, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: alwaysinline
define private void @l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(4) %11, ptr addrspace(4) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27) #1 {
  %29 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %0, 0
  %30 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %29, ptr addrspace(2) %1, 1
  %31 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %30, i64 %2, 2
  %32 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %31, i64 %3, 3, 0
  %33 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %32, i64 %7, 4, 0
  %34 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %33, i64 %4, 3, 1
  %35 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %34, i64 %8, 4, 1
  %36 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %35, i64 %5, 3, 2
  %37 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %36, i64 %9, 4, 2
  %38 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %37, i64 %6, 3, 3
  %39 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %38, i64 %10, 4, 3
  %40 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %39, ptr %40, align 8
  %41 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(4) %11, 0
  %42 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %41, ptr addrspace(4) %12, 1
  %43 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %42, i64 %13, 2
  %44 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %43, i64 %14, 3, 0
  %45 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %44, i64 %18, 4, 0
  %46 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %45, i64 %15, 3, 1
  %47 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %46, i64 %19, 4, 1
  %48 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %47, i64 %16, 3, 2
  %49 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %48, i64 %20, 4, 2
  %50 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %49, i64 %17, 3, 3
  %51 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %50, i64 %21, 4, 3
  %52 = alloca { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %51, ptr %52, align 8
  call void @_mlir_ciface_l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr %40, ptr %52, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_l12l0_cbuf_to_cb_bfloat16_t_transpose(ptr, ptr, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: alwaysinline
define private void @mmadl0_a5_bfloat16_t_to_float(ptr addrspace(3) %0, ptr addrspace(3) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(4) %11, ptr addrspace(4) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, ptr addrspace(5) %22, ptr addrspace(5) %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36) #1 {
  %38 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(3) %0, 0
  %39 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %38, ptr addrspace(3) %1, 1
  %40 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %39, i64 %2, 2
  %41 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %40, i64 %3, 3, 0
  %42 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %41, i64 %7, 4, 0
  %43 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %42, i64 %4, 3, 1
  %44 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %43, i64 %8, 4, 1
  %45 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %44, i64 %5, 3, 2
  %46 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %45, i64 %9, 4, 2
  %47 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %46, i64 %6, 3, 3
  %48 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %47, i64 %10, 4, 3
  %49 = alloca { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(3), ptr addrspace(3), i64, [4 x i64], [4 x i64] } %48, ptr %49, align 8
  %50 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(4) %11, 0
  %51 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %50, ptr addrspace(4) %12, 1
  %52 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %51, i64 %13, 2
  %53 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %52, i64 %14, 3, 0
  %54 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %53, i64 %18, 4, 0
  %55 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %54, i64 %15, 3, 1
  %56 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %55, i64 %19, 4, 1
  %57 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %56, i64 %16, 3, 2
  %58 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %57, i64 %20, 4, 2
  %59 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %58, i64 %17, 3, 3
  %60 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %59, i64 %21, 4, 3
  %61 = alloca { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %60, ptr %61, align 8
  %62 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(5) %22, 0
  %63 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %62, ptr addrspace(5) %23, 1
  %64 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %63, i64 %24, 2
  %65 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %64, i64 %25, 3, 0
  %66 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %65, i64 %29, 4, 0
  %67 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %66, i64 %26, 3, 1
  %68 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %67, i64 %30, 4, 1
  %69 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %68, i64 %27, 3, 2
  %70 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %69, i64 %31, 4, 2
  %71 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %70, i64 %28, 3, 3
  %72 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %71, i64 %32, 4, 3
  %73 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %72, ptr %73, align 8
  call void @_mlir_ciface_mmadl0_a5_bfloat16_t_to_float(ptr %49, ptr %61, ptr %73, i64 %33, i64 %34, i64 %35, i64 %36)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_mmadl0_a5_bfloat16_t_to_float(ptr, ptr, ptr, i64, i64, i64, i64) #1

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
define private void @l12l0_cbuf_to_cb_bfloat16_t(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(4) %11, ptr addrspace(4) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27) #1 {
  %29 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %0, 0
  %30 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %29, ptr addrspace(2) %1, 1
  %31 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %30, i64 %2, 2
  %32 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %31, i64 %3, 3, 0
  %33 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %32, i64 %7, 4, 0
  %34 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %33, i64 %4, 3, 1
  %35 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %34, i64 %8, 4, 1
  %36 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %35, i64 %5, 3, 2
  %37 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %36, i64 %9, 4, 2
  %38 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %37, i64 %6, 3, 3
  %39 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %38, i64 %10, 4, 3
  %40 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %39, ptr %40, align 8
  %41 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(4) %11, 0
  %42 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %41, ptr addrspace(4) %12, 1
  %43 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %42, i64 %13, 2
  %44 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %43, i64 %14, 3, 0
  %45 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %44, i64 %18, 4, 0
  %46 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %45, i64 %15, 3, 1
  %47 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %46, i64 %19, 4, 1
  %48 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %47, i64 %16, 3, 2
  %49 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %48, i64 %20, 4, 2
  %50 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %49, i64 %17, 3, 3
  %51 = insertvalue { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %50, i64 %21, 4, 3
  %52 = alloca { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(4), ptr addrspace(4), i64, [4 x i64], [4 x i64] } %51, ptr %52, align 8
  call void @_mlir_ciface_l12l0_cbuf_to_cb_bfloat16_t(ptr %40, ptr %52, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_l12l0_cbuf_to_cb_bfloat16_t(ptr, ptr, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smin.i32(i32, i32) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smax.i32(i32, i32) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.INTRA.BLOCKI.mode(i64, i64) #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.INTRA.BLOCKI.mode(i64, i64) #5

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.INTRA.BLOCK.mode(i64, i64) #5

attributes #0 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #1 = { alwaysinline }
attributes #2 = { nounwind readnone  }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn readnone  }
attributes #4 = { nounwind }
attributes #5 = { nounwind  inaccessiblememonly }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @attention_mix_aic, !"kernel", i32 1}

