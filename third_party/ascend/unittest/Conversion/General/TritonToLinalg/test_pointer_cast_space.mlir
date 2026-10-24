// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// The precise PointerCastOp rewrite must keep the source memory space. A TLA
// pointer entry carries a GM address space on its 1-D pointer_cast; dropping it
// during the rewrite makes the downstream reinterpret_cast compare a plain
// memref<?xf32> against a GM result type and fail verification.

// CHECK-LABEL: func.func @test_gm_pointer_entry
// CHECK: %[[POINTER:.*]] = hivm.hir.pointer_cast({{.*}}) [{{.*}}] : memref<?xf32, #hivm.address_space<gm>>
// CHECK: memref.reinterpret_cast %[[POINTER]] to offset: [0], sizes: [128], strides: [1]
// CHECK-SAME: : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @test_gm_pointer_entry(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %addr = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
    %cast = hivm.hir.pointer_cast(%addr) : memref<1xf32, #hivm.address_space<gm>>
    %view = memref.reinterpret_cast %cast to offset: [%c0], sizes: [128], strides: [1] : memref<1xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%view : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128xf32, #hivm.address_space<ub>>)
    tt.return
  }
}

// -----

// A plain Triton int_to_ptr pointer carries no memory space; the rewrite must
// keep it that way instead of inventing one.

// CHECK-LABEL: func.func @test_plain_int_to_ptr
// CHECK: hivm.hir.pointer_cast({{.*}}) [{{.*}}] : memref<?xi32>

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @test_plain_int_to_ptr(%addr_ptr: !tt.ptr<i64>, %out_ptr: !tt.ptr<i32>) attributes {noinline = false} {
    %raw_addr = tt.load %addr_ptr : !tt.ptr<i64>
    %ptr = tt.int_to_ptr %raw_addr : i64 -> !tt.ptr<i32>
    %ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %load_ptrs = tt.addptr %ptrs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %data = tt.load %load_ptrs : tensor<8x!tt.ptr<i32>>
    %out_ptrs = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out_addptr = tt.addptr %out_ptrs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %out_addptr, %data : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
