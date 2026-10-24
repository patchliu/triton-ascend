// RUN: triton-opt %s -split-input-file -inline | FileCheck %s

// Checks that the standard func inliner interface is registered so that
// func-inlining of func.func does not abort on DialectInlinerInterface.

// CHECK-LABEL: inline_simple
// CHECK: func.func @inline_simple
// CHECK-NOT: func.call @callee
// CHECK: arith.addi
func.func private @callee(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

func.func @inline_simple(%arg0: i32, %arg1: i32) -> i32 {
  %0 = func.call @callee(%arg0, %arg1) : (i32, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: inline_nested
// CHECK: func.func @inline_nested
// CHECK-NOT: func.call @leaf
// CHECK: arith.muli
func.func private @leaf(%arg0: i32) -> i32 {
  %0 = arith.muli %arg0, %arg0 : i32
  return %0 : i32
}

func.func private @mid(%arg0: i32) -> i32 {
  %0 = func.call @leaf(%arg0) : (i32) -> i32
  return %0 : i32
}

func.func @inline_nested(%arg0: i32) -> i32 {
  %0 = func.call @mid(%arg0) : (i32) -> i32
  return %0 : i32
}
