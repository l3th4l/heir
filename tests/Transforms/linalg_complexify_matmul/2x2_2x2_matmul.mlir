
// RUN: heir-opt %s --linalg-complexify-matmul-ops | \
// RUN: mlir-opt --one-shot-bufferize="bufferize-function-boundaries" \
// RUN: --convert-linalg-to-loops \
// RUN: --convert-complex-to-standard \
// RUN: --convert-complex-to-llvm \
// RUN: --convert-scf-to-cf \
// RUN: --expand-strided-metadata \
// RUN: --lower-affine \
// RUN: --convert-to-llvm \
// RUN: --reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils.so,%mlir_lib_dir/libmlir_runner_utils.so \
// RUN: | FileCheck %s

module {
  func.func @complexify_transposed(
      %X: tensor<2x2xf32> {secret.secret},
      %A: tensor<2x2xf32>,
      %Y: tensor<2x2xf32> {secret.secret},
      %B: tensor<2x2xf32>) -> tensor<2x2xf32> {

    // XA
    %init_xa = arith.constant dense<0.0> : tensor<2x2xf32>
    %xa = linalg.matmul
        ins(%X, %A : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%init_xa : tensor<2x2xf32>)
        -> tensor<2x2xf32>

    // YB
    %init_yb = arith.constant dense<0.0> : tensor<2x2xf32>
    %yb = linalg.matmul
        ins(%Y, %B : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%init_yb : tensor<2x2xf32>)
        -> tensor<2x2xf32>

    // Transpose YB
    %init_transpose = arith.constant dense<0.0> : tensor<2x2xf32>
    %transposed = linalg.transpose
        ins(%yb : tensor<2x2xf32>)
        outs(%init_transpose : tensor<2x2xf32>)
        permutation = [1, 0]

    // Outer matmul
    %init_result = arith.constant dense<0.0> : tensor<2x2xf32>
    %result = linalg.matmul
        ins(%xa, %transposed : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%init_result : tensor<2x2xf32>)
        -> tensor<2x2xf32>

    return %result : tensor<2x2xf32>
  }

  func.func @complexify_right_left_right(
    %B: tensor<2x2xf32>,
    %X: tensor<2x2xf32> {secret.secret},
    %A: tensor<2x2xf32>,
    %C: tensor<2x2xf32>) -> tensor<2x2xf32> {

    // XA
    %init_xa = arith.constant dense<0.0> : tensor<2x2xf32>
    %xa = linalg.matmul
        ins(%X, %A : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%init_xa : tensor<2x2xf32>)
        -> tensor<2x2xf32>

    // B(XA)
    %init_bxa = arith.constant dense<0.0> : tensor<2x2xf32>
    %bxa = linalg.matmul
        ins(%B, %xa : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%init_bxa : tensor<2x2xf32>)
        -> tensor<2x2xf32>

    // B(XA)C
    %init_result = arith.constant dense<0.0> : tensor<2x2xf32>
    %result = linalg.matmul
        ins(%bxa, %C : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%init_result : tensor<2x2xf32>)
        -> tensor<2x2xf32>

    return %result : tensor<2x2xf32>
  }

  func.func @main() {
      // Test 1: (XA)(YB)^T
      %X = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
          : tensor<2x2xf32>
      %A = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]>
          : tensor<2x2xf32>
  
      %Y = arith.constant dense<[[9.0, 10.0], [11.0, 12.0]]>
          : tensor<2x2xf32>
      %B = arith.constant dense<[[13.0, 14.0], [15.0, 16.0]]>
          : tensor<2x2xf32>
  
      %result = func.call @complexify_transposed(%X, %A, %Y, %B)
          : (tensor<2x2xf32>, tensor<2x2xf32>,
             tensor<2x2xf32>, tensor<2x2xf32>)
            -> tensor<2x2xf32>
  
      %unranked = tensor.cast %result
          : tensor<2x2xf32> to tensor<*xf32>
  
      func.call @printMemrefF32(%unranked)
          : (tensor<*xf32>) -> ()
  
      // CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
      // CHECK-SAME: rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
      // CHECK-NEXT: [22730,   27498]
      // CHECK-NEXT: [51562,   62378]
  
  
      // Test 2: B(XA)C
      %B2 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
          : tensor<2x2xf32>
      %X2 = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]>
          : tensor<2x2xf32>
      %A2 = arith.constant dense<[[9.0, 10.0], [11.0, 12.0]]>
          : tensor<2x2xf32>
      %C2 = arith.constant dense<[[13.0, 14.0], [15.0, 16.0]]>
          : tensor<2x2xf32>
  
      %result2 = func.call @complexify_right_left_right(%B2, %X2, %A2, %C2)
          : (tensor<2x2xf32>, tensor<2x2xf32>,
             tensor<2x2xf32>, tensor<2x2xf32>)
            -> tensor<2x2xf32>
  
      %unranked2 = tensor.cast %result2
          : tensor<2x2xf32> to tensor<*xf32>
  
      func.call @printMemrefF32(%unranked2)
          : (tensor<*xf32>) -> ()
  
      // CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
      // CHECK-SAME: rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
      // CHECK-NEXT: [24358,   26092]
      // CHECK-NEXT: [55262,   59196]
  
      return
  }

  func.func private @printMemrefF32(tensor<*xf32>)
}

