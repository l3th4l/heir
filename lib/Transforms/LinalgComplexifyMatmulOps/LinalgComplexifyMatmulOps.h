#ifndef LIB_TRANSFORMS_LINALGCOMPLEXIFYMATMULOPS_LINALGCOMPLEXIFYMATMULOPS_H_
#define LIB_TRANSFORMS_LINALGCOMPLEXIFYMATMULOPS_LINALGCOMPLEXIFYMATMULOPS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

std::unique_ptr<Pass> createLinalgComplexifyMatmulOpsPass();

#define GEN_PASS_DECL
#include "lib/Transforms/LinalgComplexifyMatmulOps/LinalgComplexifyMatmulOps.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LinalgComplexifyMatmulOps/LinalgComplexifyMatmulOps.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LINALGCOMPLEXIFYMATMULOPS_LINALGCOMPLEXIFYMATMULOPS_H_
