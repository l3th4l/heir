#include "lib/Transforms/LinalgComplexifyMatmulOps/LinalgComplexifyMatmulOps.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

// TODO Remove unused includes 

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Traits.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"    // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"    // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
                                                                      //

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LINALGCOMPLEXIFYMATMULOPS
#include "lib/Transforms/LinalgComplexifyMatmulOps/LinalgComplexifyMatmulOps.h.inc"

namespace {

// Transforms (XA)(YB)^T : where X and Y are secret tensors 
// 
// TODO This is structurally similar to (XA)(B^T)(Y^T)
// Find other structural similarities and imilement those in
// this rewrite pattern 


struct ComplexifyTransposedMultiplication: public OpRewritePattern<linalg::MatmulOp>{
    const DataFlowSolver *solver; 

    ComplexifyTransposedMultiplication(
            MLIRContext *context, 
            const DataFlowSolver *solver)
            : mlir::OpRewritePattern<linalg::MatmulOp>(context), 
              solver(solver) {}
    
    LogicalResult matchAndRewrite(linalg::MatmulOp matmul3,
                                  PatternRewriter &rewriter) const override{
        
        // get the two inputs of the outer matmuls : %lhs, %rhs, \equiv (XA), (YB)^T
        Value lhs = matmul3.getDpsInputs()[0]; 
        Value rhs = matmul3.getDpsInputs()[1]; 

        // check that lhs is produced by a matmul
        auto matmulXA = lhs.getDefiningOp<linalg::MatmulOp>();

        if (!matmulXA){
            return failure(); 
        }

        // check that rhs is produced by a matmul and then a transpose 
        auto transposeYB = rhs.getDefiningOp<linalg::TransposeOp>();

        if (!transposeYB){
            return failure(); 
        }

        // check that the input to the transpose comes from a matmul 
        Value yb = transposeYB.getDpsInputs()[0];

        auto matmulYB = yb.getDefiningOp<linalg::MatmulOp>();

        if (!matmulYB){
            return failure(); 
        }

        Value X = matmulXA.getDpsInputs()[0];         
        Value A = matmulXA.getDpsInputs()[1];
        
        Value Y = matmulYB.getDpsInputs()[0];         
        Value B = matmulYB.getDpsInputs()[1];

        // checks that X,Y are secret tensors and A, B aren't 
        
        bool xSecret = isSecret(X, solver); 
        bool ySecret = isSecret(Y, solver);
        
        bool aSecret = isSecret(A, solver); 
        bool bSecret = isSecret(B, solver);

        if (!xSecret || !ySecret || aSecret || bSecret) {
            return failure();
        }

        // TODO replace tensor ops with complexified tensors 

        return failure();

    }
}

// TODO Transforms B(XA)C 

// TODO Keep on Hold until clarification of FF1 and FF2 complexification 
// from the Tricycle paper 

// TODO Transforms AB + CD -> (A + Ci)(B - Di)
//
}

struct LinalgComplexifyMatmulOps
    : public impl::LinalgComplexifyMatmulOpsBase<LinalgComplexifyMatmulOps> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(module))) {
      module->emitOpError() << "Failed to run SecretnessAnalysis.\n";
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    // add the rewrite pattern here 
    patterns.add<FuseScaleIntoLinearOp, FuseDivIntoLinearOp,
                 FuseAddIntoLinearOp, FuseSubIntoLinearOp>(context, solver);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createLinalgComplexifyMatmulOpsPass() {
  return std::make_unique<LinalgComplexifyMatmulOps>();
}

}  // namespace heir
}  // namespace mlir
