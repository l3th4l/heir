#include "lib/Transforms/LinalgComplexifyMatmulOps/LinalgComplexifyMatmulOps.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

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

// Transforms (XA)(XB)^T 
struct ComplexifyTransposedMultiplication: public OpRewritePattern<linalg::MatmulOp>{
    const DataFlowSolver *solver; 

    ComplexifyTransposedMultiplication(
            MLIRContext *context, 
            const DataFlowSolver * solver): mlir::OpRewritePattern<linalg::MatmulOp>

}

// Transforms B(XA)C 



// TODO Keep on Hold until clarification of FF1 and FF2 complexification 
// from the Tricycle paper 

// Transforms AB + CD -> (A + Ci)(B - Di) : Extract the real part 
struct ComplexifyProductProductAddition : public OpRewritePattern<arith::AddFOp>{
    using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

    const DataFlowSolver *solver; 
    
    ComplexifyProductProductAddition(
        MLIRContext *context,
        const DataFlowSolver *solver)
        : OpRewritePattern<arith::AddFOp>(context),
          solver(solver) {}

    LogicalResult matchAndRewrite(
            arith::AddFOp add, 
            PatternRewriter &rewriter) const override{

        // Checks that AB and CD are produced by matmuls
        auto matmulAB = add.getLhs().getDefiningOp<linalg::MatmulOp>(); 
        auto matmulCD = add.getRhs().getDefiningOp<linalg::MatmulOp>(); 

        if (!matmulAB || !matmulCD)
            return failure();

        // TODO Maybe we need additional checks 
        Value A = matmulAB.getDpsInputs()[0]; 
        Value B = matmulAB.getDpsInputs()[1]; 

        Value C = matmulCD.getDpsInputs()[0]; 
        Value D = matmulCD.getDpsInputs()[1];

        if (!solver) 
        {
            return failure();
        }

        // TODO Query : Is this only done if A and C are secret tensors? 
        // If so, then is there a better way to do the complefication than
        // to multiply C with i 

        bool aSecret = isSecret(A, solver); 
        bool cSecret = isSecret(C, solver); 

        if (!aSecret || !cSecret)
        {
            return failure(); 
        }

        // TODO Rewrite tensor ops
        // multiply C with i * I whhere I is an nxn identity matrix where n 
        // is the number of rows of C 
        //
        // Do the same with D
        //

        return failure();

        

    }

}


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
