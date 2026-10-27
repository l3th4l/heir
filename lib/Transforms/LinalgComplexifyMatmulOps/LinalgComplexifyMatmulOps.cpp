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
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
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

static FailureOr<Value> packColumns(Location loc, Value input, PatternRewriter &rewriter){
    auto inputType = dyn_cast<RankedTensorType>(input.getType()); 
    
    // TODO Figure out what to do in case of batched matmul 
    if (!inputType || inputType.getRank() != 2 || !inputType.hasStaticShape()){
        return failure();
    }

    auto elemType = dyn_cast<FloatType>(inputType.getElementType());

    if (!elemType){
        return failure(); 
    }

    int64_t rows = inputType.getDimSize(0);
    int64_t cols = inputType.getDimSize(1);

    // ceil(cols/2) : accounts for odd number of columns 
    int64_t packedCols = (cols + 1) / 2; 

    auto complexType = ComplexType::get(elemType);

    auto packedType = RankedTensorType::get({rows, packedCols}, complexType); 

    Value packed = rewriter.create<tensor::GenerateOp>(
            loc, packedType, ValueRange{}, 
            [&](OpBuilder &builder, Location loc, ValueRange indices){

                Value row = indices[0]; 
                Value col = indices[1];

                // real component : input[row][col]
                Value real = builder.create<tensor::ExtractOp>(loc, input, ValueRange{row, col});

                // imagnary component : input[row, col + packedCols]
                Value offset = builder.create<arith::ConstantIndexOp>(loc, packedCols); 

                Value secondCol = builder.create<arith::AddIOp>(loc, col, offset); 

                Value bound = builder.create<arith::ConstantIndexOp>(loc, cols);

                Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, secondCol, bound);

                // Only extract the imaginary component when index in bounds
                auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{elemType}, inBounds, /*withElseRegion=*/true);
                {
                    OpBuilder::InsertionGuard guard(builder); 
                    builder.setInsertionPointToStart(&ifOp.getThenRegion().front()); 

                    Value imag = builder.create<tensor::ExtractOp>(loc, input, ValueRange{row, secondCol});

                    builder.create<scf::YieldOp>(loc, imag);
                }
                {
                    OpBuilder::InsertionGuard guard(builder); 
                    builder.setInsertionPointToStart(&ifOp.getElseRegion().front()); 

                    Value zero = builder.create<arith::ConstantOp>(loc, elemType, builder.getZeroAttr(elemType));

                    builder.create<scf::YieldOp>(loc, zero);
                }

                Value imag = ifOp.getResult(0); 

                Value z = builder.create<complex::CreateOp>(loc, complexType, real, imag); 

                builder.create<tensor::YieldOp>(loc, z);

            });

    return packed;

};

// Transforms (XA)(YB)^T : where X and Y are secret tensors 
// 
// Free Attention Complexification from Tricycle
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

        Location loc = matmul3.getLoc(); 

        FailureOr<Value> packedAResult = packColumns(loc, A, rewriter); 
        FailureOr<Value> packedBResult = packColumns(loc, B, rewriter);  

        if (failed(packedAResult) || failed(packedBResult)) return failure();

        Value Acomp = *packedAResult;
        Value Bcomp = *packedBResult;
        ModuleOp module = matmul3->getParentOfType<ModuleOp>();

        rewriter.replaceOp(matmul3, Acomp);

        llvm::errs() << "Acomp type: " << Acomp.getType() << "\n";
        llvm::errs() << "Bcomp type: " << Bcomp.getType() << "\n\n";

        Acomp.getDefiningOp()->dump();
        Bcomp.getDefiningOp()->dump();

        module.dump();

        return success();

    }
};

// TODO Transforms B(XA)C 

// TODO Keep on Hold until clarification of FF1 and FF2 complexification 
// from the Tricycle paper 

// TODO Transforms AB + CD -> (A + Ci)(B - Di)
}

struct LinalgComplexifyMatmulOps
    : public impl::LinalgComplexifyMatmulOpsBase<LinalgComplexifyMatmulOps> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        scf::SCFDialect,
        tensor::TensorDialect,
        arith::ArithDialect,
        complex::ComplexDialect>();
  }
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
    patterns.add<ComplexifyTransposedMultiplication>(context, &solver);

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
