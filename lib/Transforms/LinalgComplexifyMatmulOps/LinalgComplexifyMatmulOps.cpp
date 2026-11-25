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

// Look up Value / FailureOr<Value>
// Is it better to do this in-line instead of making a seperate function for adding tensors?
// Change to add complex tensors
static FailureOr<Value> addTensors(Location loc, Value input1, Value input2, PatternRewriter &rewriter){
    auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
    auto input2Type = dyn_cast<RankedTensorType>(input2.getType());

    if (!input1Type || !input2Type)
        return failure();

    if (input1Type.getShape() != input2Type.getShape() || input1Type.getElementType() != input2Type.getElementType())
        return failure();

    auto outputType = input2Type; 

    Value init = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType()); 
    int64_t rank = input1Type.getRank(); 
    
    // Look up what indexing maps are
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(rank),
        rewriter.getMultiDimIdentityMap(rank),
        rewriter.getMultiDimIdentityMap(rank)
    };

    // Look up iterator and iteratorTypes
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    auto addOp = rewriter.create<linalg::GenericOp>(loc, outputType, ValueRange{input1, input2}, ValueRange{init}, 
            indexingMaps, iteratorTypes,
            [&](OpBuilder &builder, Location loc, ValueRange args){
                Value sum = builder.create<complex::AddOp>(loc, args[0], args[1]);

                builder.create<linalg::YieldOp>(loc, sum);
            });
    
    return addOp.getResult(0);
}

static FailureOr<Value> conjugateTensor(Location loc, Value input, PatternRewriter &rewriter){
    auto inputType = dyn_cast<RankedTensorType>(input.getType()); // look up dyn_cast / cast? 
    auto outputType = inputType; 
    
    // What does this do? 
    auto result = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());

    //auto zero = rewriter.create<arith::ConstantOp>(loc, outputType, 
    //        DenseElementsAttr::get(outputType, rewriter.getZeroAttr(outputType.getElementType()))); 

    // Using linalg.generic to apply complex.conj element wise. Look up the working of linalg.generic 
    SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(), utils::IteratorType::parallel); // what does this do?

    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputType.getRank()), 
        rewriter.getMultiDimIdentityMap(inputType.getRank())
    }; // check what  this does 
    
    auto generic = rewriter.create<linalg::GenericOp>(loc, outputType, ValueRange{input}, ValueRange{result},
            indexingMaps, iteratorTypes, 
            [&](OpBuilder &builder, Location loc, ValueRange args){
                Value conj = builder.create<complex::ConjOp>(loc, args[0]);
                builder.create<linalg::YieldOp>(loc, conj);
            });

    generic->setAttr("heir.zero_imaginary", rewriter.getUnitAttr());
    return generic.getResult(0);
}

// Convert real-valued tensor to complex tensors with imaginary part being 0 
static FailureOr<Value> convertToComplexTensor(Location loc, Value input, PatternRewriter &rewriter){
    auto inputType = dyn_cast<RankedTensorType>(input.getType()); 

    if (!inputType || !inputType.hasStaticShape())
        return failure();

    auto floatType = dyn_cast<FloatType>(inputType.getElementType());

    if (!floatType)
        return failure();

    auto complexType = ComplexType::get(floatType);

    auto outputType = RankedTensorType::get(inputType.getShape(), complexType);

    Value init = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());

    SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(), utils::IteratorType::parallel);

    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputType.getRank()),
        rewriter.getMultiDimIdentityMap(inputType.getRank())
    };

    auto generic = rewriter.create<linalg::GenericOp>(loc, outputType, ValueRange{input}, ValueRange{init}, 
        indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args){
            Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(floatType));

            Value comp = builder.create<complex::CreateOp>(loc, complexType, args[0], zero);

            builder.create<linalg::YieldOp>(loc, comp);
        });

    return generic.getResult(0); 
}

// Extract the real parts from a complex tensor
// TODO Place the Conjugation Logic here 
static FailureOr<Value> extractRealParts(Location loc, Value input, PatternRewriter &rewriter){

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    
    if (!inputType || !inputType.hasStaticShape())
        return failure();

    auto complexType = dyn_cast<ComplexType>(inputType.getElementType());

    if (!complexType)
        return failure();

    Type floatType = complexType.getElementType();

    auto outputType = RankedTensorType::get(inputType.getShape(), floatType);

    Value init = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), floatType);

    SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(), utils::IteratorType::parallel);

    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputType.getRank()),
        rewriter.getMultiDimIdentityMap(inputType.getRank())
    };

    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{outputType}, ValueRange{input}, ValueRange{init},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
            Value real = builder.create<complex::ReOp>(
                loc, args[0]);

            builder.create<linalg::YieldOp>(loc, real);
        });

    generic->setAttr("heir.extract_real_parts",rewriter.getUnitAttr());
    return generic.getResult(0);
};

static FailureOr<Value> getComplexTensor(Location loc, Value input, PatternRewriter &rewriter) {

    auto inputType = dyn_cast<RankedTensorType>(input.getType());

    if (!inputType || !inputType.hasStaticShape())
        return failure();

    Type elementType = inputType.getElementType();

    // Case 1: Input is already complex.
    if (isa<ComplexType>(elementType)) {
        //TODO Need to check if the imaginary parts for all the elements are 0
        //TODO need to construct hasZeroImaginaryPart
        //if (hasZeroImaginaryPart(input))
        //    return input;

        return failure();
    }

    // Case 2: Input was produced by extractRealParts().
    if (auto generic = input.getDefiningOp<linalg::GenericOp>()) {
        if (generic->hasAttr("heir.extract_real_parts")) {

            Value originalComplex = generic.getDpsInputs()[0];

            return originalComplex;

            // Original complex tensor has nonzero imaginary parts.
            // Treat the extracted real tensor as an ordinary real tensor.
        }
    }

    // Case 3: Ordinary real-valued input.
    if (isa<FloatType>(elementType))
        return convertToComplexTensor(loc, input, rewriter);

    return failure();
};

static FailureOr<Value> packColumns(Location loc, Value input, PatternRewriter &rewriter, 
        int axis = 1, bool negateImag = false){
    auto inputType = dyn_cast<RankedTensorType>(input.getType()); 
    
    // TODO Figure out what to do in case of batched matmul 
    if (!inputType || inputType.getRank() != 2 || !inputType.hasStaticShape() || (axis !=0 && axis != 1))
        return failure();

    auto elemType = dyn_cast<FloatType>(inputType.getElementType());

    if (!elemType)
        return failure();

    int64_t rows = inputType.getDimSize(0);
    int64_t cols = inputType.getDimSize(1);
    
    int64_t packedSize = (inputType.getDimSize(axis) + 1) / 2;

    int64_t outputRows = axis == 0 ? packedSize : rows; 
    int64_t outputCols = axis == 1 ? packedSize : cols;

    int64_t axisSize = inputType.getDimSize(axis);

    // ceil(cols/2) : accounts for odd number of columns 
    // TODO Figure out what to do if the elements are already complex 
    // Possibly add a check to see if the imaginary parts are all 0
   
    auto complexType = ComplexType::get(elemType);

    auto packedType = RankedTensorType::get({outputRows, outputCols}, complexType); 

    Value packed = rewriter.create<tensor::GenerateOp>(
            loc, packedType, ValueRange{}, 
            [&](OpBuilder &builder, Location loc, ValueRange indices){

                Value row = indices[0]; 
                Value col = indices[1];

                // Select the index along the axis being packed 
                Value packedIndex = axis == 0 ? row : col;

                // real component : input[row][col]
                Value real = builder.create<tensor::ExtractOp>(loc, input, ValueRange{row, col});

                // imagnary component 
                Value offset = builder.create<arith::ConstantIndexOp>(loc, packedSize); 

                Value secondIndex = builder.create<arith::AddIOp>(loc, packedIndex, offset); 

                Value bound = builder.create<arith::ConstantIndexOp>(loc, axisSize);

                Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, secondIndex, bound);

                // Only extract the imaginary component when index in bounds, else set the imaginary component to 0 
                auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{elemType}, inBounds, /*withElseRegion=*/true);
                {   
                    Value secondRow = axis == 0 ? secondIndex : row;
                    Value secondCol = axis == 1 ? secondIndex : col;

                    OpBuilder::InsertionGuard guard(builder); 
                    builder.setInsertionPointToStart(&ifOp.getThenRegion().front()); 

                    Value imag = builder.create<tensor::ExtractOp>(loc, input, ValueRange{secondRow, secondCol});

                    builder.create<scf::YieldOp>(loc, imag);
                }
                {
                    OpBuilder::InsertionGuard guard(builder); 
                    builder.setInsertionPointToStart(&ifOp.getElseRegion().front()); 

                    Value zero = builder.create<arith::ConstantOp>(loc, elemType, builder.getZeroAttr(elemType));

                    builder.create<scf::YieldOp>(loc, zero);
                }

                Value imag = ifOp.getResult(0);

                // Optionally, negate the imaginary component
                if (negateImag)
                    imag = builder.create<arith::NegFOp>(loc, imag);

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

        // TODO Convert X and Y to complex tensors

        // checks that X,Y are secret tensors and A, B aren't 
        
        bool xSecret = isSecret(X, solver); 
        bool ySecret = isSecret(Y, solver);
        
        bool aSecret = isSecret(A, solver); 
        bool bSecret = isSecret(B, solver);

        if (!xSecret || !ySecret || aSecret || bSecret) {
            return failure();
        }
        
        Location loc = matmul3.getLoc(); 
        
        // TODO In the final version we need the input secret tensors 
        // to be complex and we need to check that their imaginary parts are 0 
        // not sure if this needs to be done in place or if it's okay to convert
        // to complex later

        FailureOr<Value> xComplexResult = convertToComplexTensor(loc, X, rewriter);
        FailureOr<Value> yComplexResult = convertToComplexTensor(loc, Y, rewriter);

        if (failed(xComplexResult) || failed(yComplexResult))
            return failure();

        Value Xcomp = *xComplexResult;
        Value Ycomp = *yComplexResult;

        FailureOr<Value> packedAResult = packColumns(loc, A, rewriter, 1, false); 
        FailureOr<Value> packedBResult = packColumns(loc, B, rewriter, 1, true);  

        if (failed(packedAResult) || failed(packedBResult)) 
            return failure();

        Value Acomp = *packedAResult;
        Value Bcomp = *packedBResult;

        // rebuild matmul3 with newly constructed tensors (Check later if replacing
        // in place would work) 

        // XA -> XA_comp 

        auto xType = dyn_cast<RankedTensorType>(Xcomp.getType()); // TODO add checks  
        auto aType = dyn_cast<RankedTensorType>(Acomp.getType()); // TODO add checks 

        int64_t rowsXA = xType.getDimSize(0); 
        int64_t colsXA = aType.getDimSize(1); 
        
        auto complexType = dyn_cast<ComplexType>(aType.getElementType()); // TODO add checks 
        auto floatType = complexType.getElementType();

        Value zeroFloat = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(floatType));

        Value zeroComplex = rewriter.create<complex::CreateOp>(
            loc, complexType, zeroFloat, zeroFloat); 

        auto xaType = RankedTensorType::get({rowsXA, colsXA}, complexType); 
        
        Value xaEmpty = rewriter.create<tensor::EmptyOp>(
            loc, xaType.getShape(), xaType.getElementType());

        Value xaInit = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zeroComplex}, ValueRange{xaEmpty}).result();        
       
        auto matmulXAcomp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{Xcomp, Acomp}, ValueRange{xaInit});

        Value XAcomp = matmulXAcomp.getResult(0);

        // YB -> YB_comp 
        
        auto yType = dyn_cast<RankedTensorType>(Ycomp.getType()); // TODO add checks 
        auto bType = dyn_cast<RankedTensorType>(Bcomp.getType()); // TODO add checks 

        int64_t rowsYB = yType.getDimSize(0); 
        int64_t colsYB = bType.getDimSize(1); 
        
        //auto complexType = dyn_cast<ComplexType>(bType.getElementType()); // TODO add checks 

        auto ybType = RankedTensorType::get({rowsYB, colsYB}, complexType); 

        // Value ybInit = rewriter.create<tensor::EmptyOp>(loc, ybType.getShape(), ybType.getElementType());

        Value ybEmpty = rewriter.create<tensor::EmptyOp>(
            loc, ybType.getShape(), ybType.getElementType());
        
        Value ybInit = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zeroComplex}, ValueRange{ybEmpty}).result();

        auto matmulYBcomp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{Ycomp, Bcomp}, ValueRange{ybInit});

        Value YBcomp = matmulYBcomp.getResult(0);

        // YB_comp -> (YB_comp)^T 

        auto ybResultType = dyn_cast<RankedTensorType>(YBcomp.getType()); // TODO add checks 

        int64_t transposedRows = ybResultType.getDimSize(1); 
        int64_t transposedCols = ybResultType.getDimSize(0); 

        auto ybTransposedType = RankedTensorType::get({transposedRows, transposedCols}, ybResultType.getElementType());

        Value transposedInit = rewriter.create<tensor::EmptyOp>(loc, ybTransposedType.getShape(),
                ybTransposedType.getElementType());

        // TODO Check if this works the same way for batched matmul
        auto transposeYBcomp = rewriter.create<linalg::TransposeOp>(loc, YBcomp, transposedInit, ArrayRef<int64_t>{1, 0});

        Value YBcompTranposed = transposeYBcomp.getResult().front();

        // M = XA_comp(YB_comp)^T
        auto xaResultType = dyn_cast<RankedTensorType>(XAcomp.getType()); // TODO add checks
        auto ybtransResultType = dyn_cast<RankedTensorType>(YBcompTranposed.getType()); // TODO add checks

        int64_t outputRows = xaResultType.getDimSize(0); 
        int64_t outputCols = ybtransResultType.getDimSize(1);

        auto outputType = RankedTensorType::get({outputRows, outputCols}, xaResultType.getElementType());
        //Value outputInit = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());
        
        Value outputEmpty = rewriter.create<tensor::EmptyOp>(
            loc, outputType.getShape(), outputType.getElementType());
        
        Value outputInit = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zeroComplex}, ValueRange{outputEmpty}).result();

        auto matmul3comp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{XAcomp, YBcompTranposed}, ValueRange{outputInit});
        Value M = matmul3comp.getResult(0);

        // conj(M)
        // TODO Move the Conjugation Logic to extractRealParts

        FailureOr<Value> conjugateM = conjugateTensor(loc, M, rewriter); 

        if(failed(conjugateM))
            return failure(); 

        Value conjugated = *conjugateM; 

        // (M + conj(M))
        FailureOr<Value> sumResult = addTensors(loc, M, conjugated, rewriter);
        
        if (failed(sumResult))
            return failure();

        Value sum = *sumResult;

        // real(sum)
        FailureOr<Value> realResult = extractRealParts(loc, sum, rewriter);

        if(failed(realResult))
            return failure();

        // TODO we need to propagate the shape and type of the modified operations 
        // since the input and output types and shapes of future operators might 
        // have to be changed. 
        // NOTE : We only need to do this if we yield a complex tensor instead of 
        // a real one
        
        Value result = *realResult; 

        // replace original matmul3 
        rewriter.replaceOp(matmul3, result); 

        return success();

    }
};

// TODO Transforms B(XA)C : Free Attenetion Complexification
// initially we take into account (B(XA))C 
// equivalent of (BX)(AC), B(X(AC)), B((XA)C)
struct ComplexifyRightLeftRightMultiplication: public OpRewritePattern<linalg::MatmulOp>{
    const DataFlowSolver *solver; 

    ComplexifyRightLeftRightMultiplication(
            MLIRContext *context, 
            const DataFlowSolver *solver)
            : mlir::OpRewritePattern<linalg::MatmulOp>(context), 
              solver(solver) {}
    
    LogicalResult matchAndRewrite(linalg::MatmulOp matmul3,
                                  PatternRewriter &rewriter) const override{
        
        // Outer matmul: (B(XA)) * C 
        Value lhs = matmul3.getDpsInputs()[0]; 
        Value C = matmul3.getDpsInputs()[1]; 

        // Middle matmul: B * (XA)
        auto matmul2 = lhs.getDefiningOp<linalg::MatmulOp>();
        if (!matmul2)
            return failure();

        Value B = matmul2.getDpsInputs()[0];
        Value innerRhs = matmul2.getDpsInputs()[1];
        
        // Inner matmul: X * A
        auto matmul1 = innerRhs.getDefiningOp<linalg::MatmulOp>();
        if (!matmul1)
            return failure();

        Value X = matmul1.getDpsInputs()[0];         
        Value A = matmul1.getDpsInputs()[1];

        // checks that X is a ssecret tensors and A, C aren't 
        
        bool xSecret = isSecret(X, solver);
        
        bool aSecret = isSecret(A, solver); 
        bool cSecret = isSecret(C, solver);

        if (!xSecret || aSecret || cSecret)
            return failure();
        
        Location loc = matmul3.getLoc(); 

        FailureOr<Value> xComplexResult = convertToComplexTensor(loc, X, rewriter);
        
        // if B is already complex with imaginary parts 0, we do nothing. If its imaginary parts are 
        // not 0, we return failure. 
        // If B comes from an extractRealParts, which has a complex tensor as operand with imaginary parts all 0, 
        // we remove the extractRealParts operration. If it's imagiinary parts are not 0, then we treat this as a 
        // normal real-only valued tensor 
        // For real-only valued tensor, we use the convertToComplexTensor function
        // This is implemented in the function getComplexTensor()

        FailureOr<Value> bComplexResult = getComplexTensor(loc, B, rewriter);

        if (failed(xComplexResult) || failed(bComplexResult))
            return failure();

        Value Xcomp = *xComplexResult;
        Value Bcomp = *bComplexResult;

        FailureOr<Value> packedAResult = packColumns(loc, A, rewriter, 1, true); 
        FailureOr<Value> packedCResult = packColumns(loc, C, rewriter, 0, false);  

        if (failed(packedAResult) || failed(packedCResult)) 
            return failure();

        Value Acomp = *packedAResult;
        Value Ccomp = *packedCResult;

        // rebuild matmul3 with newly constructed tensors (Check later if replacing
        // in place would work) 

        // XA -> XA_comp 

        auto xType = dyn_cast<RankedTensorType>(Xcomp.getType()); // TODO add checks  
        auto aType = dyn_cast<RankedTensorType>(Acomp.getType()); // TODO add checks 

        int64_t rowsXA = xType.getDimSize(0); 
        int64_t colsXA = aType.getDimSize(1); 
        
        auto complexType = dyn_cast<ComplexType>(aType.getElementType()); // TODO add checks 
        auto floatType = complexType.getElementType();

        Value zeroFloat = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(floatType));

        Value zeroComplex = rewriter.create<complex::CreateOp>(
            loc, complexType, zeroFloat, zeroFloat); 

        auto xaType = RankedTensorType::get({rowsXA, colsXA}, complexType); 
        
        Value xaEmpty = rewriter.create<tensor::EmptyOp>(
            loc, xaType.getShape(), xaType.getElementType());

        Value xaInit = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zeroComplex}, ValueRange{xaEmpty}).result();        
       
        auto matmul1Comp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{Xcomp, Acomp}, ValueRange{xaInit}).getResult(0);
        
        // matmul2Comp = B * matmul1Comp 
        auto bType = dyn_cast<RankedTensorType>(Bcomp.getType()); // TODO add checks 
        auto matmul1CompType = dyn_cast<RankedTensorType>(matmul1Comp.getType()); // TODO add checks. Is this line redundant?  

        int64_t rowsMatmul2 = bType.getDimSize(0);
        int64_t colsMatmul2 = matmul1CompType.getDimSize(1);

        auto matmul2Type = RankedTensorType::get({rowsMatmul2, colsMatmul2}, complexType);

        Value matmul2Empty = rewriter.create<tensor::EmptyOp>(
            loc, matmul2Type.getShape(), matmul2Type.getElementType());

        Value matmul2Init = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zeroComplex}, ValueRange{matmul2Empty}).result();  

        auto matmul2Comp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{Bcomp, matmul1Comp}, ValueRange{matmul2Init}).getResult(0);

        // matmul3Comp = matmul2Comp * C 
        auto matmul2CompType = dyn_cast<RankedTensorType>(matmul2Comp.getType()); // TODO add checks. Is this line redundant?  
        auto cType = dyn_cast<RankedTensorType>(Ccomp.getType()); // TODO add checks 
        
        int64_t rowsMatmul3 = matmul2CompType.getDimSize(0);
        int64_t colsMatmul3 = cType.getDimSize(1);

        auto matmul3Type = RankedTensorType::get({rowsMatmul3, colsMatmul3}, complexType);

        Value matmul3Empty = rewriter.create<tensor::EmptyOp>(
            loc, matmul3Type.getShape(), matmul3Type.getElementType());

        Value matmul3Init = rewriter.create<linalg::FillOp>(
            loc, ValueRange{zeroComplex}, ValueRange{matmul3Empty}).result();  

        auto matmul3Comp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{matmul2Comp, Ccomp}, ValueRange{matmul3Init}).getResult(0);
        Value M = matmul3Comp;

        // conj(M)

        FailureOr<Value> conjugateM = conjugateTensor(loc, M, rewriter); 

        if(failed(conjugateM))
            return failure(); 

        Value conjugated = *conjugateM; 

        // (M + conj(M))
        FailureOr<Value> sumResult = addTensors(loc, M, conjugated, rewriter);
        
        if (failed(sumResult))
            return failure();

        Value sum = *sumResult;

        // real(sum)
        FailureOr<Value> realResult = extractRealParts(loc, sum, rewriter);

        if(failed(realResult))
            return failure();

        // TODO we need to propagate the shape and type of the modified operations 
        // since the input and output types and shapes of future operators might 
        // have to be changed. 
        // NOTE : We only need to do this if we yield a complex tensor instead of 
        // a real one
        
        Value result = *realResult; 

        // replace original matmul3 
        rewriter.replaceOp(matmul3, result); 

        return success();

    }
};

// TODO Transforms AB + CD -> (A + Ci)(B - Di)
// TODO Keep on Hold until clarification of FF1 and FF2 complexification 
// from the Tricycle paper 
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
    patterns.add<ComplexifyRightLeftRightMultiplication>(context, &solver);

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
