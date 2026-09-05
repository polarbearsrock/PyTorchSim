// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Preserve BF16 memory as i16 bits. No numeric FP16 conversion is performed.
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
Type *bitsType(Type *type) {
  Type *i16 = Type::getInt16Ty(type->getContext());
  if (auto *vector = dyn_cast<VectorType>(type))
    return VectorType::get(i16, vector->getElementCount());
  return i16;
}

struct BF16Memory : PassInfoMixin<BF16Memory> {
  PreservedAnalyses run(Function &function, FunctionAnalysisManager &) {
    SmallVector<Instruction *> pending;
    for (Instruction &instruction : instructions(function))
      pending.push_back(&instruction);
    bool changed = false;
    for (Instruction *instruction : pending) {
      IRBuilder<> builder(instruction);
      if (auto *load = dyn_cast<LoadInst>(instruction)) {
        if (!load->getType()->getScalarType()->isBFloatTy())
          continue;
        auto *bits = builder.CreateAlignedLoad(bitsType(load->getType()), load->getPointerOperand(), load->getAlign(), load->isVolatile());
        bits->setOrdering(load->getOrdering());
        bits->setSyncScopeID(load->getSyncScopeID());
        load->replaceAllUsesWith(builder.CreateBitCast(bits, load->getType()));
      } else if (auto *store = dyn_cast<StoreInst>(instruction)) {
        Value *value = store->getValueOperand();
        if (!value->getType()->getScalarType()->isBFloatTy())
          continue;
        auto *bits = builder.CreateBitCast(value, bitsType(value->getType()));
        auto *replacement = builder.CreateAlignedStore(bits, store->getPointerOperand(), store->getAlign(), store->isVolatile());
        replacement->setOrdering(store->getOrdering());
        replacement->setSyncScopeID(store->getSyncScopeID());
      } else if (auto *intrinsic = dyn_cast<IntrinsicInst>(instruction)) {
        auto id = intrinsic->getIntrinsicID();
        bool load = id == Intrinsic::masked_load || id == Intrinsic::masked_gather;
        bool store = id == Intrinsic::masked_store || id == Intrinsic::masked_scatter;
        if (!load && !store)
          continue;
        Type *valueType = load ? intrinsic->getType() : intrinsic->getArgOperand(0)->getType();
        if (!valueType->getScalarType()->isBFloatTy())
          continue;
        Type *integerType = bitsType(valueType);
        SmallVector<Value *> arguments(intrinsic->args());
        unsigned valueIndex = load ? 3 : 0;
        arguments[valueIndex] = builder.CreateBitCast(arguments[valueIndex], integerType);
        Type *pointerType = arguments[load ? 0 : 1]->getType();
        Function *replacement = Intrinsic::getDeclaration(function.getParent(), id, {integerType, pointerType});
        Value *result = builder.CreateCall(replacement, arguments);
        if (load)
          intrinsic->replaceAllUsesWith(builder.CreateBitCast(result, valueType));
      } else {
        continue;
      }
      instruction->eraseFromParent();
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "PyTorchSimBF16Memory", "0.1", [](PassBuilder &builder) {
    builder.registerPipelineParsingCallback([](StringRef name, FunctionPassManager &manager, ArrayRef<PassBuilder::PipelineElement>) {
      if (name != "pytorchsim-bf16-memory")
        return false;
      manager.addPass(BF16Memory());
      return true;
    });
  }};
}
