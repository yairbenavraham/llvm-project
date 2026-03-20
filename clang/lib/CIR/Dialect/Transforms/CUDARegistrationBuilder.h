//====- LoweringPrepareCXXABI.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// CUDA Registration utilities
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_LOWERING_PREPARE_CUDA_REGISTRATION
#define LLVM_CLANG_LIB_CIR_LOWERING_PREPARE_CUDA_REGISTRATION

#include "mlir/IR/Location.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir {

struct CUDARegistrationBuilder {
public:
  CUDARegistrationBuilder(mlir::ModuleOp &module, cir::CIRDataLayout &dataLayout,
                          cir::CIRBaseBuilderTy &builder, clang::ASTContext& astCtx,
                        llvm::StringMap<cir::FuncOp>& kernelMap);

  mlir::ModuleOp theModule;
  cir::CIRDataLayout theDataLayout;
  cir::CIRBaseBuilderTy builder;
  std::string cudaPrefix;
  llvm::StringMap<cir::FuncOp> cudaKernelMap;
  llvm::StringMap<cir::FuncOp> cudaCachedFunc;
  clang::ASTContext* astCtx;

  // Common types
  cir::VoidType voidTy;
  cir::PointerType voidPtrTy, voidPtrPtrTy;
  cir::IntType intTy;
  mlir::Type charTy;
  mlir::Location theLoc;

  GlobalOp fatbinWrapper, gpuBinHandle;

  unsigned fatMagic;
  bool isHIP, isCUDA;

  void build();
  void buildFatBinGlobals(std::unique_ptr<llvm::MemoryBuffer>& gpuBinary);
  cir::FuncOp getOrCreateRuntimeFunc(llvm::StringRef name, FuncType funcTy);
  std::unique_ptr<llvm::MemoryBuffer> readGPUBinary();
};

} // namespace

#endif