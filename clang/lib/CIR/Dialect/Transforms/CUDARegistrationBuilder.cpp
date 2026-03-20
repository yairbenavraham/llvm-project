//====- CUDARegistrationBuilder.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// CUDA Registration infrastructure. Most of these functionalities aim to
// glue host and device execution.
//===----------------------------------------------------------------------===//

#include "CUDARegistrationBuilder.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace cir;

constexpr unsigned cudaFatMagic = 0x466243b1;
constexpr unsigned hipFatMagic = 0x48495046; // "HIPF"

static std::string getCUDAPrefix(clang::ASTContext *astCtx) {
  if (astCtx->getLangOpts().HIP)
    return "hip";
  return "cuda";
}

static std::string addUnderscoredPrefix(llvm::StringRef cudaPrefix,
                                 llvm::StringRef cudaFunctionName) {
  return ("__" + cudaPrefix + cudaFunctionName).str();
}

std::unique_ptr<llvm::MemoryBuffer> CUDARegistrationBuilder::readGPUBinary() {

  if (isHIP)
    assert(!cir::MissingFeatures::hipModuleCtor());
  if (astCtx->getLangOpts().GPURelocatableDeviceCode)
    llvm_unreachable("NYI");

  mlir::Attribute cudaBinaryHandleAttr =
      theModule->getAttr(CIRDialect::getCUDABinaryHandleAttrName());
  if (!cudaBinaryHandleAttr) {
    if (isHIP) {
      assert(!cir::MissingFeatures::hipModuleCtor());
      return {};
    }
    return {};
  }

  std::string cudaGPUBinaryName =
      mlir::cast<CUDABinaryHandleAttr>(cudaBinaryHandleAttr).getName();

  // TODO: MAC OS X needs special care, but we haven't supported that in CIR
  // yet.

  // Read the GPU binary and create a constant array for it.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> cudaGPUBinaryOrErr =
      llvm::MemoryBuffer::getFile(cudaGPUBinaryName);
  if (std::error_code ec = cudaGPUBinaryOrErr.getError()) {
    theModule->emitError("cannot open file: " + cudaGPUBinaryName +
                         ec.message());
    return {};
  }

  std::unique_ptr<llvm::MemoryBuffer> cudaGPUBinary =
      std::move(cudaGPUBinaryOrErr.get());

  return cudaGPUBinary;
}

CUDARegistrationBuilder::CUDARegistrationBuilder(mlir::ModuleOp &module,
                                                 cir::CIRDataLayout &dataLayout,
                                                 cir::CIRBaseBuilderTy &cirBaseBuilder,
                                                clang::ASTContext& astCtx, llvm::StringMap<cir::FuncOp>& kernelMap)
    : theModule(module), theDataLayout(dataLayout), builder(cirBaseBuilder), cudaKernelMap(kernelMap), astCtx(&astCtx), theLoc(module->getLoc())  {

  // The Types
  voidTy = builder.getVoidTy();
  voidPtrTy = builder.getVoidPtrTy();
  voidPtrPtrTy = builder.getPointerTo(voidPtrTy);
  intTy = builder.getSIntNTy(32);
  charTy = cir::IntType::get(module->getContext(),
                             astCtx.getCharWidth(),
                             /*isSigned=*/false);
  builder.setInsertionPointToStart(theModule.getBody());

  isHIP = astCtx.getLangOpts().HIP;
  isCUDA = astCtx.getLangOpts().CUDA;
  fatMagic = isHIP ? hipFatMagic : cudaFatMagic;
  cudaPrefix = getCUDAPrefix(&astCtx);
  
}

void CUDARegistrationBuilder::build() {
  // No need to generate ctors/dtors if there is no GPU binary.
  if (cudaKernelMap.empty())
    return;
  std::unique_ptr<llvm::MemoryBuffer> binary = readGPUBinary();
  if (!binary)
    return;
  buildFatBinGlobals(binary);

  // TODO: kernel and shadow var registration
  assert(!cir::MissingFeatures::globalRegistration());
}

void CUDARegistrationBuilder::buildFatBinGlobals(
    std::unique_ptr<llvm::MemoryBuffer> &cudaGPUBinary) {

  // The section names are different for MAC OS X.
  llvm::StringRef fatbinConstName =
      astCtx->getLangOpts().HIP ? ".hip_fatbin" : ".nv_fatbin";

  llvm::StringRef fatbinSectionName =
      astCtx->getLangOpts().HIP ? ".hipFatBinSegment" : ".nvFatBinSegment";

  // Create a global variable with the contents of GPU binary.
  auto fatbinType = ArrayType::get(theModule.getContext(), charTy,
                                   cudaGPUBinary->getBuffer().size());

  // OG gives an empty name to this global constant,
  // which is not allowed in CIR.
  std::string fatbinStrName = addUnderscoredPrefix(cudaPrefix, "_fatbin_str");
  GlobalOp fatbinStr = GlobalOp::create(
      builder, theLoc, fatbinStrName, fatbinType, /*isConstant=*/true,
      /*linkage=*/cir::GlobalLinkageKind::PrivateLinkage);
  fatbinStr.setAlignment(8);
  fatbinStr.setInitialValueAttr(cir::ConstArrayAttr::get(
      fatbinType, builder.getStringAttr(cudaGPUBinary->getBuffer())));
  assert(!cir::MissingFeatures::opGlobalSection());
  fatbinStr.setPrivate();

  // Create a record FatbinWrapper, pointing to the GPU binary.
  // Record layout:
  //    struct { int magicNum; int version; void *fatbin; void *unused; };
  // This will be initialized in the module ctor below.
  auto fatbinWrapperType =
      RecordType::get(theModule.getContext(),
                      {intTy, intTy, voidPtrTy, voidPtrTy}, /*packed=*/false,
                      /*padded=*/false, RecordType::RecordKind::Struct);

  std::string fatbinWrapperName =
      addUnderscoredPrefix(cudaPrefix, "_fatbin_wrapper");
  fatbinWrapper =
      GlobalOp::create(builder, theLoc, fatbinWrapperName, fatbinWrapperType,
                       /*isConstant=*/true,
                       /*linkage=*/cir::GlobalLinkageKind::PrivateLinkage);
  auto magicInit = IntAttr::get(intTy, fatMagic);
  auto versionInit = IntAttr::get(intTy, 1);
  auto fatbinStrSymbol =
      mlir::FlatSymbolRefAttr::get(fatbinStr.getSymNameAttr());
  auto fatbinInit = GlobalViewAttr::get(voidPtrTy, fatbinStrSymbol);
  auto unusedInit = builder.getConstNullPtrAttr(voidPtrTy);
  fatbinWrapper.setInitialValueAttr(cir::ConstRecordAttr::get(
      fatbinWrapperType,
      mlir::ArrayAttr::get(theModule.getContext(),
                           {magicInit, versionInit, fatbinInit, unusedInit})));

  // GPU fat binary handle is also a global variable in OG.
  std::string gpubinHandleName =
      addUnderscoredPrefix(cudaPrefix, "_gpubin_handle");
  gpuBinHandle = GlobalOp::create(
      builder, theLoc, gpubinHandleName, voidPtrPtrTy,
      /*isConstant=*/false, /*linkage=*/GlobalLinkageKind::InternalLinkage);
  gpuBinHandle.setInitialValueAttr(builder.getConstNullPtrAttr(voidPtrPtrTy));
  gpuBinHandle.setPrivate();
}
