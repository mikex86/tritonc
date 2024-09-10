#pragma once

#include <mlir/IR/MLIRContext.h>
#include <string>

void initLLVM();

mlir::MLIRContext *createContext();

/**
 * @return {ptx, shared_mem_bytes} for the compiled kernel
 */
std::pair<std::string, uint32_t>
compile(mlir::MLIRContext *context, const std::string &contents, int computeCapability, int ptxVersion, int numCTAs,
        int numStages,
        int numWarps, bool enableFpFusion, const std::vector<std::string> &libraryNames,
        const std::vector<std::string> &libraryPaths, const std::string &upToStage);