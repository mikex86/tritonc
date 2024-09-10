#pragma once

#include <mlir/IR/MLIRContext.h>
#include <string>
#include <vector>
#include "kernel_runner.h"

void
autotune(const std::string &codeTemplate, int computeCapability, int ptxVersion, bool enableFpFusion,
         const std::vector<std::string> &libraryNames, const std::vector<std::string> &libraryPaths,
         const std::string &to_launch_function_name,
         const KernelLaunchBounds &launchBounds,
         std::vector<KernelArgument> kernel_arguments,
         const std::vector<std::pair<std::string, std::vector<int>>> &autotuneConstants,
         const std::vector<std::pair<std::string, std::vector<int>>> &intrinsicAutotuneConstants);