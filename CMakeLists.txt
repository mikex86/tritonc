cmake_minimum_required(VERSION 3.28)

project(tritonc CXX)

set(CMAKE_CXX_STANDARD 17)

set(TRITON_CODEGEN_BACKENDS "nvidia" CACHE STRING "Enable different codegen backends" FORCE)
set(TRITON_BUILD_PYTHON_MODULE ON)
set(TRITON_BUILD_PROTON OFF)
set(PYTHON_INCLUDE_DIRS ON)

add_subdirectory(third_party/argparse)
add_subdirectory(third_party/triton)

add_executable(tritonc src/main.cpp)
target_link_libraries(tritonc PRIVATE argparse)

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

set(TRITON_LIBRARIES
        ${triton_libs}

        # mlir
        MLIRAMDGPUDialect
        MLIRNVVMDialect
        MLIRNVVMToLLVMIRTranslation
        MLIRGPUToNVVMTransforms
        MLIRGPUToGPURuntimeTransforms
        MLIRGPUTransforms
        MLIRIR
        MLIRControlFlowToLLVM
        MLIRBytecodeWriter
        MLIRPass
        MLIRTransforms
        MLIRLLVMDialect
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRMathToLLVM
        MLIRROCDLToLLVMIRTranslation
        MLIRGPUDialect
        MLIRSCFToControlFlow
        MLIRIndexToLLVM
        MLIRGPUToROCDLTransforms

        # LLVM
        LLVMPasses
        LLVMNVPTXCodeGen
        # LLVMNVPTXAsmPrinter
        LLVMAMDGPUCodeGen
        LLVMAMDGPUAsmParser

        # Nvidia specific
        TritonNVIDIAGPUToLLVM NVGPUToLLVM MLIRNVGPUToNVVM
        MLIRNVVMDialect MLIRNVGPUDialect

)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR # Linux arm64
        CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")     # macOS arm64
    list(APPEND TRITON_LIBRARIES
            LLVMAArch64CodeGen
            LLVMAArch64AsmParser
    )
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    list(APPEND TRITON_LIBRARIES
            LLVMX86CodeGen
            LLVMX86AsmParser
    )
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
    list(APPEND TRITON_LIBRARIES
            LLVMPowerPCAsmParser
            LLVMPowerPCCodeGen
    )
else()
    message(FATAL_ERROR "LLVM codegen/ASM parser libs: This HW architecture (${CMAKE_SYSTEM_PROCESSOR}) is not configured in cmake lib dependencies.")
endif()

target_link_libraries(tritonc PRIVATE ${TRITON_LIBRARIES})

# triton doesn't use targets in the best practise way... (sigh)
# hence we have to do this...
set(TRITON_INCLUDE_DIRS
        "third_party/triton/include"
        "third_party/triton/lib"
        "third_party/triton/third_party/nvidia/include"
)
set(TRITON_GENERATED_INCLUDE_DIRS
        "${CMAKE_BINARY_DIR}/third_party/triton/include"
        "${CMAKE_BINARY_DIR}/third_party/triton/third_party"
)

target_include_directories(tritonc PRIVATE ${TRITON_INCLUDE_DIRS})
target_include_directories(tritonc PRIVATE ${TRITON_GENERATED_INCLUDE_DIRS})