#include "kernel_runner.h"
#include "templating.h"
#include "tinyscript.h"

#include <cuda.h>
#include <curand.h>
#include <cassert>
#include <iostream>
#include <thread>

inline void cudaCheck(CUresult error, const char *file, int line) {
    if (error != CUDA_SUCCESS) {
        const char *error_string;
        cuGetErrorString(error, &error_string);
        printf("[CUDA ERROR] at file %s:%d: %s\n", file, line, error_string);
        exit(EXIT_FAILURE);
    }
};

inline void cudaCheck(curandStatus error, const char *file, int line) {
    if (error != CURAND_STATUS_SUCCESS) {
        printf("[CUDA ERROR] at file %s:%d: %s\n", file, line, std::to_string(error).c_str());
        exit(EXIT_FAILURE);
    }
};
#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

float benchmarkKernel(CompiledKernel &kernel,
                      const std::string &to_launch_function_name,
                      const KernelLaunchBounds &launchBounds,
                      const std::vector<KernelArgument> &arguments) {
    static bool cuda_init = false;
    static CUdevice device;
    static CUcontext ctx;
    static curandGenerator_t gen;
    if (!cuda_init) {
        // TODO: this is all pretty fragile... make more configurable and robust
        CUDA_CHECK(cuInit(0));
        CUDA_CHECK(cuDeviceGet(&device, 0));
        CUDA_CHECK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, device));
        CUDA_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        cuda_init = true;
    }

    CUmodule module;
    if (cuModuleLoadData(&module, kernel.ptx.data()) != CUDA_SUCCESS) {
        return -1.0f; // invalid kernel
    }

    CUfunction function{};
    if (cuModuleGetFunction(&function, module, to_launch_function_name.c_str()) != CUDA_SUCCESS) {
        std::cerr << "[KernelRunner] Kernel function not found: \"" << to_launch_function_name
                  << "\". Please check your #pragma launch usages." << std::endl;
    }

    if (kernel.shared_mem_bytes > 49152) {
        int shared_optin{};
        CUDA_CHECK(cuDeviceGetAttribute(&shared_optin,
                                        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                        device));
        if (shared_optin >= 49152) {
            CUDA_CHECK(cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED));

            int shared_total{}, shared_static{};
            CUDA_CHECK(cuDeviceGetAttribute(&shared_total,
                                            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));

            CUDA_CHECK(cuFuncGetAttribute(
                    &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));

            CUDA_CHECK(cuFuncSetAttribute(function,
                                          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          shared_optin - shared_static));
        }
    }

    std::vector<void *> kernel_params{};
    std::vector<CUdeviceptr> to_free{};
    std::vector<std::pair<CUdeviceptr, size_t>> to_clear{};

    // build kernel arguments
    for (auto &argument: arguments) {
        void *argument_memory;
        switch (argument.data_type) {
            case PTR:
            case I64:
                argument_memory = malloc(8);
                break;
            case I32:
                argument_memory = malloc(4);
                break;
        }
        if (argument.is_malloc) {
            assert(argument.data_type == PTR);
            auto size_expr = argument.init_expression;
            size_expr = applyTemplate(size_expr, kernel.autotune_values);
            size_t n_bytes = tinyScriptEval(size_expr);
            CUdeviceptr ptr{};
            CUDA_CHECK(cuMemAlloc_v2(&ptr, n_bytes));

            if (argument.is_dst_ptr) {
                to_clear.emplace_back(ptr, n_bytes);
            } else {
                // fill with random values to avoid making matmuls faster due to too many zero bits
                // (yes that's a thing)
                CUDA_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

                // yes, we mis-interpret as floats. We don't care too much about the bit pattern
                CUDA_CHECK(curandGenerateNormal(gen, reinterpret_cast<float *>(ptr), n_bytes / sizeof(float), 0.0f, 1.0f));
                CUDA_CHECK(cuCtxSynchronize());
            }
            to_free.push_back(ptr);
            *((uint64_t *) argument_memory) = ptr;
        } else {
            // normal initializer
            std::string expression = applyTemplate(argument.init_expression, kernel.autotune_values);
            uint64_t value = tinyScriptEval(expression);
            switch (argument.data_type) {
                case PTR:
                case I64:
                    *((uint64_t *) argument_memory) = value;
                    break;
                case I32:
                    *((uint32_t *) argument_memory) = value;
                    break;
            }
        }
        kernel_params.push_back(argument_memory);
    }
    int gridX = 1, gridY = 1, gridZ = 1;
    if (!launchBounds.grid_x_expression.empty()) {
        std::string expression = applyTemplate(launchBounds.grid_x_expression, kernel.autotune_values);
        gridX = static_cast<int>(tinyScriptEval(expression));
    }
    if (!launchBounds.grid_y_expression.empty()) {
        std::string expression = applyTemplate(launchBounds.grid_y_expression, kernel.autotune_values);
        gridY = static_cast<int>(tinyScriptEval(expression));
    }
    if (!launchBounds.grid_z_expression.empty()) {
        std::string expression = applyTemplate(launchBounds.grid_z_expression, kernel.autotune_values);
        gridZ = static_cast<int>(tinyScriptEval(expression));
    }

    int num_warps = kernel.intrinsic_autotune_values["num_warps"];
    CUevent begin{}, end{};
    CUstream stream{};
    CUDA_CHECK(cuStreamCreate(&stream, 0));
    CUDA_CHECK(cuEventCreate(&begin, 0));
    CUDA_CHECK(cuEventCreate(&end, 0));

    // warmup
    for (int i = 0; i < 48; i++) {
        if (cuLaunchKernel(function,
                           gridX, gridY, gridZ,
                           32 * num_warps, 1, 1,
                           kernel.shared_mem_bytes,
                           stream,
                           kernel_params.data(),
                           nullptr) != CUDA_SUCCESS
                ) {
            for (auto ptr: to_free) {
                CUDA_CHECK(cuMemFree_v2(ptr));
            }
            for (auto param: kernel_params) {
                free(param);
            }
            return -1.0f;
        }
    }
    float totalMsElapsed = 0.0f;
    CUDA_CHECK(cuEventRecord(begin, stream));
    for (int i = 0; i < 1024; i++) {
        CUDA_CHECK(
                cuLaunchKernel(function,
                               gridX, gridY, gridZ,
                               32 * num_warps, 1, 1,
                               kernel.shared_mem_bytes,
                               stream,
                               kernel_params.data(),
                               nullptr)
        );
        /*
        // clear result arrays to prevent additive results from skewing up results
        for (auto &[ptr, byte_size]: to_clear) {
            CUDA_CHECK(cuMemsetD8Async(ptr, 0, byte_size, stream));
        }
        if (!to_clear.empty()) {
            CUDA_CHECK(cuStreamSynchronize(stream));
        }*/
    }
    CUDA_CHECK(cuEventRecord(end, stream));
    if (cuStreamSynchronize(stream) != CUDA_SUCCESS) {
        for (auto ptr: to_free) {
            CUDA_CHECK(cuMemFree_v2(ptr));
        }
        for (auto param: kernel_params) {
            free(param);
        }
        return -1.0f;
    }
    CUDA_CHECK(cuEventElapsedTime(&totalMsElapsed, begin, end));
    CUDA_CHECK(cuStreamDestroy_v2(stream));

    for (auto ptr: to_free) {
        CUDA_CHECK(cuMemFree_v2(ptr));
    }
    for (auto param: kernel_params) {
        free(param);
    }
    return totalMsElapsed;
}
