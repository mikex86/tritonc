#include "autotuner.h"
#include "tritonc.h"
#include "kernel_runner.h"
#include "templating.h"
#include <sstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>

#include <mlir/IR/MLIRContext.h>


#define MAX_QUEUE_SIZE 256

struct KernelCompileTask {
    std::unordered_map<std::string, int> autotune_values;
    std::unordered_map<std::string, int> intrinsic_autotune_values;

    std::string contents;
    int computeCapability{};
    int ptxVersion{};
    int numCTAs{};
    int numStages{};
    int numWarps{};
    bool enableFpFusion{};
    std::vector<std::string> libraryNames;
    std::vector<std::string> libraryPaths;

    KernelCompileTask &operator=(const KernelCompileTask &other) {
        autotune_values = other.autotune_values;
        intrinsic_autotune_values = other.intrinsic_autotune_values;
        contents = other.contents;
        computeCapability = other.computeCapability;
        ptxVersion = other.ptxVersion;
        numCTAs = other.numCTAs;
        numStages = other.numStages;
        numWarps = other.numWarps;
        enableFpFusion = other.enableFpFusion;
        libraryNames = other.libraryNames;
        libraryPaths = other.libraryPaths;
        return *this;
    }
};


std::mutex compile_tasks_queue_mutex{};
std::queue<KernelCompileTask> compile_tasks{};


std::mutex compiled_kernel_queue_mutex{};
std::vector<CompiledKernel> compiled_kernels{};


std::atomic<bool> compiling(true);
std::atomic<int> kernel_counter{0};

bool nextCompileTask(KernelCompileTask &taskOut) {
    std::lock_guard guard{compile_tasks_queue_mutex};
    if (compile_tasks.empty()) {
        return false;
    }
    auto task = compile_tasks.front();
    taskOut = task;
    compile_tasks.pop();
    return true;
}

void compile_thread() {
    while (compiling || !compile_tasks.empty()) {
        while (compiling) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            {
                compile_tasks_queue_mutex.lock();
                if (!compile_tasks.empty()) {
                    compile_tasks_queue_mutex.unlock();
                    break;
                }
                compile_tasks_queue_mutex.unlock();
            }
        }

        KernelCompileTask task{};
        if (!nextCompileTask(task)) continue;

        auto [values, intrinsic_values, kernel_code, computeCapability, ptxVersion, numCTAs,
                num_stages, num_warps, enableFusion,
                libraryNames, libraryPaths] = task;

        auto context = createContext();
        try {
            auto [ptx, shared_mem_bytes] = compile(context, kernel_code, computeCapability, ptxVersion, numCTAs,
                                                   num_stages, num_warps, enableFusion,
                                                   libraryNames, libraryPaths, "ptx");

            // publish compiled kernel
            {
                std::lock_guard guard{compiled_kernel_queue_mutex};
                compiled_kernels.push_back(CompiledKernel{
                        .ptx=ptx,
                        .shared_mem_bytes=shared_mem_bytes,
                        .autotune_values=values,
                        .intrinsic_autotune_values=intrinsic_values
                });
            }

            if (++kernel_counter % 128 == 0) {
                std::cout << "[Autotuner] Compiled " << kernel_counter << " kernels..." << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "Failed to compile a kernel permutation, skipping: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Failed to compile a kernel permutation, skipping..." << std::endl;
        }
        delete context;
    }
}

void
compileKernelPermutation(const std::string &codeTemplate,
                         int computeCapability, int ptxVersion, bool enableFusion,
                         const std::vector<std::string> &libraryNames,
                         const std::vector<std::string> &libraryPaths,
                         const std::unordered_map<std::string, int> &autotuneValues,
                         const std::vector<std::pair<std::string, std::vector<int>>> &intrinsicAutotuneConstants) {

    // compile kernel
    auto kernel_code = applyTemplate(codeTemplate, autotuneValues);

    int numCTAs = 1; // TODO: Don't hardcode this

    auto permutation_counter = std::vector<uint32_t>();
    permutation_counter.resize(intrinsicAutotuneConstants.size());

    // enumerate all permutations
    while (true) {
        permutation_available:

        // get values of permutation
        std::unordered_map<std::string, int> instrinsic_permutation_values{};
        for (size_t j = 0; j < permutation_counter.size(); j++) {
            size_t value_idx = permutation_counter[j];
            auto key = intrinsicAutotuneConstants[j].first;
            auto value = intrinsicAutotuneConstants[j].second[value_idx];
            instrinsic_permutation_values[key] = value;
        }

        auto num_stages = instrinsic_permutation_values["num_stages"];
        auto num_warps = instrinsic_permutation_values["num_warps"];

        {
            std::lock_guard guard{compile_tasks_queue_mutex};
            compile_tasks.push(KernelCompileTask{
                    autotuneValues, instrinsic_permutation_values,
                    kernel_code, computeCapability, ptxVersion, numCTAs,
                    num_stages, num_warps, enableFusion,
                    libraryNames, libraryPaths
            });
        }
        while (compile_tasks.size() >= MAX_QUEUE_SIZE) {
            std::this_thread::yield();
        }

        // advance permutation
        {
            size_t i = 0;
            while (true) {
                while (permutation_counter[i] < intrinsicAutotuneConstants[i].second.size() - 1) {
                    permutation_counter[i]++;
                    goto permutation_available;
                }
                permutation_counter[i] = 0;
                i++;
                if (i == permutation_counter.size()) {
                    goto no_permutation_available;
                }
            }
        }
    }
    no_permutation_available:
}

void
autotune(const std::string &codeTemplate, int computeCapability, int ptxVersion, bool enableFpFusion,
         const std::vector<std::string> &libraryNames, const std::vector<std::string> &libraryPaths,
         const std::string &to_launch_function_name,
         const KernelLaunchBounds &launchBounds,
         std::vector<KernelArgument> kernel_arguments,
         const std::vector<std::pair<std::string, std::vector<int>>> &autotuneConstants,
         const std::vector<std::pair<std::string, std::vector<int>>> &intrinsicAutotuneConstants) {

    auto permutation_counter = std::vector<uint32_t>();
    permutation_counter.resize(autotuneConstants.size());

    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads{};
    for (int i = 0; i < n_threads; i++) {
        threads.push_back(std::thread(compile_thread));
    }
    std::cout << "[Autotuner] Compiling kernels..." << std::endl;

    // enumerate all permutations
    while (true) {
        permutation_available:

        // get values of permutation
        std::unordered_map<std::string, int> permutation_values{};
        for (size_t j = 0; j < permutation_counter.size(); j++) {
            size_t value_idx = permutation_counter[j];
            auto key = autotuneConstants[j].first;
            auto value = autotuneConstants[j].second[value_idx];
            permutation_values[key] = value;
        }

        compileKernelPermutation(codeTemplate, computeCapability, ptxVersion, enableFpFusion,
                                 libraryNames, libraryPaths, permutation_values, intrinsicAutotuneConstants);

        // advance permutation
        {
            size_t i = 0;
            while (true) {
                while (permutation_counter[i] < autotuneConstants[i].second.size() - 1) {
                    permutation_counter[i]++;
                    goto permutation_available;
                }
                permutation_counter[i] = 0;
                i++;
                if (i == permutation_counter.size()) {
                    goto no_permutation_available;
                }
            }
        }
    }

    no_permutation_available:
    compiling = false;
    for (auto &t: threads) {
        t.join();
    }
    std::cout << "[Autotuner] Finished compiling kernels." << std::endl;
    std::cout << "[Autotuner] Start testing kernels..." << std::endl;

    // we don't overlap kernel compilation and kernel testing because the heavy load might
    // skew timings due to heavy context switching

    CompiledKernel best_kernel;
    float best_time = -1.0f;
    size_t i = 0;
    for (auto &compiled_kernel: compiled_kernels) {
        float ms = benchmarkKernel(compiled_kernel, to_launch_function_name, launchBounds, kernel_arguments);
        i++;
        if (ms != -1.0f && (ms < best_time || best_time == -1.0f)) {
            best_kernel = compiled_kernel;
            best_time = ms;
        }
        if (i % 128 == 0) {
            std::cout << "[Autotuner] Benchmarked " << i << "/" << compiled_kernels.size() << " kernels..."
                      << std::endl;
        }
    }
    std::cout << "Best kernel: " << best_time << "ms" << std::endl;

    std::cout << "values:" << std::endl;
    for (auto &[key, value]: best_kernel.autotune_values) {
        std::cout << '\t' << key << ": " << value << std::endl;
    }

    std::cout << "intrinsic values:" << std::endl;
    for (auto &[key, value]: best_kernel.intrinsic_autotune_values) {
        std::cout << '\t' << key << ": " << value << std::endl;
    }
}

