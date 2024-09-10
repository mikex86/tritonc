#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

struct CompiledKernel {
    std::string ptx;
    uint32_t shared_mem_bytes;
    std::unordered_map<std::string, int> autotune_values;
    std::unordered_map<std::string, int> intrinsic_autotune_values;
};

enum KerArgDtype {
    PTR, I32, I64
};

struct KernelArgument {
    KerArgDtype data_type;
    std::string init_expression;
    bool is_malloc;
    bool is_dst_ptr;
};

struct KernelLaunchBounds {
    std::string grid_x_expression;
    std::string grid_y_expression;
    std::string grid_z_expression;
};

float benchmarkKernel(CompiledKernel &kernel,
                      const std::string &to_launch_function_name, const KernelLaunchBounds &launchBounds,
                      const std::vector<KernelArgument> &arguments);