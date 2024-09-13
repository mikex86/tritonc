#include <iostream>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <argparse/argparse.hpp>

#include "tritonc.h"
#include "autotuner.h"
#include "templating.h"

std::string readFileContents(const std::filesystem::path &path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

std::vector<std::string> pragmaSplit(const std::string &str) {
    std::vector<std::string> result;
    std::string current;
    bool insideBraces = false;
    int bracesCountParen = 0;

    for (char c: str) {
        if (c == '{' || c == '(') {
            insideBraces = true;
            current += c;
            if (c == '(') {
                bracesCountParen++;
            }
        } else if (c == '}' || c == ')') {
            if (c == ')') {
                if (bracesCountParen > 0) {
                    bracesCountParen--;
                }
            }
            if (bracesCountParen == 0) {
                insideBraces = false;
            }
            current += c;
            if (!insideBraces) {
                result.push_back(current);
                current.clear();
            }
        } else if (c == ' ' && !insideBraces) {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }

    // Add any remaining characters that weren't followed by a space
    if (!current.empty()) {
        result.push_back(current);
    }

    return result;
}

std::vector<int> parseArray(const std::string &str) {
    std::vector<int> result;
    std::string number;
    bool insideBraces = false;

    for (char c: str) {
        if (c == '{') {
            insideBraces = true;
        } else if (c == '}') {
            insideBraces = false;
            if (!number.empty()) {
                result.push_back(std::stoi(number));
                number.clear();
            }
        } else if (insideBraces) {
            if (std::isdigit(c) || c == '-') {
                number += c;  // Accumulate digit characters or minus sign for negative numbers
            } else if (c == ',' && !number.empty()) {
                result.push_back(std::stoi(number));
                number.clear();
            }
        }
    }

    return result;
}

void preprocError(const std::string &error_msg, const std::string &line) {
    std::cerr << "[Error]: tritonc preprocessor error in line: \n" << line << "\nError: " << error_msg << std::endl;
    exit(-1);
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("tritonc");

    program.add_argument("input_file")
            .help("Specify the input file to compile.");

    program.add_argument("--compute-capability")
            .help("CUDA Compute Capability")
            .default_value(75)
            .scan<'i', int>();

    program.add_argument("--num-ctas")
            .help("Number of CTAs")
            .default_value(1)
            .scan<'i', int>();

    program.add_argument("--num-stages")
            .help("Number of Stages")
            .default_value(1)
            .scan<'i', int>();

    program.add_argument("--num-warps")
            .help("Number of Warps")
            .default_value(4)
            .scan<'i', int>();

    program.add_argument("--enable-fp-fusion")
            .help("Enable floating point fusion")
            .default_value(true);

    program.add_argument("--link")
            .help("Link against an LLVM bitcode module")
            .default_value(std::vector<std::string>{});

    program.add_argument("--ptx-version")
            .help("Target ptx version")
            .default_value(84)
            .scan<'i', int>();

    program.add_argument("--no-autotune")
            .help("Disables auto-tuning and uses specified defaults for autotune-constants")
            .implicit_value(true)
            .default_value(false);

    /*
    program.add_argument("--up-to")
            .help("Specify up to which stage to compile (llvm, ptx)")
            .default_value(std::string{"ptx"});*/

    program.add_argument("-o")
            .help("Specify the output path");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    auto inputFile = program.get<std::string>("input_file");
    int computeCapability = program.get<int>("--compute-capability");
    int ptxVersion = program.get<int>("--ptx-version");
    int numCTAs = program.get<int>("--num-ctas");
    int numStages = program.get<int>("--num-stages");
    int numWarps = program.get<int>("--num-warps");
    bool enableFpFusion = program.get<bool>("--enable-fp-fusion");
    auto libraryPaths = program.get<std::vector<std::string>>("--link");
    bool noAutotune = program.get<bool>("--no-autotune");

    std::string outputPath;
    if (program.present("-o")) {
        outputPath = program.get<std::string>("-o");
    } else {
        outputPath = inputFile.substr(0, inputFile.find_last_of('.')) + ".ptx";
    }
    std::vector<std::string> libraryNames{};
    for (auto &libraryPath: libraryPaths) {
        libraryNames.push_back(libraryPath.substr(libraryPath.find_last_of('/') + 1));
    }

    auto contents = readFileContents(inputFile);
    std::stringstream content_out;

    // process pragma directives
    std::istringstream stream(contents);
    std::string line;

    bool should_autotune = false;

    std::vector<std::pair<std::string, std::vector<int>>> autotune_constants{};
    std::vector<std::pair<std::string, std::vector<int>>> autotune_intrinsic_constants{};
    std::unordered_map<std::string, int> default_autotune_constants{};
    std::unordered_map<std::string, int> default_intrinsic_autotune_constants{};

    std::vector<KernelArgument> kernel_arguments;
    std::optional<KernelLaunchBounds> launch_bounds = std::nullopt;
    std::string launch_function_name = "";

    while (std::getline(stream, line)) {
        if (line.find("//") == 0) {
            // ignore comments
        } else if (line.find("#pragma") == 0) {
            std::string command = line.substr(8);
            std::vector<std::string> args = pragmaSplit(command);
            if (args.empty()) {
                preprocError("pragma preprocessor instruction excepts at least one directive", line);
            }
            const std::string &directive = args[0];
            if (directive == "num_warps") {
                if (args.size() != 2) {
                    preprocError("num_warps directive expects two arguments", line);
                }
                const std::string &value = args[1];
                int new_num_warps = std::stoi(value);
                if (program.is_used("--num-warps") && new_num_warps != numWarps) {
                    std::cerr
                            << "[Warning] tritonc CLI argument --num-warps conflicts with source-level #pragma directive. Directive requests "
                            << new_num_warps << ", while CLI argument requests " << numWarps
                            << "!\nPrioritizing source-level directive over CLI argument..." << std::endl;
                }
                numWarps = new_num_warps;
            } else if (directive == "num_stages") {
                if (args.size() != 2) {
                    preprocError("num_stages directive expects one argument", line);
                }
                const std::string &value = args[1];
                int new_num_stages = std::stoi(value);
                if (program.is_used("--num-stages") && new_num_stages != numStages) {
                    std::cerr
                            << "[Warning] tritonc CLI argument --num-stages conflicts with source-level #pragma directive. Directive requests "
                            << new_num_stages << ", while CLI argument requests " << numStages
                            << "!\nPrioritizing source-level directive over CLI argument..." << std::endl;
                }
                numStages = new_num_stages;
            } else if (directive == "autotune") {
                should_autotune = true;
                if (args[1] == "intrinsic") {
                    if (args.size() >= 4) {
                        const std::string &autotune_constant = args[2];
                        const std::string &autotune_rage = args[3];
                        auto range = parseArray(autotune_rage);
                        autotune_intrinsic_constants.emplace_back(autotune_constant, range);
                    }
                    if (args.size() == 6) {
                        if (args[4] == "default") {
                            const std::string &autotune_constant = args[2];
                            int value = std::stoi(args[5]);
                            default_autotune_constants[autotune_constant] = value;
                        } else {
                            preprocError(
                                    "autotune directive expects an optional 'default' keyword after grid initializer list",
                                    line);
                        }
                    }
                    if (args.size() != 4 && args.size() != 6) {
                        preprocError("autotune directive expects either three or five arguments", line);
                    }
                    continue;
                }
                if (args.size() >= 3) {
                    const std::string &autotune_constant = args[1];
                    const std::string &autotune_range = args[2];
                    auto range = parseArray(autotune_range);
                    autotune_constants.emplace_back(autotune_constant, range);
                }

                if (args.size() == 5) {
                    if (args[3] == "default") {
                        const std::string &autotune_constant = args[1];
                        int value = std::stoi(args[4]);
                        default_autotune_constants[autotune_constant] = value;
                    } else {
                        preprocError(
                                "autotune directive expects an optional 'default' keyword after grid initializer list",
                                line);
                    }
                }
                if (args.size() != 3 && args.size() != 5) {
                    preprocError("autotune directive expects two or four arguments", line);
                }
            } else if (directive == "argument") {
                if (args.size() != 4 && args.size() != 5) {
                    preprocError("argument directive expects three or four arguments", line);
                }
                const std::string &arg_idx_str = args[1];
                int arg_idx = std::stoi(arg_idx_str);

                const std::string &dtype_str = args[2];
                std::string value = args[3];
                if (arg_idx >= kernel_arguments.size()) {
                    kernel_arguments.resize(arg_idx + 1);
                }
                KerArgDtype dtype{};
                if (dtype_str == "ptr") {
                    dtype = PTR;
                } else if (dtype_str == "i32") {
                    dtype = I32;
                } else if (dtype_str == "i64") {
                    dtype = I64;
                } else {
                    preprocError("Unknown datatype used in argument directive: " + dtype_str, line);
                }

                bool is_malloc = false;
                if (value.find("cuMalloc") == 0) {
                    is_malloc = true;
                    if (dtype != PTR) {
                        preprocError("cuMalloc() cannot be used as initialized for non ptr argument!", line);
                    }
                    value = value.substr(8);
                    if (value[0] != '(') {
                        preprocError("argument initializing with cuMalloc() must have a '(' after cuMalloc", line);
                    }
                    if (value[value.size() - 1] != ')') {
                        preprocError("argument directive involving cuMalloc must end with a closing ')'", line);
                    }
                    value = value.substr(1, value.size() - 2);
                }
                bool is_dst_ptr = false;
                if (args.size() == 5) {
                    auto postfix = args[4];
                    if (postfix == "dst") {
                        if (dtype != PTR) {
                            preprocError("dst postfix can only be used for ptr type argument", line);
                        }
                        is_dst_ptr = true;
                    }
                }
                kernel_arguments[arg_idx] = KernelArgument{dtype, value, is_malloc, is_dst_ptr};
            } else if (directive == "grid") {
                if (args.size() != 3) {
                    preprocError("grid directive expects two arguments", line);
                }
                const std::string &grid_dim_name = args[1];
                const std::string &value_expression = args[2];

                if (grid_dim_name != "x" && grid_dim_name != "y" && grid_dim_name != "z") {
                    preprocError("grid directive expects valid dimension name. Either 'x', 'y' or 'z'", line);
                }
                if (launch_bounds == std::nullopt) {
                    launch_bounds = KernelLaunchBounds{};
                }
                if (grid_dim_name == "x") {
                    launch_bounds->grid_x_expression = value_expression;
                }
                if (grid_dim_name == "y") {
                    launch_bounds->grid_y_expression = value_expression;
                }
                if (grid_dim_name == "z") {
                    launch_bounds->grid_z_expression = value_expression;
                }
            } else if (directive == "launch") {
                if (args.size() != 2) {
                    preprocError("launch directive expects one argument", line);
                }
                const std::string &function_name = args[1];
                launch_function_name = function_name;
            }
        } else {
            content_out << line << '\n';
        }
    }

    initLLVM();
    mlir::MLIRContext *context = createContext();
    if (!should_autotune || noAutotune) {
        auto source = content_out.str();
        if (should_autotune) {
            auto default_num_warps = default_intrinsic_autotune_constants["num_warps"];
            if (default_num_warps != 0) {
                numWarps = default_num_warps;
            }
            auto default_num_stages = default_intrinsic_autotune_constants["num_stages"];
            if (default_num_stages != 0) {
                numStages = default_num_stages;
            }
            source = applyTemplate(source, default_autotune_constants);
        }
        try {
            auto [ptxCode, shared_mem_bytes] = compile(context, source, computeCapability, ptxVersion,
                                                       numCTAs, numStages,
                                                       numWarps,
                                                       enableFpFusion,
                                                       libraryNames,
                                                       libraryPaths, "ptx");
            std::ofstream ostream(outputPath);
            ostream << "// Shared memory requirements: " << std::to_string(shared_mem_bytes) << '\n';
            ostream << ptxCode;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
        } catch (...) {
        }
    } else {
        if (!launch_bounds) {
            std::cerr
                    << "[Error]: autotune pragma commands were issued, but no launch bounds were specified. Use eg. #pragma grid x 123 to set the grid size along the specified dimension."
                    << std::endl;
            return -1;
        }
        if (launch_function_name.empty()) {
            std::cerr
                    << "[Error]: autotune pragma commands were issued, but no function name to launch was specified. Use eg. #pragma launch function_name to set the name of the function to tune."
                    << std::endl;
            return -1;
        }
        autotune(content_out.str(), computeCapability, ptxVersion,
                 enableFpFusion, libraryNames, libraryPaths,
                 launch_function_name,
                 *launch_bounds,
                 kernel_arguments, autotune_constants, autotune_intrinsic_constants);
    }
    return 0;
}
