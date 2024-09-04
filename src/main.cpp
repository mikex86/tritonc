#include <iostream>
#include <filesystem>
#include <fstream>
#include <argparse/argparse.hpp>

// LLVM targets
#include <llvm/Support/TargetSelect.h>

// MLIR dialects
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

// must be included before triton Passes.h.inc files
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

// TritonGPU
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <triton/Dialect/Triton/Transforms/Passes.h>
#include <triton/Dialect/TritonGPU/IR/Dialect.h>
#include <triton/Dialect/TritonGPU/Transforms/Passes.h>
#include <triton/Dialect/TritonNvidiaGPU/IR/Dialect.h>
#include <triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h>
#include <triton/Conversion/TritonToTritonGPU/Passes.h>
#include <triton/Conversion/TritonGPUToLLVM/Passes.h>

// TritonNVIDIA
#include <Dialect/NVGPU/IR/Dialect.h>
#include <TritonNVIDIAGPUToLLVM/Passes.h>

// miscellaneous LLVM/MLIR
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Target/TargetMachine.h>


static mlir::MLIRContext *createContext() {
    auto context = new mlir::MLIRContext();
    auto &diagEngine = context->getDiagEngine();

    diagEngine.registerHandler([](const mlir::Diagnostic &diag) {});

    mlir::DialectRegistry registry;
    registry.insert<
            // triton general dialects
            mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
            mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::index::IndexDialect,
            mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
            mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect,

            // nvidia specific dialects
            mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
            mlir::triton::nvgpu::NVGPUDialect>();

    // mlir format llvm to llvm module translation
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    // also nvidia specific
    mlir::registerNVVMDialectTranslation(registry);
    mlir::LLVM::registerInlinerInterface(registry);
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parseMlirModule(mlir::MLIRContext *context, const std::string &mlirString) {
    mlir::DiagnosticEngine &engine = context->getDiagEngine();
    mlir::DiagnosticEngine::HandlerID id = -1;

    try {
        // Set up a diagnostic handler to obtain the error message
        std::string errorMsg{};
        id = engine.registerHandler(
                [&errorMsg](mlir::Diagnostic &diag) -> mlir::LogicalResult {
                    std::string severityName;
                    switch (diag.getSeverity()) {
                        case mlir::DiagnosticSeverity::Note:
                            severityName = "note";
                            break;
                        case mlir::DiagnosticSeverity::Warning:
                            severityName = "warning";
                            break;
                        case mlir::DiagnosticSeverity::Error:
                            severityName = "error";
                            break;
                        case mlir::DiagnosticSeverity::Remark:
                            severityName = "remark";
                            break;
                    }

                    std::string locationStr;
                    llvm::raw_string_ostream os(locationStr);
                    diag.getLocation().print(os);
                    os.flush();
                    errorMsg += "[" + severityName + "] " + locationStr + ": " + diag.str() + "\n";
                    return mlir::failure(false);
                }
        );

        // parse module
        auto mod = mlir::parseSourceString<mlir::ModuleOp>(mlirString, context);
        engine.eraseHandler(id);

        if (!mod) {
            std::cerr << errorMsg << std::endl;
            return nullptr;
        }

        return std::move(mod);
    } catch (const std::exception &e) {
        if (id != -1) {
            engine.eraseHandler(id);
        }
        throw; // Re-throw the caught exception
    }
}

llvm::LogicalResult
convertTTIRToTTGIR(mlir::OwningOpRef<mlir::ModuleOp> &mod, int numWarps, int numStages, int numCTAs,
                   int computeCapability) {
    mlir::DiagnosticEngine::HandlerID id = -1;
    mlir::DiagnosticEngine &engine = mod->getContext()->getDiagEngine();
    try {
        mlir::PassManager pm(mod->getContext());
        std::string targetString = "cuda:" + std::to_string(computeCapability);

        // TTIR -> TTGIR
        pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(targetString, numWarps, 32, numCTAs));

        // optimize TTGIR
        pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());

        if (computeCapability / 10 >= 8)
            pm.addPass(mlir::triton::gpu::createTritonGPUF32DotTC());

        // nvidia.passes.ttnvgpuir
        mlir::triton::nvidia_gpu::ClusterInfo clusterInfo{};
        pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(&clusterInfo));

        // passes.ttgpuir
        pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
        pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
        pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
        pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
        pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(
                {.hoistLayoutConversion = computeCapability >= 80}));

        // passes.common
        pm.addPass(mlir::createCSEPass());

        // passes.ttgpuir
        if (computeCapability / 10 >= 8) {
            pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
            pm.addPass(mlir::triton::gpu::createTritonGPUPipeline({.numStages = numStages}));
        }

        // passes.ttgpuir
        pm.addPass(mlir::triton::gpu::createTritonGPUPrefetch());
        pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(
                {.hoistLayoutConversion = computeCapability >= 80}));
        pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
        pm.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
        pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructions());

        // passes.common
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createSymbolDCEPass());

        // nvidia.passes
        if (computeCapability / 10 >= 9) {
            pm.addPass(mlir::createTritonNvidiaGPUFenceInsertionPass());
            pm.addPass(mlir::createTritonNvidiaGPUTMALoweringPass());
        }

        // passes.common
        pm.addPass(mlir::createCanonicalizerPass());

        std::string errorMsg;
        id = engine.registerHandler([&errorMsg](mlir::Diagnostic &diag) -> mlir::LogicalResult {
            std::string severityName;
            switch (diag.getSeverity()) {
                case mlir::DiagnosticSeverity::Note:
                    severityName = "note";
                    break;
                case mlir::DiagnosticSeverity::Warning:
                    severityName = "warning";
                    break;
                case mlir::DiagnosticSeverity::Error:
                    severityName = "error";
                    break;
                case mlir::DiagnosticSeverity::Remark:
                    severityName = "remark";
                    break;
            }

            std::string locationStr;
            llvm::raw_string_ostream os(locationStr);
            diag.getLocation().print(os);
            os.flush();

            errorMsg += std::string("[") + severityName + "] " + locationStr + ": " + diag.str() + "\n";
            std::cerr << errorMsg << std::endl;
            return mlir::failure(false);
        });

        mlir::LogicalResult result = pm.run(mod->getOperation());
        engine.eraseHandler(id);
        return result;
    } catch (const std::exception &e) {
        if (id != -1) {
            engine.eraseHandler(id);
        }
        throw;
    }
}

std::string readFileContents(const std::filesystem::path &path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

mlir::LogicalResult optimizeTtir(mlir::OwningOpRef<mlir::ModuleOp> &mod) {
    mlir::PassManager pm(mod->getContext());

    // passes.common
    pm.addPass(mlir::createInlinerPass());

    // passes.ttir
    pm.addPass(mlir::triton::createRewriteTensorPointerPass());
    pm.addPass(mlir::triton::createCombineOpsPass());

    // passes.common
    pm.addPass(mlir::createCanonicalizerPass());

    // passes.ttir
    pm.addPass(mlir::triton::createReorderBroadcastPass());

    // passes.common
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::createSymbolDCEPass());

    // Set up a diagnostic handler to obtain the error message
    std::string errorMsg;

    mlir::DiagnosticEngine &engine = mod->getContext()->getDiagEngine();
    mlir::DiagnosticEngine::HandlerID id = engine.registerHandler(
            [&errorMsg](mlir::Diagnostic &diag) -> mlir::LogicalResult {
                std::string severityName;
                switch (diag.getSeverity()) {
                    case mlir::DiagnosticSeverity::Note:
                        severityName = "note";
                        break;
                    case mlir::DiagnosticSeverity::Warning:
                        severityName = "warning";
                        break;
                    case mlir::DiagnosticSeverity::Error:
                        severityName = "error";
                        break;
                    case mlir::DiagnosticSeverity::Remark:
                        severityName = "remark";
                        break;
                }

                std::string locationStr;
                llvm::raw_string_ostream os(locationStr);
                diag.getLocation().print(os);
                os.flush();

                errorMsg += "[" + severityName + "] " + locationStr + ": " + diag.str() + "\n";

                return mlir::failure(false);
            });

    mlir::LogicalResult result = pm.run(mod->getOperation());
    engine.eraseHandler(id);

    if (mlir::failed(result)) {
        throw std::runtime_error("Failed to optimize TTIR: " + errorMsg);
    }

    return result;
}

// hack because the header includes stuff that is not public...
// Triton is not clean about separating PRIVATE and PUBLIC headers... (sigh)
namespace mlir {
    std::unique_ptr<Pass> createLLVMDIScopePass();
}

std::unique_ptr<llvm::Module>
convertTTGIRToLLIR(
        llvm::LLVMContext &llvmContext,
        mlir::OwningOpRef<mlir::ModuleOp> &mod,
        int computeCapability, bool isROCM,
        const std::vector<std::string> &libraryNames,
        const std::vector<std::string> &libraryPaths) {
    mlir::DiagnosticEngine &engine = mod->getContext()->getDiagEngine();
    auto id = static_cast<mlir::DiagnosticEngine::HandlerID>(-1);

    std::string errorMsg;
    id = engine.registerHandler([&errorMsg](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        std::string severityName;
        switch (diag.getSeverity()) {
            case mlir::DiagnosticSeverity::Note:
                severityName = "note";
                break;
            case mlir::DiagnosticSeverity::Warning:
                severityName = "warning";
                break;
            case mlir::DiagnosticSeverity::Error:
                severityName = "error";
                break;
            case mlir::DiagnosticSeverity::Remark:
                severityName = "remark";
                break;
        }

        std::string locationStr;
        llvm::raw_string_ostream os(locationStr);
        diag.getLocation().print(os);
        os.flush();

        errorMsg += "[" + severityName + "] " + locationStr + ": " + diag.str() + "\n";
        return mlir::failure(false);
    });

    mlir::PassManager pm(mod->getContext());
    pm.addPass(mlir::triton::NVIDIA::createDecomposeUnsupportedConversionsPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());
    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(computeCapability));
    pm.addPass(mlir::createConvertNVGPUToNVVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createLLVMDIScopePass());

    if (mlir::failed(pm.run(*mod))) {
        engine.eraseHandler(id);
        throw std::runtime_error("Failed to lower TritonGPU to LLVM IR (still in MLIR format): " + errorMsg);
    }
    engine.eraseHandler(id);

    auto llvmModule = mlir::translateModuleToLLVMIR(*mod, llvmContext);

    if (!llvmModule) {
        throw std::runtime_error("Failed to convert MLIR format LLVM IR to LLVM Module: " + errorMsg);
    }

    llvm::Type *i32 = llvm::Type::getInt32Ty(llvmContext);
    auto *mdFour = llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 4));
    auto *mdName = llvm::MDString::get(llvmContext, "nvvm-reflect-ftz");
    auto *mdOne = llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 1));
    auto *reflect = llvm::MDNode::get(llvmContext, {mdFour, mdName, mdOne});
    llvmModule->addModuleFlag(reflect);

    for (size_t i = 0; i < libraryNames.size(); ++i) {
        const auto &libraryName = libraryNames[i];
        const auto &libraryPath = libraryPaths[i];

        llvm::SMDiagnostic err;
        auto extMod = llvm::parseIRFile(libraryPath, err, llvmContext);
        if (!extMod) {
            llvm::errs() << "Failed to load " << libraryPath;
            continue;
        }
        extMod->setTargetTriple(llvmModule->getTargetTriple());
        extMod->setDataLayout(llvmModule->getDataLayout());
        if (llvm::Linker::linkModules(*llvmModule, std::move(extMod), llvm::Linker::Flags::LinkOnlyNeeded)) {
            llvm::errs() << "Failed to link library " << libraryName << " at location " << libraryPath;
            continue;
        }
    }

    llvm::OptimizationLevel opt = llvm::OptimizationLevel::O3;

    llvm::PassBuilder pb;
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback([&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
        fpm.addPass(llvm::InstCombinePass());
    });

    mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
    mpm.run(*llvmModule, mam);

    return llvmModule;
}

static std::string translateLLVMIRToASM(llvm::Module &module,
                                        const std::string &triple,
                                        const std::string &proc,
                                        const std::string &features,
                                        const std::vector<std::string> &flags,
                                        bool enable_fp_fusion, bool isObject) {
    using namespace mlir;
    // options
    auto options = llvm::cl::getRegisteredOptions();
    for (std::string flag: flags) {
        auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
        assert(shortPtr);
        shortPtr->setValue(true);
    }

    // inline everything
    for (llvm::Function &f: module.functions())
        if (!f.hasFnAttribute(llvm::Attribute::NoInline))
            f.addFnAttr(llvm::Attribute::AlwaysInline);

    // verify and store llvm
    llvm::legacy::PassManager pm;
    pm.add(llvm::createAlwaysInlinerLegacyPass());
    pm.add(llvm::createVerifierPass());
    pm.run(module);

    // create machine
    module.setTargetTriple(triple);
    std::string error;
    auto target =
            llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
    llvm::TargetOptions opt;
    if (enable_fp_fusion)
        opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    opt.UnsafeFPMath = false;
    opt.NoInfsFPMath = false;
    opt.NoNaNsFPMath = true;
    opt.TrapUnreachable = true;
    std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
            module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
            std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
    // set data layout
    module.setDataLayout(machine->createDataLayout());
    // emit machine code
    std::string result;
    {
        llvm::raw_string_ostream stream(result);
        llvm::buffer_ostream pstream(stream);
        for (llvm::Function &f: module.functions())
            f.addFnAttr(llvm::Attribute::AlwaysInline);
        llvm::legacy::PassManager pass;
        // emit
        auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                                 : llvm::CodeGenFileType::AssemblyFile;
        machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
        pass.run(module);
    }
    return result;
}

static std::string translateToAsm(llvm::Module &module,
                                  const std::string &triple, const std::string &proc,
                                  const std::string &features, const std::vector<std::string> &flags,
                                  bool enable_fp_fusion, bool isObject) {
    return translateLLVMIRToASM(
            module, triple, proc, features, flags, enable_fp_fusion, isObject
    );
}

void compile(mlir::MLIRContext *context, const std::string &contents, int computeCapability, int numCTAs, int numStages,
             int numWarps, bool enableFpFusion, const std::vector<std::string> &libraryNames,
             const std::vector<std::string> &libraryPaths, const std::string &upToStage,
             const std::string &outputPath) {
    auto module = parseMlirModule(context, contents);

    if (llvm::failed(optimizeTtir(module))) {
        return;
    }

    if (llvm::failed(convertTTIRToTTGIR(module, numWarps, numStages, numCTAs, computeCapability))) {
        return;
    }

    llvm::LLVMContext llvmContext{};
    auto llvmModule = convertTTGIRToLLIR(llvmContext, module, computeCapability, false, libraryNames, libraryPaths);

    std::string proc = computeCapability == 90 ? "sm_90a" : "sm_" + std::to_string(computeCapability);
    std::string ptxCode = translateToAsm(*llvmModule, "nvptx64-nvidia-cuda", proc, "", {"nvptx-short-ptr"},
                                         enableFpFusion, false);

    std::ofstream stream(outputPath);
    stream << ptxCode;
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
            .default_value(3)
            .scan<'i', int>();

    program.add_argument("--num-warps")
            .help("Number of Warps")
            .default_value(2)
            .scan<'i', int>();

    program.add_argument("--enable-fp-fusion")
            .help("Enable floating point fusion")
            .default_value(true)
            .implicit_value(false);

    program.add_argument("--link")
            .help("Link against an LLVM bitcode module")
            .default_value(std::vector<std::string>{});

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
    int numCTAs = program.get<int>("--num-ctas");
    int numStages = program.get<int>("--num-stages");
    int numWarps = program.get<int>("--num-warps");
    bool enableFpFusion = program.get<bool>("--enable-fp-fusion");
    auto libraryPaths = program.get<std::vector<std::string>>("--link");
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

    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    auto contents = readFileContents(inputFile);
    mlir::MLIRContext *context = createContext();
    try {
        compile(context, contents, computeCapability, numCTAs, numStages, numWarps, enableFpFusion, libraryNames,
                libraryPaths, "ptx", outputPath);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    } catch (...) {
    }
    return 0;
}
