#include "tritonc.h"

#include <unordered_set>
#include <iostream>
#include <fstream>


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
#include <llvm/Passes/StandardInstrumentations.h>
#include "NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "Target/LLVMIR/LLVMPasses.h"

void initLLVM() {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
}

mlir::MLIRContext *createContext() {
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
        throw e;
    }
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
        int computeCapability, bool isROCM, int ptxVersion,
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
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());
    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(computeCapability));
    pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());
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

    // attach data-layout
    {
        std::string error;
        auto triple = "nvptx64-nvidia-cuda";
        auto proc = (computeCapability == 90) ? "sm_90a" : "sm_" + std::to_string(computeCapability);
        auto target = llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target) {
            throw std::runtime_error("target lookup error: " + error);
        }

        std::string features = "+ptx" + std::to_string(ptxVersion);

        llvm::TargetOptions opt;
        // Target machine is only used to create the data layout.
        std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
                triple, proc, features, opt, llvm::Reloc::PIC_, std::nullopt,
                llvm::CodeGenOptLevel::None)};

        // set data layout
        llvmModule->setDataLayout(machine->createDataLayout());
    }

    // set_nvvm_reflect_ftz
    {
        // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
        // this will enable fast math path in libdevice
        // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
        // sqrt.approx.ftz.f32
        using namespace llvm;
        auto &ctx = llvmModule->getContext();
        Type *i32 = Type::getInt32Ty(ctx);
        auto *mdFour = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 4));
        auto *mdName = MDString::get(ctx, "nvvm-reflect-ftz");
        auto *mdOne = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 1));
        auto *reflect = MDNode::get(ctx, {mdFour, mdName, mdOne});
        llvmModule->addModuleFlag(reflect);
    }

    // TODO: maxnreg

    // link libraries
    {
        llvm::LLVMContext &ctx = llvmModule->getContext();
        llvm::Linker linker(*llvmModule);
        for (const std::string &path: libraryPaths) {
            llvm::SMDiagnostic err;
            std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
            if (!libMod) {
                std::string message = "Failed to parse library at " + path;
                throw std::invalid_argument(message);
            }
            libMod->setTargetTriple(llvmModule->getTargetTriple());
            libMod->setDataLayout(llvmModule->getDataLayout());

            std::unordered_set<std::string> externalFns;
            for (llvm::Function &fn: libMod->functions()) {
                if (!fn.isDeclaration())
                    externalFns.insert(fn.getName().str());
            }

            if (linker.linkInModule(std::move(libMod),
                                    llvm::Linker::Flags::LinkOnlyNeeded)) {
                std::string message = "Failed to link library at " + path;
                throw std::invalid_argument(message);
            }

            // Mark linked-in functions as internal because backends use external
            // linkage as a signifier of kernel functions.
            for (llvm::Function &fn: llvmModule->functions()) {
                if (externalFns.count(fn.getName().str())) {
                    fn.setLinkage(llvm::GlobalValue::InternalLinkage);
                }
            }
        }
    }

    // optimize module
    {
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;

        llvm::PassInstrumentationCallbacks *instrCbPtr = nullptr;
        llvm::PassInstrumentationCallbacks passInstrCb;
        llvm::StandardInstrumentations standardInstr(llvmModule->getContext(),
                /*DebugLogging*/ true);

        llvm::PipelineTuningOptions tuningOptions;
        tuningOptions.LoopUnrolling = true;
        tuningOptions.LoopInterleaving = true;
        tuningOptions.LoopVectorization = true;
        // TODO: currently we run SLP vectorizer with an empty target machine.
        // This cause the vectorizer to create larger vector which could be bad.
        // Disabling it would currently cause regressions as this pass also applies
        // some scheduling that helps performance in some cases. We should work on
        // using NVPTX target instead and address the performance regressions with
        // some scheduling solution.
        tuningOptions.SLPVectorization = true;

        llvm::PassBuilder pb(nullptr /*targetMachine*/, tuningOptions, std::nullopt,
                             instrCbPtr);

        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm;
        pb.registerVectorizerStartEPCallback(
                [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
                    // Triton generates large structure of scalars which may pessimise
                    // optimizations, we run a pass to break up phi of struct to make
                    // sure all the struct are removed for the following passes.
                    fpm.addPass(llvm::BreakStructPhiNodesPass());
                    fpm.addPass(llvm::InstCombinePass());
                });
        mpm.addPass(pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
        mpm.run(*llvmModule, mam);
    }

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

std::pair<std::string, uint32_t>
compile(mlir::MLIRContext *context, const std::string &contents, int computeCapability, int ptxVersion, int numCTAs,
        int numStages,
        int numWarps, bool enableFpFusion, const std::vector<std::string> &libraryNames,
        const std::vector<std::string> &libraryPaths, const std::string &upToStage) {
    auto module = parseMlirModule(context, contents);

    if (llvm::failed(optimizeTtir(module))) {
        return {"", 0};
    }

    if (llvm::failed(convertTTIRToTTGIR(module, numWarps, numStages, numCTAs, computeCapability))) {
        return {"", 0};
    }

    llvm::LLVMContext llvmContext{};
    auto llvmModule = convertTTGIRToLLIR(llvmContext, module, computeCapability, false, ptxVersion, libraryNames,
                                         libraryPaths);

    auto shared_mem_bytes = module.get()->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared").getInt();

    std::string proc = computeCapability == 90 ? "sm_90a" : "sm_" + std::to_string(computeCapability);
    std::string ptxCode = translateToAsm(*llvmModule, "nvptx64-nvidia-cuda", proc, "", {"nvptx-short-ptr"},
                                         enableFpFusion, false);

    // post process
    std::string ptx_version = std::to_string(ptxVersion / 10) + "." + std::to_string(ptxVersion % 10);

    // replace ptx version
    {
        size_t index = ptxCode.find("\n.version ");
        ptxCode = ptxCode.substr(0, index + 10) + ptx_version + ptxCode.substr(ptxCode.find('\n', index + 1));
    }

    // strip debug flag that prevent ptxas from optimizing the code
    {
        size_t index = ptxCode.find("\n.target ");
        size_t endLineIdx = ptxCode.find('\n', index + 1);

        size_t i;
        bool found = false;
        for (i = index + 9; i < endLineIdx; i++) {
            if (ptxCode[i] == ',') {
                found = true;
                break;
            }
        }
        if (found) {
            ptxCode = ptxCode.substr(0, i) + ptxCode.substr(endLineIdx);
        }
    }
    return {ptxCode, shared_mem_bytes};
}