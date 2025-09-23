#include "cmath_jit_llvm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>

using namespace llvm;
using namespace llvm::orc;

namespace {
    struct IR {
        std::vector<cm_instr> code;
        size_t num_slots = 0;
        uint32_t result = 0;
    };

    cm_jit_options default_opts() {
        cm_jit_options o{};
        o.opt_level = 3;
        o.enable_const_fold = 1;
        o.enable_cse = 1;
        o.enable_dce = 1;
        o.enable_auto_fma = 1;
        o.powi_limit = 8;
        o.vec_width_hint = 0;
        o.interleave_hint = 2;
        o.unroll_hint = 4;
        o.alignment = 16;
        o.prefetch_distance = 64;
        o.block_size = 0;
        o.assume_noalias = 1;
        o.nontemporal_store = 0;
        return o;
    }

    void initNativeTargetOnce() {
        static bool inited = false;
        if (!inited) {
            InitializeNativeTarget();
            InitializeNativeTargetAsmPrinter();
            InitializeNativeTargetDisassembler();
            inited = true;
        }
    }

    void setFastMath(IRBuilder<> &b) {
        FastMathFlags fmf;
        fmf.setFast();
        b.setFastMathFlags(fmf);
    }

    void setFMFOn(CallInst *ci) {
        if (!ci) return;
        FastMathFlags fmf;
        fmf.setFast();
        ci->setFastMathFlags(fmf);
    }

    struct Key {
        uint32_t op, a, b, c;
        int32_t aux;
        double imm;

        bool operator==(const Key &o) const {
            return op == o.op && a == o.a && b == o.b && c == o.c && aux == o.aux && imm == o.imm;
        }
    };

    struct KeyHash {
        size_t operator()(const Key &k) const noexcept {
            uint64_t uimm;
            std::memcpy(&uimm, &k.imm, sizeof(double));
            uint64_t h = ((uint64_t) k.op << 56) ^ ((uint64_t) k.a << 40) ^ ((uint64_t) k.b << 24) ^
                         ((uint64_t) k.c << 8) ^ ((uint64_t) (uint32_t) k.aux) ^ (uimm * 1315423911u);
            return static_cast<size_t>(h);
        }
    };

    Value *buildFMA(IRBuilder<> &b, Value *x, Value *y, Value *z) {
        Module *M = b.GetInsertBlock()->getModule();
        Function *fmaDecl = getOrInsertDeclaration(M, Intrinsic::fma, {b.getDoubleTy()});
        auto *ci = b.CreateCall(fmaDecl, {x, y, z}, "fma");
        setFMFOn(ci);
        return ci;
    }

    Value *buildFAbs(IRBuilder<> &b, Value *x) {
        Module *M = b.GetInsertBlock()->getModule();
        Function *fabsDecl = getOrInsertDeclaration(M, Intrinsic::fabs, {b.getDoubleTy()});
        auto *ci = b.CreateCall(fabsDecl, {x}, "fabs");
        setFMFOn(ci);
        return ci;
    }

    Value *buildPowi(IRBuilder<> &b, Value *x, int e) {
        auto *one = ConstantFP::get(b.getDoubleTy(), 1.0);
        if (e == 0) return one;
        if (e == 1) return x;
        if (e == -1) return b.CreateFDiv(one, x, "recip");
        const bool neg = e < 0;
        const auto k = static_cast<unsigned>(neg ? -e : e);
        Value *acc = nullptr;
        switch (k) {
            case 2: acc = b.CreateFMul(x, x, "sq");
                break;
            case 3: {
                auto *xx = b.CreateFMul(x, x, "xx");
                acc = b.CreateFMul(xx, x, "x3");
                break;
            }
            case 4: {
                auto *xx = b.CreateFMul(x, x, "xx");
                acc = b.CreateFMul(xx, xx, "x4");
                break;
            }
            default: {
                acc = b.CreateFMul(x, x, "p2");
                for (unsigned i = 2; i < k; ++i) acc = b.CreateFMul(acc, x, "p");
                break;
            }
        }
        if (neg) acc = b.CreateFDiv(one, acc, "recip_pow");
        return acc;
    }

    IR optimize_frontend(const cm_instr *in_code, size_t n,
                         size_t num_slots, uint32_t result_slot,
                         const cm_jit_options &opts);

    // common LLVM/JIT helpers
    std::unique_ptr<TargetMachine> makeTM(JITTargetMachineBuilder &jtmb, DataLayout &DL) {
        auto TMExp = jtmb.createTargetMachine();
        if (!TMExp) return nullptr;
        auto DLExp = jtmb.getDefaultDataLayoutForTarget();
        if (!DLExp) return nullptr;
        DL = *DLExp;
        return std::move(*TMExp);
    }

    int runO3(Module &M, TargetMachine *TM, int opt_level) {
        PassBuilder PB(TM);
        LoopAnalysisManager LAM;
        FunctionAnalysisManager FAM;
        CGSCCAnalysisManager CGAM;
        ModuleAnalysisManager MAM;
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
        OptimizationLevel OL = OptimizationLevel::O3;
        if (opt_level == 0) OL = OptimizationLevel::O0;
        else if (opt_level == 1) OL = OptimizationLevel::O1;
        else if (opt_level == 2) OL = OptimizationLevel::O2;
        ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OL);
        MPM.run(M, MAM);
        return 0;
    }

    MDNode *buildLoopMD(LLVMContext &C, int vecW, int ilv, int unroll) {
        MDBuilder MB(C);
        SmallVector<Metadata *, 8> MDs;
        MDs.push_back(MDNode::get(C, {})); // self-ref
        MDs.push_back(MDNode::get(C, {
                                      MB.createString("llvm.loop.vectorize.enable"),
                                      MB.createConstant(ConstantInt::get(Type::getInt1Ty(C), 1))
                                  }));
        if (vecW > 0) {
            MDs.push_back(MDNode::get(C, {
                                          MB.createString("llvm.loop.vectorize.width"),
                                          MB.createConstant(ConstantInt::get(Type::getInt32Ty(C), vecW))
                                      }));
        }
        if (ilv > 0) {
            MDs.push_back(MDNode::get(C, {
                                          MB.createString("llvm.loop.interleave.count"),
                                          MB.createConstant(ConstantInt::get(Type::getInt32Ty(C), ilv))
                                      }));
        }
        if (unroll > 0) {
            MDs.push_back(MDNode::get(C, {
                                          MB.createString("llvm.loop.unroll.count"),
                                          MB.createConstant(ConstantInt::get(Type::getInt32Ty(C), unroll))
                                      }));
            MDs.push_back(MDNode::get(C, {
                                          MB.createString("llvm.loop.unroll.enable"),
                                          MB.createConstant(ConstantInt::get(Type::getInt1Ty(C), 1))
                                      }));
        }
        auto *LoopID = MDNode::get(C, MDs);
        LoopID->replaceOperandWith(0, LoopID);
        return LoopID;
    }

    // scalar build
    int build_and_jit_scalar(const IR &ir, const cm_jit_options &opts,
                                    cm_jit_fn *out_fn, void **out_state) {
        initNativeTargetOnce();

        auto jtmbExp = JITTargetMachineBuilder::detectHost();
        if (!jtmbExp) return 101;
        JITTargetMachineBuilder jtmb = std::move(*jtmbExp);
        DataLayout DL;
        auto TM = makeTM(jtmb, DL);
        if (!TM) return 102;

        auto jitExp = LLJITBuilder().setJITTargetMachineBuilder(std::move(jtmb))
                .setDataLayout(DL).create();
        if (!jitExp) return 103;
        std::unique_ptr<LLJIT> jit = std::move(*jitExp);

        auto Ctx = std::make_unique<LLVMContext>();
        LLVMContext &C = *Ctx;
        auto M = std::make_unique<Module>("cm_module_scalar", C);
        M->setDataLayout(jit->getDataLayout());
        IRBuilder<> b(C);
        setFastMath(b);

        Type *f64 = b.getDoubleTy();
        Type *i64 = b.getInt64Ty();
        PointerType *ptrTy = PointerType::getUnqual(C); // opaque ptr

        FunctionType *FTy = FunctionType::get(f64, {ptrTy}, false);
        Function *F = Function::Create(FTy, Function::ExternalLinkage, "cm_entry", M.get());
        {
            F->addParamAttr(0, Attribute::NonNull);
            F->addParamAttr(0, Attribute::NoUndef);
            F->addFnAttr("no-nans-fp-math", "true");
            F->addFnAttr("no-infs-fp-math", "true");
            F->addFnAttr("approx-func", "true");
            F->addFnAttr("unsafe-fp-math", "true");
            F->addFnAttr("less-precise-fpmad", "true");
            F->getArg(0)->setName("vars");
        }

        BasicBlock *entry = BasicBlock::Create(C, "entry", F);
        b.SetInsertPoint(entry);
        Value *varsPtr = F->getArg(0);
        std::vector<Value *> slots(ir.num_slots, UndefValue::get(f64));
        auto getSlot = [&](uint32_t id) {
            assert(id<slots.size());
            return slots[id];
        };
        auto putSlot = [&](uint32_t id, Value *v) {
            assert(id<slots.size());
            slots[id] = v;
        };
        auto loadVar = [&](uint32_t idx)-> Value * {
            Value *vi = ConstantInt::get(i64, (uint64_t) idx);
            Value *gep = b.CreateInBoundsGEP(f64, varsPtr, vi, "var.ptr");
            return b.CreateLoad(f64, gep, "var");
        };
        auto getConst = [&](double d)-> Value * { return ConstantFP::get(f64, d); };
        Value *kOne = getConst(1.0);

        int powi_limit = (opts.powi_limit > 0 ? opts.powi_limit : 8);
        for (const auto &[op, dst, a, R, c, aux, imm]: ir.code) {
            switch (op) {
                case CM_OP_CONST: putSlot(dst, getConst(imm));
                    break;
                case CM_OP_VAR: putSlot(dst, loadVar(static_cast<uint32_t>(aux)));
                    break;
                case CM_OP_ADD: putSlot(dst, b.CreateFAdd(getSlot(a), getSlot(R), "add"));
                    break;
                case CM_OP_SUB: putSlot(dst, b.CreateFSub(getSlot(a), getSlot(R), "sub"));
                    break;
                case CM_OP_MUL: putSlot(dst, b.CreateFMul(getSlot(a), getSlot(R), "mul"));
                    break;
                case CM_OP_DIV: putSlot(dst, b.CreateFDiv(getSlot(a), getSlot(R), "div"));
                    break;
                case CM_OP_NEG: putSlot(dst, b.CreateFNeg(getSlot(a), "neg"));
                    break;
                case CM_OP_ABS: putSlot(dst, buildFAbs(b, getSlot(a)));
                    break;
                case CM_OP_SQRT: {
                    auto *sq = getOrInsertDeclaration(M.get(), Intrinsic::sqrt, {f64});
                    auto *ci = b.CreateCall(sq, {getSlot(a)}, "sqrt");
                    setFMFOn(ci);
                    putSlot(dst, ci);
                    break;
                }
                case CM_OP_ADD_K: putSlot(dst, b.CreateFAdd(getSlot(a), getConst(imm), "addk"));
                    break;
                case CM_OP_MUL_K: putSlot(dst, b.CreateFMul(getSlot(a), getConst(imm), "mulk"));
                    break;
                case CM_OP_RECIP: putSlot(dst, b.CreateFDiv(kOne, getSlot(a), "recip"));
                    break;
                case CM_OP_POWI: {
                    int e = aux;
                    if (e > powi_limit)e = powi_limit;
                    if (e < -powi_limit)e = -powi_limit;
                    putSlot(dst, buildPowi(b, getSlot(a), e));
                    break;
                }
                case CM_OP_FMA: putSlot(dst, buildFMA(b, getSlot(a), getSlot(R), getSlot(c)));
                    break;
                default: return 104;
            }
        }
        Value *result = (ir.result < slots.size() && slots[ir.result]) ? slots[ir.result] : ConstantFP::get(f64, 0.0);
        b.CreateRet(result);

#ifndef NDEBUG
        {
            std::string err;
            raw_string_ostream os(err);
            if (verifyModule(*M, &os)) return 105;
        }
#endif

        runO3(*M, TM.get(), opts.opt_level);

        ThreadSafeModule TSM(std::move(M), std::move(Ctx));
        if (auto e = jit->addIRModule(std::move(TSM))) return 106;

        auto sym = jit->lookup("cm_entry");
        if (!sym) return 107;
        cm_jit_fn fn = sym->toPtr<cm_jit_fn>();

        struct State {
            std::unique_ptr<LLJIT> jit;
        };
        auto *state = new State();
        state->jit = std::move(jit);
        *out_fn = fn;
        *out_state = state;
        return 0;
    }

    // batch build
    int build_and_jit_batch(const IR &ir, size_t num_vars, const cm_jit_options &opts,
                            cm_jit_fn_batch *out_fn, void **out_state) {
        initNativeTargetOnce();

        auto jtmbExp = JITTargetMachineBuilder::detectHost();
        if (!jtmbExp) return 201;
        JITTargetMachineBuilder jtmb = std::move(*jtmbExp);
        DataLayout DL;
        auto TM = makeTM(jtmb, DL);
        if (!TM) return 202;

        auto jitExp = LLJITBuilder().setJITTargetMachineBuilder(std::move(jtmb))
                .setDataLayout(DL).create();
        if (!jitExp) return 203;
        std::unique_ptr<LLJIT> jit = std::move(*jitExp);

        auto Ctx = std::make_unique<LLVMContext>();
        LLVMContext &C = *Ctx;
        auto M = std::make_unique<Module>("cm_module_batch", C);
        M->setDataLayout(jit->getDataLayout());
        IRBuilder<> b(C);
        setFastMath(b);

        Type *f64 = b.getDoubleTy();
        Type *i64 = b.getInt64Ty();
        PointerType *ptrTy = PointerType::getUnqual(C); // opaque ptr

        // void cm_entry_batch(const double* const* inputs, size_t n, double* out)
        FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), {ptrTy, Type::getInt64Ty(C), ptrTy}, false);
        Function *F = Function::Create(FTy, Function::ExternalLinkage, "cm_entry_batch", M.get());
        {
            if (opts.assume_noalias) {
                F->addParamAttr(0, Attribute::NonNull);
                F->addParamAttr(0, Attribute::NoUndef);
                F->addParamAttr(0, Attribute::ReadOnly);
                F->addParamAttr(0, Attribute::NoAlias);
                F->addParamAttr(2, Attribute::NonNull);
                F->addParamAttr(2, Attribute::NoUndef);
                F->addParamAttr(2, Attribute::WriteOnly);
                F->addParamAttr(2, Attribute::NoAlias);
            } else {
                F->addParamAttr(0, Attribute::NonNull);
                F->addParamAttr(2, Attribute::NonNull);
            }
            F->addFnAttr("no-nans-fp-math", "true");
            F->addFnAttr("no-infs-fp-math", "true");
            F->addFnAttr("approx-func", "true");
            F->addFnAttr("unsafe-fp-math", "true");
            F->addFnAttr("less-precise-fpmad", "true");
            F->getArg(0)->setName("inputs");
            F->getArg(1)->setName("n");
            F->getArg(2)->setName("out");
        }

        BasicBlock *entry = BasicBlock::Create(C, "entry", F);
        BasicBlock *loopHdr = BasicBlock::Create(C, "loop.hdr", F);
        BasicBlock *loopBody = BasicBlock::Create(C, "loop.body", F);
        BasicBlock *loopExit = BasicBlock::Create(C, "loop.exit", F);

        b.SetInsertPoint(entry);
        Value *inputs = F->getArg(0);
        Value *n = F->getArg(1);
        Value *outPtr = F->getArg(2);

        // Load per-variable base pointers (inputs[j])
        std::vector<Value *> inBases(num_vars, nullptr);
        for (size_t j = 0; j < num_vars; ++j) {
            Value *idx = ConstantInt::get(i64, (uint64_t) j);
            Value *gep = b.CreateInBoundsGEP(ptrTy, inputs, idx, "in.ptrptr");
            LoadInst *ldp = b.CreateLoad(ptrTy, gep, "in.base");
            if (opts.alignment >= 8) ldp->setAlignment(Align(8));
            inBases[j] = ldp;
        }

        b.CreateBr(loopHdr);

        // i loop header
        b.SetInsertPoint(loopHdr);
        PHINode *i = b.CreatePHI(i64, 2, "i");
        i->addIncoming(ConstantInt::get(i64, 0), entry);
        Value *cond = b.CreateICmpULT(i, n, "cond");
        auto *br = b.CreateCondBr(cond, loopBody, loopExit);
        {
            MDNode *LoopMD = buildLoopMD(C, opts.vec_width_hint, opts.interleave_hint, opts.unroll_hint);
            br->setMetadata(LLVMContext::MD_loop, LoopMD);
        }

        // loop body
        b.SetInsertPoint(loopBody);

        // Optional prefetch
        if (opts.prefetch_distance > 0) {
            int PD = opts.prefetch_distance;
            Value *i_pref = b.CreateAdd(i, ConstantInt::get(i64, (uint64_t) PD), "i.pref");
            for (size_t j = 0; j < num_vars; ++j) {
                Value *addr = b.CreateInBoundsGEP(f64, inBases[j], i_pref);
                Module *Mod = M.get();
                Function *PFI = getOrInsertDeclaration(Mod, Intrinsic::prefetch, {ptrTy});
                Value *p = b.CreateBitCast(addr, ptrTy);
                b.CreateCall(PFI, {
                                 p,
                                 ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 3),
                                 ConstantInt::get(Type::getInt32Ty(C), 1)
                             });
            }
        }

        // Load current element
        std::vector<Value *> curVars(num_vars, nullptr);
        for (size_t j = 0; j < num_vars; ++j) {
            Value *addr = b.CreateInBoundsGEP(f64, inBases[j], i, "in.elem");
            LoadInst *ld = b.CreateLoad(f64, addr, "var");
            if (opts.alignment >= 8) ld->setAlignment(Align((unsigned) opts.alignment));
            curVars[j] = ld;
        }

        // Evaluate expression
        std::vector<Value *> slots(ir.num_slots, UndefValue::get(f64));
        auto getSlot = [&](uint32_t id) {
            assert(id<slots.size());
            return slots[id];
        };
        auto putSlot = [&](uint32_t id, Value *v) {
            assert(id<slots.size());
            slots[id] = v;
        };
        auto getConst = [&](double d)-> Value * { return ConstantFP::get(f64, d); };
        Value *kOne = getConst(1.0);

        int powi_limit = (opts.powi_limit > 0 ? opts.powi_limit : 8);
        for (const auto &ins: ir.code) {
            switch (ins.op) {
                case CM_OP_CONST: putSlot(ins.dst, getConst(ins.imm));
                    break;
                case CM_OP_VAR: putSlot(ins.dst, curVars[(size_t) ins.aux]);
                    break;
                case CM_OP_ADD: putSlot(ins.dst, b.CreateFAdd(getSlot(ins.a), getSlot(ins.b), "add"));
                    break;
                case CM_OP_SUB: putSlot(ins.dst, b.CreateFSub(getSlot(ins.a), getSlot(ins.b), "sub"));
                    break;
                case CM_OP_MUL: putSlot(ins.dst, b.CreateFMul(getSlot(ins.a), getSlot(ins.b), "mul"));
                    break;
                case CM_OP_DIV: putSlot(ins.dst, b.CreateFDiv(getSlot(ins.a), getSlot(ins.b), "div"));
                    break;
                case CM_OP_NEG: putSlot(ins.dst, b.CreateFNeg(getSlot(ins.a), "neg"));
                    break;
                case CM_OP_ABS: putSlot(ins.dst, buildFAbs(b, getSlot(ins.a)));
                    break;
                case CM_OP_SQRT: {
                    auto *sq = getOrInsertDeclaration(M.get(), Intrinsic::sqrt, {f64});
                    auto *ci = b.CreateCall(sq, {getSlot(ins.a)}, "sqrt");
                    setFMFOn(ci);
                    putSlot(ins.dst, ci);
                    break;
                }
                case CM_OP_ADD_K: putSlot(ins.dst, b.CreateFAdd(getSlot(ins.a), getConst(ins.imm), "addk"));
                    break;
                case CM_OP_MUL_K: putSlot(ins.dst, b.CreateFMul(getSlot(ins.a), getConst(ins.imm), "mulk"));
                    break;
                case CM_OP_RECIP: putSlot(ins.dst, b.CreateFDiv(kOne, getSlot(ins.a), "recip"));
                    break;
                case CM_OP_POWI: {
                    int e = ins.aux;
                    if (e > powi_limit)e = powi_limit;
                    if (e < -powi_limit)e = -powi_limit;
                    putSlot(ins.dst, buildPowi(b, getSlot(ins.a), e));
                    break;
                }
                case CM_OP_FMA: putSlot(ins.dst, buildFMA(b, getSlot(ins.a), getSlot(ins.b), getSlot(ins.c)));
                    break;
                default: return 204;
            }
        }
        Value *res = (ir.result < slots.size() && slots[ir.result]) ? slots[ir.result] : getConst(0.0);
        Value *outAddr = b.CreateInBoundsGEP(f64, outPtr, i, "out.elem");
        StoreInst *storeInst = b.CreateStore(res, outAddr);
        if (opts.alignment >= 8) storeInst->setAlignment(Align((unsigned) opts.alignment));
        if (opts.nontemporal_store) {
            LLVMContext &CtxRef = C;
            Metadata *One = ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(CtxRef), 1));
            storeInst->setMetadata(LLVMContext::MD_nontemporal, MDNode::get(CtxRef, One));
        }

        // i++
        Value *inext = b.CreateAdd(i, ConstantInt::get(i64, 1), "i.next");
        i->addIncoming(inext, loopBody);
        b.CreateBr(loopHdr);

        // epilogue
        b.SetInsertPoint(loopExit);
        b.CreateRetVoid();

#ifndef NDEBUG
        {
            std::string err;
            raw_string_ostream os(err);
            if (verifyModule(*M, &os)) return 205;
        }
#endif

        runO3(*M, TM.get(), opts.opt_level);

        ThreadSafeModule TSM(std::move(M), std::move(Ctx));
        if (auto e = jit->addIRModule(std::move(TSM))) return 206;

        auto sym = jit->lookup("cm_entry_batch");
        if (!sym) return 207;
        cm_jit_fn_batch fn = sym->toPtr<cm_jit_fn_batch>();

        struct State {
            std::unique_ptr<LLJIT> jit;
        };
        auto *state = new State();
        state->jit = std::move(jit);
        *out_fn = fn;
        *out_state = state;
        return 0;
    }

    // front-end optimizer
    IR optimize_frontend(const cm_instr *in_code, size_t n,
                         size_t num_slots, uint32_t result_slot,
                         const cm_jit_options &opts) {
        IR ir;
        ir.code.assign(in_code, in_code + n);
        ir.num_slots = num_slots;
        ir.result = result_slot;
        if (ir.code.empty()) return ir;

        auto rebuild_use = [&](std::vector<uint32_t> &use) {
            use.assign(ir.num_slots, 0);
            for (const auto &ins: ir.code) {
                switch (ins.op) {
                    case CM_OP_ADD:
                    case CM_OP_SUB:
                    case CM_OP_MUL:
                    case CM_OP_DIV: ++use[ins.a];
                        ++use[ins.b];
                        break;
                    case CM_OP_NEG:
                    case CM_OP_SQRT:
                    case CM_OP_RECIP:
                    case CM_OP_POWI:
                    case CM_OP_ADD_K:
                    case CM_OP_MUL_K:
                    case CM_OP_ABS: ++use[ins.a];
                        break;
                    case CM_OP_FMA: ++use[ins.a];
                        ++use[ins.b];
                        ++use[ins.c];
                        break;
                    default: break;
                }
            }
        };
        std::vector<uint32_t> use;
        rebuild_use(use);

        // Track constants
        std::vector<uint8_t> is_const(ir.num_slots, 0);
        std::vector<double> const_val(ir.num_slots, 0.0);
        for (const auto &ins: ir.code)
            if (ins.op == CM_OP_CONST) {
                is_const[ins.dst] = 1;
                const_val[ins.dst] = ins.imm;
            }

        // Const folding
        std::vector<cm_instr> folded;
        folded.reserve(ir.code.size());
        for (const auto &ins: ir.code) {
            cm_instr out = ins;
            bool changed = false;
            auto setK = [&](double v) {
                out.op = CM_OP_CONST;
                out.a = out.b = out.c = 0;
                out.aux = 0;
                out.imm = v;
                is_const[out.dst] = 1;
                const_val[out.dst] = v;
                changed = true;
            };

            switch (ins.op) {
                case CM_OP_ADD:
                case CM_OP_SUB:
                case CM_OP_MUL:
                case CM_OP_DIV:
                    if (opts.enable_const_fold && is_const[ins.a] && is_const[ins.b]) {
                        double A = const_val[ins.a], B = const_val[ins.b], R = 0;
                        if (ins.op == CM_OP_ADD) R = A + B;
                        else if (ins.op == CM_OP_SUB) R = A - B;
                        else if (ins.op == CM_OP_MUL) R = A * B;
                        else R = A / B;
                        setK(R);
                    }
                    break;
                case CM_OP_NEG:
                case CM_OP_RECIP:
                case CM_OP_SQRT:
                case CM_OP_ABS:
                    if (opts.enable_const_fold && is_const[ins.a]) {
                        double A = const_val[ins.a], R = 0;
                        if (ins.op == CM_OP_NEG) R = -A;
                        else if (ins.op == CM_OP_RECIP) R = 1.0 / A;
                        else if (ins.op == CM_OP_SQRT) R = std::sqrt(A);
                        else R = std::fabs(A);
                        setK(R);
                    }
                    break;
                case CM_OP_ADD_K:
                    if (opts.enable_const_fold && is_const[ins.a]) setK(const_val[ins.a] + ins.imm);
                    break;
                case CM_OP_MUL_K:
                    if (opts.enable_const_fold && is_const[ins.a]) setK(const_val[ins.a] * ins.imm);
                    break;
                case CM_OP_POWI:
                    if (opts.enable_const_fold && is_const[ins.a]) {
                        int e = ins.aux;
                        e = std::max(-opts.powi_limit, std::min(opts.powi_limit, e));
                        double x = const_val[ins.a], r = 0;
                        if (e == 0) r = 1.0;
                        else if (e == 1) r = x;
                        else if (e == -1) r = 1.0 / x;
                        else {
                            int neg = e < 0;
                            auto k = static_cast<unsigned>(neg ? -e : e);
                            switch (k) {
                                case 2: r = x * x;
                                    break;
                                case 3: r = (x * x) * x;
                                    break;
                                case 4: {
                                    double xx = x * x;
                                    r = xx * xx;
                                    break;
                                }
                                default: r = x * x;
                                    for (unsigned i = 2; i < k; ++i)r *= x;
                                    break;
                            }
                            if (neg) r = 1.0 / r;
                        }
                        setK(r);
                    }
                    break;
                default: break;
            }
            if (!changed) {
                if (out.op == CM_OP_CONST) {
                    is_const[out.dst] = 1;
                    const_val[out.dst] = out.imm;
                } else is_const[out.dst] = 0;
            }
            folded.push_back(out);
        }
        ir.code.swap(folded);
        rebuild_use(use);

        // Peephole FMA fusion
        if (opts.enable_auto_fma) {
            std::vector<cm_instr> ph;
            ph.reserve(ir.code.size());
            std::vector<int> mul_idx(ir.num_slots, -1);
            for (int i = 0; i < (int) ir.code.size(); ++i) if (ir.code[i].op == CM_OP_MUL) mul_idx[ir.code[i].dst] = i;
            for (int i = 0; i < (int) ir.code.size(); ++i) {
                const auto &ins = ir.code[i];
                if ((ins.op == CM_OP_ADD || ins.op == CM_OP_SUB) && (mul_idx[ins.a] >= 0 || mul_idx[ins.b] >= 0)) {
                    uint32_t t = (mul_idx[ins.a] >= 0) ? ins.a : ins.b;
                    int midx = mul_idx[t];
                    if (midx >= 0 && use[t] == 1) {
                        const auto &mul = ir.code[midx];
                        uint32_t other = (t == ins.a) ? ins.b : ins.a;
                        cm_instr f{};
                        f.op = CM_OP_FMA;
                        f.dst = ins.dst;
                        f.a = mul.a;
                        f.b = mul.b;
                        f.c = other;
                        if (ins.op == CM_OP_SUB && t == ins.a) {
                            cm_instr neg{};
                            neg.op = CM_OP_NEG;
                            neg.dst = mul.dst;
                            neg.a = other;
                            ph.push_back(neg);
                            f.c = neg.dst;
                        } else if (ins.op == CM_OP_SUB && t == ins.b) {
                            cm_instr nb{};
                            nb.op = CM_OP_NEG;
                            nb.dst = mul.dst;
                            nb.a = mul.a;
                            ph.push_back(nb);
                            f.a = nb.dst;
                            f.c = other;
                        }
                        ph.push_back(f);
                        use[mul.dst] = 0;
                        continue;
                    }
                }
                ph.push_back(ins);
            }
            ir.code.swap(ph);
            rebuild_use(use);
        }

        // Peephole: sqrt(x*x) -> abs(x)
        {
            // Def map
            std::vector def(ir.num_slots, -1);
            for (int i = 0; i < static_cast<int>(ir.code.size()); ++i) def[ir.code[i].dst] = i;

            for (auto &ins: ir.code) {
                if (ins.op == CM_OP_SQRT) {
                    uint32_t s = ins.a;
                    int di = (s < def.size()) ? def[s] : -1;
                    if (di >= 0) {
                        const auto &mul = ir.code[di];
                        if (mul.op == CM_OP_MUL && mul.a == mul.b) {
                            ins.op = CM_OP_ABS;
                            ins.a = mul.a;
                            ins.b = ins.c = 0;
                            ins.aux = 0;
                            ins.imm = 0.0;
                        }
                    }
                }
            }
            rebuild_use(use);
        }

        // CSE
        if (opts.enable_cse) {
            std::unordered_map<Key, uint32_t, KeyHash> memo;
            std::vector<cm_instr> out;
            out.reserve(ir.code.size());
            for (auto ins: ir.code) {
                if (ins.op == CM_OP_ADD || ins.op == CM_OP_MUL) if (ins.a > ins.b) std::swap(ins.a, ins.b);
                Key k{ins.op, ins.a, ins.b, ins.c, ins.aux, ins.imm};
                auto it = memo.find(k);
                if (it != memo.end()) {
                    ins.op = CM_OP_ADD_K;
                    ins.a = it->second;
                    ins.b = ins.c = 0;
                    ins.aux = 0;
                    ins.imm = 0.0;
                    out.push_back(ins);
                    continue;
                }
                memo.emplace(k, ins.dst);
                out.push_back(ins);
            }
            ir.code.swap(out);
        }

        // DCE
        if (opts.enable_dce) {
            std::vector<uint8_t> live(ir.num_slots, 0);
            if (ir.result < live.size()) live[ir.result] = 1;
            for (int i = (int) ir.code.size() - 1; i >= 0; --i) {
                const auto &ins = ir.code[i];
                if (ins.dst < live.size() && live[ins.dst]) {
                    switch (ins.op) {
                        case CM_OP_ADD:
                        case CM_OP_SUB:
                        case CM_OP_MUL:
                        case CM_OP_DIV: live[ins.a] = live[ins.b] = 1;
                            break;
                        case CM_OP_NEG:
                        case CM_OP_SQRT:
                        case CM_OP_RECIP:
                        case CM_OP_POWI:
                        case CM_OP_ADD_K:
                        case CM_OP_MUL_K:
                        case CM_OP_ABS: live[ins.a] = 1;
                            break;
                        case CM_OP_FMA: live[ins.a] = live[ins.b] = live[ins.c] = 1;
                            break;
                        default: break;
                    }
                }
            }
            std::vector<cm_instr> kept;
            kept.reserve(ir.code.size());
            for (const auto &ins: ir.code) if (ins.dst < live.size() && live[ins.dst]) kept.push_back(ins);
            ir.code.swap(kept);
        }

        return ir;
    }
} // namespace

extern "C" {
int cm_llvm_jit_supported(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    return 1;
#else
    return 0;
#endif
}

int cm_llvm_jit_compile(const cm_instr *code,
                        size_t n_insts,
                        size_t num_vars,
                        size_t num_slots,
                        uint32_t result_slot,
                        cm_jit_fn *out_fn,
                        void **out_state) {
    cm_jit_options o = default_opts();
    return cm_llvm_jit_compile_ex(code, n_insts, num_vars, num_slots, result_slot, &o, out_fn, out_state);
}

int cm_llvm_jit_compile_ex(const cm_instr *code,
                           size_t n_insts,
                           size_t num_vars,
                           size_t num_slots,
                           uint32_t result_slot,
                           const cm_jit_options *opts,
                           cm_jit_fn *out_fn,
                           void **out_state) {
    if (!code || !out_fn || !out_state) return 1;
    if (!n_insts || !num_slots || result_slot >= num_slots) return 2;
    cm_jit_options o = opts ? *opts : default_opts();
    IR ir = optimize_frontend(code, n_insts, num_slots, result_slot, o);
    return build_and_jit_scalar(ir, o, out_fn, out_state);
}

int cm_llvm_jit_compile_batch(const cm_instr *code,
                              size_t n_insts,
                              size_t num_vars,
                              size_t num_slots,
                              uint32_t result_slot,
                              cm_jit_fn_batch *out_fn,
                              void **out_state) {
    cm_jit_options o = default_opts();
    return cm_llvm_jit_compile_batch_ex(code, n_insts, num_vars, num_slots, result_slot, &o, out_fn, out_state);
}

int cm_llvm_jit_compile_batch_ex(const cm_instr *code,
                                 size_t n_insts,
                                 size_t num_vars,
                                 size_t num_slots,
                                 uint32_t result_slot,
                                 const cm_jit_options *opts,
                                 cm_jit_fn_batch *out_fn,
                                 void **out_state) {
    if (!code || !out_fn || !out_state) return 1;
    if (!n_insts || !num_slots || result_slot >= num_slots || !num_vars) return 2;
    cm_jit_options o = opts ? *opts : default_opts();
    IR ir = optimize_frontend(code, n_insts, num_slots, result_slot, o);
    return build_and_jit_batch(ir, num_vars, o, out_fn, out_state);
}

void cm_llvm_jit_release(void *opaque) {
    if (!opaque) return;
    struct State {
        std::unique_ptr<LLJIT> jit;
    };
    const auto *st = static_cast<State *>(opaque);
    delete st;
}
} // extern "C"
