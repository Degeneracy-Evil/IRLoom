// Read-only structural audit against the same SDK as the historical bridge.
#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "polygeist/Dialect.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
using namespace mlir;
namespace json = llvm::json;
struct MemRefElement : MemRefElementTypeInterface::FallbackModel<MemRefElement> {};
template <typename T>
struct PointerElement : LLVM::PointerElementTypeInterface::ExternalModel<PointerElement<T>, T> {};

static void visit(Operation *op, std::string owner, SmallVector<int64_t> path,
                  json::Array &loops) {
  if (isa<FunctionOpInterface>(op)) {
    owner = "function:" + SymbolTable::getSymbolName(op).str();
    path.clear();
  }
  if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
    json::Array positions, duplicates, aliases, types;
    for (int64_t p : path) positions.push_back(p);
    auto inits = loop.getInits();
    for (auto item : llvm::enumerate(inits)) {
      for (unsigned j = 0; j < item.index(); ++j)
        if (item.value() == inits[j]) duplicates.push_back(json::Array{int64_t(j), int64_t(item.index())});
      if (llvm::is_contained(loop.getControlOperands(), item.value())) aliases.push_back(int64_t(item.index()));
      std::string type;
      llvm::raw_string_ostream stream(type); item.value().getType().print(stream);
      types.push_back(type);
    }
    auto trip = affine::getConstantTripCount(loop);
    bool eligible = !trip || *trip >= 2;
    bool cleanup = eligible && !llvm::hasSingleElement(loop.getBody()->getOperations()) &&
                   affine::getLargestDivisorOfTripCount(loop) % 2 != 0;
    bool risk = cleanup && (!duplicates.empty() || !aliases.empty());
    json::Object row{{"descriptor", json::Object{{"owner", owner}, {"path", std::move(positions)}, {"kind", "affine.for"}}},
                     {"init_count", int64_t(inits.size())}, {"init_types", std::move(types)},
                     {"duplicate_init_pairs", std::move(duplicates)}, {"init_bound_alias_positions", std::move(aliases)},
                     {"needs_cleanup", cleanup}, {"legacy_cleanup_risk", risk}};
    if (risk) {
      std::string text; llvm::raw_string_ostream stream(text); loop.print(stream);
      row["loop_ir"] = text;
    }
    loops.push_back(std::move(row));
  }
  int64_t r = 0;
  for (Region &region : op->getRegions()) {
    int64_t b = 0;
    for (Block &block : region) {
      int64_t i = 0;
      for (Operation &child : block) {
        auto next = path; next.append({r, b, i++});
        visit(&child, owner, next, loops);
      }
      ++b;
    }
    ++r;
  }
}
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  DialectRegistry registry; registerAllDialects(registry);
  registry.insert<polygeist::PolygeistDialect>();
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *) {
    LLVM::LLVMFunctionType::attachInterface<MemRefElement>(*ctx);
    LLVM::LLVMArrayType::attachInterface<MemRefElement>(*ctx);
    LLVM::LLVMPointerType::attachInterface<MemRefElement>(*ctx);
    LLVM::LLVMStructType::attachInterface<MemRefElement>(*ctx);
    LLVM::LLVMStructType::attachInterface<PointerElement<LLVM::LLVMStructType>>(*ctx);
    LLVM::LLVMPointerType::attachInterface<PointerElement<LLVM::LLVMPointerType>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *) {
    MemRefType::attachInterface<PointerElement<MemRefType>>(*ctx);
  });
  MLIRContext context(registry); context.disableMultithreading();
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module || failed(verify(*module))) return 3;
  json::Array loops; visit(module->getOperation(), "module", {}, loops);
  std::error_code error; llvm::raw_fd_ostream out(argv[2], error);
  if (error) return 2;
  out << llvm::formatv("{0:2}", json::Value(json::Object{{"loops", std::move(loops)}}));
}
