#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "llvm/Support/MemoryBuffer.h"
#ifdef IRLOOM_POLYGEIST
#include "polygeist/Dialect.h"
#endif

#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include <map>

using namespace mlir;
namespace json = llvm::json;
using llvm::StringRef;
using llvm::raw_string_ostream;
using llvm::raw_fd_ostream;
using llvm::errs;
using llvm::formatv;

static bool legalInlineBody(InlinerInterface &interface, Region &source,
                            Region *destination, IRMapping &mapping) {
  for (Block &block : source)
    for (Operation &operation : block) {
      if (!interface.isLegalToInline(&operation, destination, true, mapping)) return false;
      if (interface.shouldAnalyzeRecursively(&operation))
        for (Region &nested : operation.getRegions())
          if (!legalInlineBody(interface, nested, destination, mapping)) return false;
    }
  return true;
}

#ifdef IRLOOM_POLYGEIST
// Same external type models as the pinned official polygeist-opt driver.
struct MemRefElement : MemRefElementTypeInterface::FallbackModel<MemRefElement> {};
template <typename T>
struct PointerElement : LLVM::PointerElementTypeInterface::ExternalModel<PointerElement<T>, T> {};
#endif

// All IDs are local to a single parsed snapshot. No pointer is serialized.
struct Graph {
  json::Array nodes, edges, candidates;
  DenseMap<Operation *, int64_t> ops;
  DenseMap<Block *, int64_t> blocks;
  DenseMap<Value, int64_t> values;
  int64_t node(StringRef kind, StringRef name, StringRef anchor = "") {
    int64_t id = nodes.size();
    nodes.push_back(json::Object{{"id", id}, {"kind", kind.str()}, {"name", name.str()},
                                 {"anchor", anchor.str()}, {"features", json::Array{}}});
    return id;
  }
  void edge(int64_t src, int64_t dst, StringRef kind) {
    edges.push_back(json::Array{src, dst, kind});
  }
  int64_t value(Value v) {
    if (values.count(v)) return values[v];
    auto id = node("value", "value");
    values[v] = id;
    auto *obj = nodes[id].getAsObject();
    std::string type;
    raw_string_ostream stream(type);
    v.getType().print(stream);
    (*obj)["type"] = type;
    json::Array features;
    if (auto shaped = dyn_cast<ShapedType>(v.getType())) {
      features.push_back(shaped.hasRank() ? shaped.getRank() : -1);
      if (shaped.hasRank())
        for (auto dim : shaped.getShape()) features.push_back(dim);
    }
    (*obj)["features"] = std::move(features);
    return id;
  }
  void visit(Operation *op, int64_t parent = -1) {
    std::string kind = "operation", anchor;
    if (isa<ModuleOp>(op)) { kind = "module"; anchor = "module"; }
    if (isa<FunctionOpInterface>(op)) {
      kind = "function"; anchor = "function:" + SymbolTable::getSymbolName(op).str();
    }
    auto id = node(kind, op->getName().getStringRef(), anchor);
    ops[op] = id;
    (*nodes[id].getAsObject())["features"] = json::Array{
        int64_t(op->getNumOperands()), int64_t(op->getNumResults()),
        int64_t(op->getNumRegions()), int64_t(isMemoryEffectFree(op))};
    auto &features = *nodes[id].getAsObject()->getArray("features");
    if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
      features.push_back(loop.hasConstantBounds() ? loop.getConstantLowerBound() : -1);
      features.push_back(loop.hasConstantBounds() ? loop.getConstantUpperBound() : -1);
      features.push_back(loop->getAttrOfType<IntegerAttr>("step").getInt());
    }
    if (auto integer = op->getAttrOfType<IntegerAttr>("value"))
      if (integer.getValue().getBitWidth() <= 64) features.push_back(integer.getInt());
    if (auto number = op->getAttrOfType<FloatAttr>("value")) features.push_back(number.getValueAsDouble());
    // Structural attributes come from typed IR, never text matching. Do not
    // feed symbol/benchmark names or dense constant payloads to the network.
    std::string attributes;
    raw_string_ostream attributeStream(attributes);
    for (StringRef key : {"indexing_maps", "iterator_types", "map", "lower_bound", "upper_bound"})
      if (Attribute attribute = op->getAttr(key)) attributeStream << key << '=' << attribute << ';';
    (*nodes[id].getAsObject())["attributes"] = attributes;
    (*nodes[id].getAsObject())["memory"] = isMemoryEffectFree(op) ? "none" : "conservative-unknown-alias";
    if (parent >= 0) edge(parent, id, "contains");
    for (Value v : op->getResults()) edge(id, value(v), "defines");
    for (Region &region : op->getRegions()) {
      // Only the function body and module region survive this adapter's lowering
      // by contract. Nested region identity is deliberately snapshot-local.
      std::string ra = anchor.empty() ? "" : anchor + ":body";
      auto rid = node("region", "region", ra);
      edge(id, rid, "contains");
      for (Block &block : region) {
        auto bid = node("block", "block");
        blocks[&block] = bid; edge(rid, bid, "contains");
        for (Value v : block.getArguments()) edge(bid, value(v), "argument");
        for (Operation &nested : block) visit(&nested, bid);
      }
    }
  }
  void finish(ModuleOp module, StringRef stage) {
    module.walk([&](Operation *op) {
      auto id = ops.lookup(op);
      for (Value v : op->getOperands()) edge(value(v), id, "uses");
      for (Block *b : op->getSuccessors()) edge(id, blocks.lookup(b), "cfg");
      if (auto call = dyn_cast<CallOpInterface>(op)) {
        Operation *callee = call.resolveCallable();
        if (callee && ops.count(callee)) edge(id, ops.lookup(callee), "call");
        (*nodes[id].getAsObject())["call"] = callee ? "resolved-symbol" : "external-or-indirect-unknown";
      }
      if (!isMemoryEffectFree(op)) edge(0, id, "conservative-memory");
    });
    candidates.push_back(json::Object{{"kind", "advance"}, {"target", int64_t(0)},
                                      {"parameters", json::Object{}}, {"precondition", "verified module"}});
    module.walk([&](Operation *op) {
      if (stage == "S0") {
#ifdef IRLOOM_POLYGEIST
        if (auto call = dyn_cast<func::CallOp>(op)) {
          auto *calleeOp = cast<CallOpInterface>(op).resolveCallable();
          auto callee = dyn_cast_or_null<func::FuncOp>(calleeOp);
          InlinerInterface interface(module.getContext());
          IRMapping mapping;
          if (callee && !callee.isExternal() && callee.getNumArguments() == call.getNumOperands())
            mapping.map(callee.getArguments(), call.getOperands());
          if (callee && !callee.isExternal() && callee != op->getParentOfType<func::FuncOp>() &&
              interface.isLegalToInline(op, calleeOp, true) &&
              interface.isLegalToInline(op->getParentRegion(), &callee.getBody(), true, mapping) &&
              legalInlineBody(interface, callee.getBody(), op->getParentRegion(), mapping))
            candidates.push_back(json::Object{{"kind", "inline"}, {"target", ops.lookup(op)},
                {"parameters", json::Object{}}, {"precondition", "defined non-self call; dialect legality; trial rechecks body"}});
        }
#endif
        if (auto generic = dyn_cast<linalg::GenericOp>(op))
          for (OpOperand &operand : generic->getOpOperands())
            if (isa_and_nonnull<linalg::GenericOp>(operand.get().getDefiningOp()) &&
                linalg::areElementwiseOpsFusable(&operand))
              candidates.push_back(json::Object{{"kind", "fuse"}, {"target", ops.lookup(op)},
                  {"parameters", json::Object{{"operand", int64_t(operand.getOperandNumber())}}},
                  {"precondition", "areElementwiseOpsFusable"}});
      }
      if (stage == "S1") {
        if (auto loop = dyn_cast<affine::AffineForOp>(op))
          if (!loop.hasConstantBounds() || loop.getConstantUpperBound() - loop.getConstantLowerBound() >= 2 * loop->getAttrOfType<IntegerAttr>("step").getInt())
            candidates.push_back(json::Object{{"kind", "unroll2"}, {"target", ops.lookup(op)},
                {"parameters", json::Object{}}, {"precondition", "affine loop; dynamic cleanup allowed; trial checks loopUnrollByFactor"}});
      }
    });
  }
};

static LogicalResult apply(ModuleOp module, Graph &graph, json::Object &action) {
  auto kind = action.getString("kind");
  auto target = action.getInteger("target");
  auto *params = action.getObject("parameters");
  if (!kind || !target || !params) return failure();
  Operation *op = nullptr;
  for (auto item : graph.ops) if (item.second == *target) op = item.first;
  if (!op) return failure();
  // Recheck the entire registered candidate, including its parameters.
  bool legal = false;
  for (auto &candidate : graph.candidates) {
    auto *c = candidate.getAsObject();
    if (c->getString("kind") == kind && c->getInteger("target") == target &&
        json::Value(json::Object(*c->getObject("parameters"))) == json::Value(json::Object(*params))) legal = true;
  }
  if (!legal) return failure();
  if (*kind == "advance") return success();
  if (*kind == "inline") {
    auto call = dyn_cast<CallOpInterface>(op);
    if (!call) return failure();
    auto callable = dyn_cast_or_null<CallableOpInterface>(call.resolveCallable());
    if (!callable || !callable.getCallableRegion()) return failure();
    InlinerInterface interface(module.getContext());
    if (failed(inlineCall(interface, call, callable, callable.getCallableRegion(), true))) return failure();
    op->erase();
    return success();
  }
  IRRewriter rewriter(module.getContext());
  if (*kind == "fuse") {
    auto generic = cast<linalg::GenericOp>(op);
    auto &operand = generic->getOpOperand(*params->getInteger("operand"));
    Operation *producer = operand.get().getDefiningOp();
    auto producerId = graph.ops.lookup(producer);
    rewriter.setInsertionPoint(generic);
    auto fused = linalg::fuseElementwiseOps(rewriter, &operand);
    if (failed(fused)) return failure();
    fused->fusedOp->setAttr("irloom.origins", rewriter.getI64ArrayAttr({producerId, *target}));
    for (auto [oldValue, replacement] : fused->replacements)
      rewriter.replaceUsesWithIf(oldValue, replacement, [&](OpOperand &use) {
        return use.get().getDefiningOp() != producer;
      });
    rewriter.eraseOp(generic);
    if (producer->use_empty() && isMemoryEffectFree(producer)) rewriter.eraseOp(producer);
    return success();
  }
  if (*kind == "unroll2") return affine::loopUnrollByFactor(cast<affine::AffineForOp>(op), 2);
  return failure();
}

int main(int argc, char **argv) {
  if (argc != 6 && argc != 8) {
    errs() << "usage: irloom-bridge input.mlir graph.json snapshot branch stage [action.json output.mlir]\n";
    return 2;
  }
  DialectRegistry registry;
  registerAllDialects(registry);
  func::registerInlinerExtension(registry);
#ifdef IRLOOM_POLYGEIST
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
#endif
  MLIRContext context(registry);
  context.disableMultithreading();
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module || failed(verify(*module))) return 3;
  Graph graph; graph.visit(module->getOperation()); graph.finish(*module, argv[5]);
  json::Array lineage;
  if (argc == 8) {
    auto buffer = llvm::MemoryBuffer::getFile(argv[6]);
    if (!buffer) return 2;
    auto request = json::parse((*buffer)->getBuffer());
    if (!request || !request->getAsObject()) return 2;
    auto &action = *request->getAsObject();
    if (action.getString("snapshot") != StringRef(argv[3])) { errs() << "stale snapshot\n"; return 4; }
    Builder builder(&context);
    // Temporary provenance exists only during this controlled rewrite. It is
    // removed before saving IR; arbitrary lowering never gets this guarantee.
    for (auto item : graph.ops)
      item.first->setAttr("irloom.origins", builder.getI64ArrayAttr({item.second}));
    if (failed(apply(*module, graph, action))) { errs() << "illegal action\n"; return 4; }
    if (action.getBoolean("inject_failure").value_or(false)) { errs() << "injected failure after mutation\n"; return 5; }
    if (failed(verify(*module))) return 3;
    Graph after; after.visit(module->getOperation()); after.finish(*module, argv[5]);
    for (auto item : after.ops) {
      if (auto origins = item.first->getAttrOfType<ArrayAttr>("irloom.origins")) {
        json::Array old;
        for (auto a : origins) old.push_back(cast<IntegerAttr>(a).getInt());
        lineage.push_back(json::Object{{"old", std::move(old)}, {"new", json::Array{item.second}},
                                      {"status", "known"}, {"scope", "operation"}});
      }
      item.first->removeAttr("irloom.origins");
    }
    std::error_code ec;
    raw_fd_ostream ir(argv[7], ec);
    if (ec) return 2;
    module->print(ir);
    graph = std::move(after);
  }
  std::error_code error;
  raw_fd_ostream out(argv[2], error);
  if (error) { errs() << error.message(); return 2; }
  out << formatv("{0:2}", json::Value(json::Object{
      {"schema", 1}, {"snapshot", argv[3]}, {"branch", argv[4]}, {"stage", argv[5]},
      {"nodes", std::move(graph.nodes)}, {"edges", std::move(graph.edges)},
      {"candidates", std::move(graph.candidates)}, {"lineage", std::move(lineage)},
      {"unmapped", "unknown; refresh values/blocks/new regions"}}));
  return 0;
}
