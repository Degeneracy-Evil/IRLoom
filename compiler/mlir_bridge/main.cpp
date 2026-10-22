#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

// All IDs are local to a single parsed snapshot. No pointer is serialized.
struct Graph {
  json::Array nodes, edges, candidates;
  DenseMap<Operation *, int64_t> ops;
  DenseMap<Block *, int64_t> blocks;
  DenseMap<Value, int64_t> values;
  std::map<std::string, int64_t> functions;
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
    if (auto f = dyn_cast<func::FuncOp>(op)) {
      kind = "function"; anchor = "function:" + f.getSymName().str();
    }
    auto id = node(kind, op->getName().getStringRef(), anchor);
    ops[op] = id;
    if (auto f = dyn_cast<func::FuncOp>(op)) functions[f.getSymName().str()] = id;
    (*nodes[id].getAsObject())["features"] = json::Array{
        int64_t(op->getNumOperands()), int64_t(op->getNumResults()),
        int64_t(op->getNumRegions()), int64_t(isMemoryEffectFree(op))};
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
  void finish(ModuleOp module) {
    module.walk([&](Operation *op) {
      auto id = ops.lookup(op);
      for (Value v : op->getOperands()) edge(value(v), id, "uses");
      for (Block *b : op->getSuccessors()) edge(id, blocks.lookup(b), "cfg");
      if (auto call = dyn_cast<func::CallOp>(op)) {
        auto found = functions.find(call.getCallee().str());
        if (found != functions.end()) edge(id, found->second, "call");
        (*nodes[id].getAsObject())["call"] = found != functions.end() ? "direct" : "external";
      } else if (isa<func::CallIndirectOp>(op)) {
        (*nodes[id].getAsObject())["call"] = "indirect-unknown";
      }
      if (!isMemoryEffectFree(op)) edge(0, id, "conservative-memory");
    });
    candidates.push_back(json::Object{{"kind", "advance"}, {"target", int64_t(0)},
                                      {"parameters", json::Object{}}, {"precondition", "verified module"}});
  }
};

int main(int argc, char **argv) {
  if (argc != 6) {
    errs() << "usage: irloom-bridge input.mlir graph.json snapshot branch stage\n";
    return 2;
  }
  DialectRegistry registry;
  registerAllDialects(registry);
  MLIRContext context(registry);
  context.disableMultithreading();
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module || failed(verify(*module))) return 3;
  Graph graph; graph.visit(module->getOperation()); graph.finish(*module);
  std::error_code error;
  raw_fd_ostream out(argv[2], error);
  if (error) { errs() << error.message(); return 2; }
  out << formatv("{0:2}", json::Value(json::Object{
      {"schema", 1}, {"snapshot", argv[3]}, {"branch", argv[4]}, {"stage", argv[5]},
      {"nodes", std::move(graph.nodes)}, {"edges", std::move(graph.edges)},
      {"candidates", std::move(graph.candidates)}}));
  return 0;
}
