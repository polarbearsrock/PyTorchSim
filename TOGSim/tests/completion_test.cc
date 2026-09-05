#include "Core.h"
#include "Scheduler.h"
#include "TileGraph.h"

#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  // Do not use assert: the production build defines NDEBUG.
  if (!condition) throw std::runtime_error(message);
}

SimulationConfig config() {
  SimulationConfig result{};
  result.num_cores = 1;
  result.num_systolic_array_per_core = 2;
  result.dram_req_size = 64;
  result.icnt_injection_ports_per_core = 2;
  return result;
}

std::shared_ptr<Instruction> instruction(Opcode op, cycle_type latency = 1,
                                        size_t elements = 256) {
  auto inst = std::make_shared<Instruction>(op, latency, 0, 0x100000,
      std::vector<size_t>{elements}, std::vector<int>{1}, 16,
      std::vector<int64_t>{0}, std::vector<int64_t>{1}, std::vector<int64_t>{});
  inst->set_overlapping_cycle(0);
  inst->set_addr_name("shared_buffer", 0);
  inst->prepare_tag_key();
  return inst;
}

std::shared_ptr<Tile> tile(std::shared_ptr<TileSubGraph> subgraph,
                          std::vector<std::shared_ptr<Instruction>> instructions = {}) {
  auto result = std::make_shared<Tile>(Tile::Status::INITIALIZED);
  result->set_owner(subgraph);
  for (auto& inst : instructions) result->append_instuction(inst);
  return result;
}

std::unique_ptr<TileGraph> graph(std::shared_ptr<TileSubGraph> subgraph, unsigned id) {
  auto result = std::make_unique<TileGraph>("regression", "test");
  result->init_cache_plan({});
  result->set_arrival_time(0);
  result->set_kernel_id(id);
  result->append_subgraph(subgraph);
  return result;
}

void last_tile_is_not_completion() {
  auto subgraph = std::make_shared<TileSubGraph>();
  auto work = tile(subgraph);
  subgraph->add_tile(work);
  require(!subgraph->is_finished(), "undispatched tile marked complete");
  require(subgraph->get_tile() == work, "wrong tile dispatched");
  require(!subgraph->is_finished(), "last dispatch incorrectly completed subgraph");
  subgraph->finish_tile(work);
  require(subgraph->is_finished(), "retired subgraph did not complete");
}

void blocked_children_and_multiple_inflight_tiles() {
  auto subgraph = std::make_shared<TileSubGraph>();
  auto parent = tile(subgraph), child = tile(subgraph), independent = tile(subgraph);
  parent->append_child(child);
  subgraph->add_tile(parent);
  subgraph->add_tile(child);
  subgraph->add_tile(independent);
  auto first = subgraph->get_tile(), second = subgraph->get_tile();
  require(first != child && second != child, "blocked child dispatched");
  subgraph->finish_tile(parent);
  require(subgraph->get_tile() == child, "child not released on parent retirement");
  subgraph->finish_tile(child);
  require(!subgraph->is_finished(), "independent in-flight tile forgotten");
  subgraph->finish_tile(independent);
  require(subgraph->is_finished(), "all retired tiles did not complete");
}

void subgraph_slot_and_scheduler_retirement() {
  cycle_type cycle = 1;
  uint64_t time = 0;
  Scheduler scheduler(config(), &cycle, &time, 0);
  auto first = std::make_shared<TileSubGraph>(), second = std::make_shared<TileSubGraph>();
  auto a = tile(first), b = tile(second);
  first->add_tile(a);
  second->add_tile(b);
  scheduler.enqueue_graph(graph(first, 0));
  scheduler.enqueue_graph(graph(second, 1));
  scheduler.peek_tile(0, 0, CoreType::WS_MESH); // Allocate the first subgraph.
  require(scheduler.get_tile(0, 0) == a, "first kernel missing");
  for (int i = 0; i < 4; ++i)
    require(scheduler.peek_tile(0, 0, CoreType::WS_MESH)->get_status() == Tile::Status::EMPTY,
            "next kernel exposed before producer retirement");
  scheduler.finish_tile(a);
  scheduler.peek_tile(0, 0, CoreType::WS_MESH);
  require(scheduler.get_tile(0, 0) == b, "retirement failed to advance scheduler");
  require(!scheduler.empty(), "consumer removed at dispatch");
  scheduler.finish_tile(b);
  require(scheduler.empty(), "final retirement failed to drain scheduler");
}

void all_assigned_subgraphs_must_retire() {
  auto a = std::make_shared<TileSubGraph>(), b = std::make_shared<TileSubGraph>();
  auto ta = tile(a), tb = tile(b);
  a->add_tile(ta);
  b->add_tile(tb);
  auto kernel = graph(a, 0);
  kernel->append_subgraph(b);
  kernel->peek_tile(0, 0);
  kernel->peek_tile(0, 1);
  kernel->get_tile(0, 0);
  kernel->get_tile(0, 1);
  a->finish_tile(ta);
  require(!kernel->is_finished(), "other slot's in-flight subgraph forgotten");
  b->finish_tile(tb);
  require(kernel->is_finished(), "multi-subgraph kernel failed to complete");
}

// Exercise the real Core + DMA + Scheduler, supplying deterministic delayed
// memory replies. The consumer must not even be admitted before producer DMA
// responses, while async DMA is still allowed to release same-kernel children.
void memory_barrier(Opcode producer_opcode, bool async, size_t elements,
                    cycle_type memory_latency) {
  cycle_type cycle = 1;
  uint64_t time = 0;
  auto cfg = config();
  Core core(0, cfg);
  Scheduler scheduler(cfg, &cycle, &time, 0);
  auto producer = std::make_shared<TileSubGraph>(), consumer = std::make_shared<TileSubGraph>();
  auto compute = instruction(Opcode::COMP, 40);
  auto transfer = instruction(producer_opcode, 1, elements);
  transfer->set_is_async(async);
  compute->add_child(transfer);
  auto producer_tile = tile(producer, {compute, transfer});
  auto child_inst = instruction(Opcode::COMP, 2);
  auto child_tile = tile(producer, {child_inst});
  producer_tile->append_child(child_tile);
  producer->add_tile(producer_tile);
  producer->add_tile(child_tile);
  auto read = instruction(Opcode::MOVIN);
  auto consumer_tile = tile(consumer, {read});
  consumer->add_tile(consumer_tile);
  scheduler.enqueue_graph(graph(producer, 0));
  scheduler.enqueue_graph(graph(consumer, 1));
  std::vector<std::pair<cycle_type, mem_fetch*>> replies;
  size_t requests = 0, responses = 0;
  bool admitted_consumer = false, saw_async_overlap = false;
  for (; cycle < 2000; ++cycle) {
    for (int slot = 0; slot < 2; ++slot) {
      auto candidate = scheduler.peek_tile(0, slot, CoreType::WS_MESH);
      if (candidate->get_status() == Tile::Status::EMPTY || !core.can_issue(candidate)) continue;
      auto dispatched = scheduler.get_tile(0, slot);
      if (dispatched == consumer_tile) {
        require(compute->finished && transfer->finished, "consumer admitted before producer instructions finished");
        require(requests == responses, "consumer admitted before producer memory responses");
        admitted_consumer = true;
      }
      if (dispatched == child_tile && requests != responses) saw_async_overlap = true;
      core.issue(dispatched);
    }
    auto finished = core.pop_finished_tile();
    if (finished->get_status() == Tile::Status::FINISH) scheduler.finish_tile(finished);
    core.cycle();
    while (core.has_memory_request()) {
      auto request = core.top_memory_request();
      core.pop_memory_request();
      if (request->get_custom_data() == transfer.get()) ++requests;
      replies.emplace_back(cycle + memory_latency, request);
    }
    for (auto it = replies.begin(); it != replies.end();) {
      if (it->first > cycle) { ++it; continue; }
      if (it->second->get_custom_data() == transfer.get()) ++responses;
      core.push_memory_response(it->second);
      it = replies.erase(it);
    }
    if (scheduler.empty() && !core.running() && replies.empty()) break;
  }
  require(cycle < 2000 && admitted_consumer && read->finished, "simulation failed to drain");
  if (elements && memory_latency > 20 && (async || producer_opcode == Opcode::MOVOUT))
    require(saw_async_overlap, "fix serialized same-kernel asynchronous DMA");
}
} // namespace

int main() {
  spdlog::set_level(spdlog::level::off);
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"last tile remains in flight", last_tile_is_not_completion},
      {"blocked children and multiple tiles", blocked_children_and_multiple_inflight_tiles},
      {"scheduler advances on retirement", subgraph_slot_and_scheduler_retirement},
      {"all assigned subgraphs retire", all_assigned_subgraphs_must_retire},
      {"store responses gate next kernel", [] { memory_barrier(Opcode::MOVOUT, false, 256, 80); }},
      {"async load responses gate next kernel", [] { memory_barrier(Opcode::MOVIN, true, 256, 80); }},
      {"synchronous load completion", [] { memory_barrier(Opcode::MOVIN, false, 256, 80); }},
      {"immediate responses before DMA retirement", [] { memory_barrier(Opcode::MOVOUT, false, 256, 0); }},
      {"zero-length store", [] { memory_barrier(Opcode::MOVOUT, false, 0, 0); }},
      {"zero-length async load", [] { memory_barrier(Opcode::MOVIN, true, 0, 0); }},
  };
  int failures = 0;
  for (auto& [name, test] : tests) {
    try { test(); std::cout << "PASS: " << name << '\n'; }
    catch (const std::exception& error) {
      ++failures;
      std::cout << "FAIL: " << name << ": " << error.what() << '\n';
    }
  }
  std::cout << tests.size() - failures << "/" << tests.size() << " passed\n";
  return failures ? 1 : 0;
}
