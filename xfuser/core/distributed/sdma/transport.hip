// ROCr SDMA transport for xDiT's Ulysses all-to-all (see transport.py), loaded with ctypes.
//
// Buffers that peers write into are allocated here with hipMalloc (or as fine-grained memory)
// and shared with HIP IPC handles. A peer's buffer opened from its handle is mapped for the
// local GPU only, so every copy names the local GPU on both sides.
//
// Copies are pushes: this rank's SDMA engines write its data into peers' buffers, one engine
// per peer, all peers at once. Behind a peer's copies, a 4-byte copy on the same engine sets
// this rank's flag word at that peer (one word per sender, so no two engines write the same
// word): an engine runs its copies in order, and the flag lands after the data. Every copy
// first waits for a trigger signal that a one-thread kernel releases on the compute stream,
// so no copy reads data the stream has not written.
//
// A call's copies are described once, as a plan: for each peer, a list of (source offset,
// destination address, bytes) and the flag word to set after them. A plan is issued with the
// source base of that call, so the per-call host work is one loop in C.
//
// The compute stream waits for flags with sx_wait_flags, a one-thread kernel whose wait is
// bounded: a flag that never lands is recorded (sx_peek_error / sx_take_error), and the stream
// either moves on or stays held for the host to abort the process. The kernel resets the
// flags it waited for, so the next use of the same words needs no separate clear.
//
// Functions return 0 on success and a negative code otherwise.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

// A bounded flag wait that ran out of time, written by sx_wait_flags into host-visible memory.
struct SxError {
  unsigned long long count;    // waits that timed out since the last sx_take_error
  unsigned long long missing;  // first one: bit i set if flag word i never read 1
  int tag;                     // first one: caller's tag (which wait) and call id
  int call;
};

namespace {

struct Gpu {
  hsa_agent_t agent;
  uint32_t bus;
};

constexpr int kTriggers = 32;  // per slot

struct Slot {
  hsa_signal_t trigger[kTriggers];
  volatile hsa_signal_value_t* trigger_value[kTriggers];
  std::vector<hsa_signal_t> data;  // one per data copy of a call
  std::vector<hsa_signal_t> flag;  // one per flag copy of a call
  int armed_data = 0;              // what the last call on this slot used
  int armed_flags = 0;
};

std::vector<Gpu> g_gpus;
std::vector<Slot> g_slots;
hsa_agent_t g_local;
SxError* g_error = nullptr;  // host-mapped
int* g_scratch = nullptr;    // device word that a dropped flag write goes to
int g_drop_call = -1, g_drop_issue = -1, g_drop_index = -1;
int g_calls = 0, g_issues = 0;  // calls armed so far; pushes and plans issued in the current call
unsigned long long g_ticks_per_s = 0;

hsa_status_t collect(hsa_agent_t agent, void*) {
  hsa_device_type_t type;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_GPU) {
    uint32_t bdf = 0;
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf);
    g_gpus.push_back({agent, (bdf >> 8) & 0xff});
  }
  return HSA_STATUS_SUCCESS;
}

// Agents are found by PCI bus: HIP's device numbering depends on the visible-device variables,
// ROCr's agent list does not.
int agent_of_bus(int bus, hsa_agent_t* out) {
  for (const Gpu& g : g_gpus)
    if (g.bus == (uint32_t)bus) {
      *out = g.agent;
      return 0;
    }
  return -2;
}

int agent_of(int hip_dev, hsa_agent_t* out) {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, hip_dev) != hipSuccess) return -1;
  return agent_of_bus(prop.pciBusID, out);
}

// Waits until `s` drops below 1 (its copy finished); false after kWaitSeconds, so a copy
// that never runs fails the call instead of hanging it.
constexpr double kWaitSeconds = 60.0;

bool wait_done(hsa_signal_t s) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(kWaitSeconds);
  while (hsa_signal_load_scacquire(s) >= 1) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, 1000000, HSA_WAIT_STATE_ACTIVE);
  }
  return true;
}

int lowest(uint32_t mask) {
  for (int e = 0; e < 32; ++e)
    if (mask >> e & 1) return e;
  return -1;
}

__global__ void release_trigger(hsa_signal_value_t* value) {
  __hip_atomic_store(value, (hsa_signal_value_t)0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Waits until all `count` words at `words` read 1 (acquire, system scope), then sets them back
// to 0. Once `budget` wall-clock ticks have passed it records which words were missing; then
// it returns, or with `block` keeps waiting, holding the stream so no work after it runs on
// data that never came (the host's watchdog aborts the process).
__global__ void wait_flags(int* words, int count, unsigned long long budget, SxError* err, int tag, int call,
                           int block) {
  unsigned long long start = wall_clock64();
  bool recorded = false;
  while (true) {
    unsigned long long missing = 0;
    for (int i = 0; i < count; ++i)
      if (__hip_atomic_load(words + i, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 1) missing |= 1ull << i;
    if (!missing) break;
    if (!recorded && wall_clock64() - start > budget) {
      if (__hip_atomic_fetch_add(&err->count, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
        err->missing = missing;
        err->tag = tag;
        err->call = call;
      }
      __threadfence_system();
      if (!block) break;
      recorded = true;
    }
    __builtin_amdgcn_s_sleep(2);
  }
  for (int i = 0; i < count; ++i) __hip_atomic_store(words + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __threadfence_system();
}

// Launched once at init: fails to launch if the library has no code object for this GPU.
__global__ void probe(int* word) { *word = 0; }

// Grows `v` to `n` signals, all at value 0; false if ROCr cannot create one.
bool grow(std::vector<hsa_signal_t>& v, int n) {
  while ((int)v.size() < n) {
    hsa_signal_t s;
    if (hsa_signal_create(0, 0, nullptr, &s) != HSA_STATUS_SUCCESS) return false;
    v.push_back(s);
  }
  return true;
}

}  // namespace

extern "C" {

// `slots` calls may be in flight at once; a slot is reused only after its copies finished.
// Each slot starts with `max_copies` signals of each kind and grows when a call needs more.
int sx_init(int local_dev, int max_copies, int slots) {
  if (!g_slots.empty()) return 0;
  if (hsa_init() != HSA_STATUS_SUCCESS) return -1;
  if (hsa_iterate_agents(collect, nullptr) != HSA_STATUS_SUCCESS) return -2;
  if (agent_of(local_dev, &g_local)) return -3;
  if (hipHostMalloc((void**)&g_error, sizeof(SxError), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess)
    return -8;
  *g_error = SxError{};
  if (hipMalloc((void**)&g_scratch, 64) != hipSuccess) return -9;
  hipLaunchKernelGGL(probe, dim3(1), dim3(1), 0, 0, g_scratch);
  if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) return -11;  // no code for this GPU
  int khz = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, local_dev) != hipSuccess || khz <= 0) return -10;
  g_ticks_per_s = (unsigned long long)khz * 1000ull;
  g_slots.resize(slots);
  for (Slot& s : g_slots) {
    // Triggers are written by a kernel, so their values must be addressable: GPU-only signals.
    for (int k = 0; k < kTriggers; ++k) {
      if (hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &s.trigger[k]) != HSA_STATUS_SUCCESS)
        return -4;
      if (hsa_amd_signal_value_pointer(s.trigger[k], &s.trigger_value[k]) != HSA_STATUS_SUCCESS) return -5;
    }
    if (!grow(s.data, max_copies) || !grow(s.flag, max_copies)) return -6;
  }
  return 0;
}

// One SDMA engine (bit index) per peer, in engines[0..n-1]: ROCr's preferred engine for the
// link the data crosses (from `local_dev` to the peer when pushing, into `local_dev` from the
// peer when pulling); a peer whose preferred engine is taken gets the lowest available engine
// no peer prefers (then any free one). Peers are named by PCI bus (sx_bus_of).
int sx_engine_plan(int local_dev, const int* peer_bus, int n, int push, int* engines) {
  hsa_agent_t local, peer;
  if (agent_of(local_dev, &local)) return -1;
  std::vector<uint32_t> avail(n), preferred(n, 0);
  for (int i = 0; i < n; ++i) {
    if (agent_of_bus(peer_bus[i], &peer)) return -2;
    hsa_agent_t dst = push ? peer : local, src = push ? local : peer;
    if (hsa_amd_memory_copy_engine_status(dst, src, &avail[i]) != HSA_STATUS_SUCCESS) return -3;
    uint32_t mask = 0;
    if (hsa_amd_memory_get_preferred_copy_engine(dst, src, &mask) == HSA_STATUS_SUCCESS) preferred[i] = mask;
  }
  uint32_t used = 0, wanted = 0;
  for (int i = 0; i < n; ++i) {
    engines[i] = lowest(preferred[i] & avail[i]);
    if (engines[i] >= 0) wanted |= 1u << engines[i];
  }
  for (int i = 0; i < n; ++i) {
    uint32_t bit = engines[i] >= 0 ? 1u << engines[i] : 0;
    if (bit && !(used & bit)) {
      used |= bit;
      continue;
    }
    engines[i] = -1;
  }
  for (int i = 0; i < n; ++i) {
    if (engines[i] >= 0) continue;
    int e = lowest(avail[i] & ~used & ~wanted);
    if (e < 0) e = lowest(avail[i] & ~used);
    if (e < 0) return -4;
    engines[i] = e;
    used |= 1u << e;
  }
  return 0;
}

int sx_bus_of(int hip_dev) {
  hipDeviceProp_t prop;
  return hipGetDeviceProperties(&prop, hip_dev) == hipSuccess ? prop.pciBusID : -1;
}

// Device memory for peers to write into, by `kind`: 0 hipMalloc, 1 fine-grained, 2 uncached.
// Sizes are rounded up to 2 MiB: a smaller fine-grained or uncached allocation that is not a
// whole number of pages cannot be exported with hipIpcGetMemHandle.
int sx_malloc(uint64_t bytes, int kind, uint64_t* out) {
  constexpr uint64_t kGranule = 2ull << 20;
  bytes = (bytes + kGranule - 1) / kGranule * kGranule;
  void* p = nullptr;
  hipError_t e = kind == 0   ? hipMalloc(&p, bytes)
                 : kind == 1 ? hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained)
                 : kind == 2 ? hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached)
                             : hipErrorInvalidValue;
  if (e != hipSuccess) return -1;
  *out = (uint64_t)p;
  return 0;
}

int sx_free(uint64_t ptr) { return hipFree((void*)ptr) == hipSuccess ? 0 : -1; }

int sx_memset(uint64_t ptr, int value, uint64_t bytes) {
  if (hipMemset((void*)ptr, value, bytes) != hipSuccess) return -1;
  return hipDeviceSynchronize() == hipSuccess ? 0 : -2;
}

int sx_ipc_handle_size() { return (int)sizeof(hipIpcMemHandle_t); }

// The IPC handle of an sx_malloc buffer, written to `out` (sx_ipc_handle_size bytes).
int sx_ipc_get(uint64_t ptr, void* out) {
  hipIpcMemHandle_t h;
  if (hipIpcGetMemHandle(&h, (void*)ptr) != hipSuccess) return -1;
  memcpy(out, &h, sizeof(h));
  return 0;
}

// Maps a peer's buffer from its IPC handle for this GPU.
int sx_ipc_open(const void* handle, uint64_t* out) {
  hipIpcMemHandle_t h;
  memcpy(&h, handle, sizeof(h));
  void* p = nullptr;
  if (hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess) != hipSuccess) return -1;
  *out = (uint64_t)p;
  return 0;
}

int sx_ipc_close(uint64_t ptr) { return hipIpcCloseMemHandle((void*)ptr) == hipSuccess ? 0 : -1; }

// ROCr's view of a pointer: hsa_amd_pointer_type_t (0 unknown, 1 HSA, 2 locked, 3 graphics,
// 4 IPC, ...).
int sx_pointer_type(uint64_t ptr, int* type) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info((void*)ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS) return -1;
  *type = (int)info.type;
  return 0;
}

// Arms `slot` for a call of `n_data` data copies and `n_flags` flag copies: waits for the
// copies of its previous use to finish and sets its triggers and the signals the call will use
// to 1. Call before sx_release and sx_push.
int sx_arm(int slot, int n_data, int n_flags) {
  Slot& s = g_slots[slot];
  for (int i = 0; i < s.armed_data; ++i)
    if (!wait_done(s.data[i])) return -1000 - i;  // data copy i never landed
  for (int i = 0; i < s.armed_flags; ++i)
    if (!wait_done(s.flag[i])) return -2000 - i;  // flag copy i never landed
  // The slot is idle now, so it can grow for a call larger than any before.
  if (!grow(s.data, n_data) || !grow(s.flag, n_flags)) return -1;
  for (int k = 0; k < kTriggers; ++k) hsa_signal_store_relaxed(s.trigger[k], 1);
  for (int i = 0; i < n_data; ++i) hsa_signal_store_relaxed(s.data[i], 1);
  for (int i = 0; i < n_flags; ++i) hsa_signal_store_relaxed(s.flag[i], 1);
  s.armed_data = n_data;
  s.armed_flags = n_flags;
  ++g_calls;
  g_issues = 0;
  return 0;
}

// Enqueues the release of `slot`'s trigger `k` on `stream`: copies gated on it start once the
// stream reaches this point. Without it (host release), sx_open must be called instead.
int sx_release(int slot, int k, hipStream_t stream) {
  if (k < 0 || k >= kTriggers) return -2;
  hipLaunchKernelGGL(release_trigger, dim3(1), dim3(1), 0, stream,
                     const_cast<hsa_signal_value_t*>(g_slots[slot].trigger_value[k]));
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

// Releases `slot`'s trigger `k` from the host, for copies issued after the host waited for the
// work they depend on.
int sx_open(int slot, int k) {
  if (k < 0 || k >= kTriggers) return -2;
  hsa_signal_store_screlease(g_slots[slot].trigger[k], 0);
  return 0;
}

}  // extern "C"

namespace {

// The flag write of issue `issue` of the current call to peer index `i`, unless the test hook
// diverts it to a scratch word, as if it were lost.
void* flag_target(uint64_t flag, int issue, int i) {
  if (g_calls - 1 == g_drop_call && issue == g_drop_issue && i == g_drop_index) {
    g_drop_call = g_drop_issue = g_drop_index = -1;
    return g_scratch;
  }
  return (void*)flag;
}

// Queues `bytes` from `src` to `dst` on `engine` once trigger k of `s` is released.
hsa_status_t copy_after_trigger(Slot& s, int k, void* dst, const void* src, uint64_t bytes, hsa_signal_t done,
                                int engine) {
  return hsa_amd_memory_async_copy_on_engine(dst, g_local, src, g_local, bytes, 1, &s.trigger[k], done,
                                             (hsa_amd_sdma_engine_id_t)(1u << engine), true);
}

struct PlanCopy {
  uint64_t src_offset, dst, bytes;
};

struct Plan {
  bool live = false;
  std::vector<int> engines;                  // per peer
  std::vector<std::vector<PlanCopy>> peers;  // per peer, in issue order
  std::vector<uint64_t> flags;               // per peer: the flag word set behind its copies
  int copies = 0;
};

std::vector<Plan> g_plans;

}  // namespace

extern "C" {

// Push `items` copies of `bytes` to each of `n` peers on `slot`, once trigger `k` is released.
// Copy j to peer i reads
//     src_base + peer_ranks[i] * src_peer_stride + j * src_item_stride
// and writes
//     dst_bases[i] + dst_offset + j * dst_item_stride
// on SDMA engine engines[i], using the slot's data signals first_data + i * items + j. Behind
// them, on the same engine, the 4 bytes at `flag_src` go to flag_bases[i] + flag_offset, using
// flag signal first_flag + i.
int sx_push(int slot, int k, int first_data, int first_flag, int n, const int* peer_ranks, uint64_t src_base,
            uint64_t src_peer_stride, uint64_t src_item_stride, const uint64_t* dst_bases, uint64_t dst_offset,
            uint64_t dst_item_stride, uint64_t bytes, int items, const int* engines, const uint64_t* flag_bases,
            uint64_t flag_offset, uint64_t flag_src) {
  Slot& s = g_slots[slot];
  if (first_data + n * items > s.armed_data || first_flag + n > s.armed_flags || k < 0 || k >= kTriggers) return -1;
  int issue = g_issues++;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < items; ++j) {
      void* dst = (void*)(dst_bases[i] + dst_offset + (uint64_t)j * dst_item_stride);
      const void* src = (const void*)(src_base + (uint64_t)peer_ranks[i] * src_peer_stride + (uint64_t)j * src_item_stride);
      hsa_status_t st = copy_after_trigger(s, k, dst, src, bytes, s.data[first_data + i * items + j], engines[i]);
      if (st != HSA_STATUS_SUCCESS) return -100 - (int)st;
    }
    hsa_status_t st = copy_after_trigger(s, k, flag_target(flag_bases[i] + flag_offset, issue, i), (const void*)flag_src,
                                         4, s.flag[first_flag + i], engines[i]);
    if (st != HSA_STATUS_SUCCESS) return -200 - (int)st;
  }
  return 0;
}

// Registers a plan for `n` peers: peer i's copies are the next counts[i] entries of
// src_offsets / dsts / bytes, on SDMA engine engines[i], followed on that engine by a 4-byte
// flag write to flag_dsts[i]. Returns the plan id (>= 0), or a negative code.
int sx_plan_create(int n, const int* engines, const int* counts, const uint64_t* src_offsets, const uint64_t* dsts,
                   const uint64_t* bytes, const uint64_t* flag_dsts) {
  Plan plan;
  plan.live = true;
  int c = 0;
  for (int i = 0; i < n; ++i) {
    if (counts[i] < 0) return -1;
    plan.engines.push_back(engines[i]);
    plan.flags.push_back(flag_dsts[i]);
    std::vector<PlanCopy> copies;
    for (int j = 0; j < counts[i]; ++j, ++c) copies.push_back({src_offsets[c], dsts[c], bytes[c]});
    plan.peers.push_back(std::move(copies));
  }
  plan.copies = c;
  for (size_t id = 0; id < g_plans.size(); ++id)
    if (!g_plans[id].live) {
      g_plans[id] = std::move(plan);
      return (int)id;
    }
  g_plans.push_back(std::move(plan));
  return (int)g_plans.size() - 1;
}

// How many data copies (and so data signals) plan `id` issues; negative if there is no such plan.
int sx_plan_copies(int id) {
  if (id < 0 || id >= (int)g_plans.size() || !g_plans[id].live) return -1;
  return g_plans[id].copies;
}

int sx_plan_free(int id) {
  if (id < 0 || id >= (int)g_plans.size()) return -1;
  g_plans[id] = Plan{};
  return 0;
}

// Issues plan `id` on `slot`, its sources at src_base + their offsets, once trigger `k` is
// released: data copies use the slot's data signals from first_data on, in plan order, and
// peer i's flag write (the 4 bytes at `flag_src`) flag signal first_flag + i.
int sx_plan_issue(int slot, int k, int first_data, int first_flag, int id, uint64_t src_base, uint64_t flag_src) {
  if (id < 0 || id >= (int)g_plans.size() || !g_plans[id].live) return -2;
  const Plan& plan = g_plans[id];
  Slot& s = g_slots[slot];
  int n = (int)plan.peers.size();
  if (first_data + plan.copies > s.armed_data || first_flag + n > s.armed_flags || k < 0 || k >= kTriggers) return -1;
  int issue = g_issues++, c = first_data;
  for (int i = 0; i < n; ++i) {
    for (const PlanCopy& copy : plan.peers[i]) {
      hsa_status_t st = copy_after_trigger(s, k, (void*)copy.dst, (const void*)(src_base + copy.src_offset), copy.bytes,
                                           s.data[c++], plan.engines[i]);
      if (st != HSA_STATUS_SUCCESS) return -100 - (int)st;
    }
    hsa_status_t st = copy_after_trigger(s, k, flag_target(plan.flags[i], issue, i), (const void*)flag_src, 4,
                                         s.flag[first_flag + i], plan.engines[i]);
    if (st != HSA_STATUS_SUCCESS) return -200 - (int)st;
  }
  return 0;
}

// Enqueues on `stream` a wait until the `count` 32-bit flags at `words` all read 1, then resets
// them to 0. After `seconds` a wait that is still missing flags is recorded (sx_peek_error,
// sx_take_error); then the stream moves on, or with `block` stays held (for the host to abort
// the process).
int sx_wait_flags(hipStream_t stream, uint64_t words, int count, double seconds, int tag, int call, int block) {
  if (count > 64) return -1;
  unsigned long long budget = (unsigned long long)(seconds * (double)g_ticks_per_s);
  hipLaunchKernelGGL(wait_flags, dim3(1), dim3(1), 0, stream, (int*)words, count, budget, g_error, tag, call, block);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

// As sx_take_error, without clearing the record: for a watchdog that polls it.
int sx_peek_error(int* tag, int* call, uint64_t* missing) {
  volatile SxError* e = g_error;
  if (e == nullptr) return 0;
  unsigned long long count = e->count;
  if (count == 0) return 0;
  *tag = e->tag;
  *call = e->call;
  *missing = e->missing;
  return (int)(count > 0x7fffffff ? 0x7fffffff : count);
}

// Returns how many bounded waits ran out since the last call (0 if none) and the first one's
// tag, call id and missing-flag mask; clears the record.
int sx_take_error(int* tag, int* call, uint64_t* missing) {
  volatile SxError* e = g_error;
  if (e == nullptr) return 0;
  unsigned long long count = e->count;
  if (count == 0) return 0;
  *tag = e->tag;
  *call = e->call;
  *missing = e->missing;
  e->count = 0;
  return (int)(count > 0x7fffffff ? 0x7fffffff : count);
}

// Test hook: in the `ahead`-th call armed from now (1 = the next), issue `issue` of that call
// (pushes and plans count in the order they are issued) loses its flag write to peer index
// `index`: it goes to a scratch word instead.
int sx_debug_drop_flag(int ahead, int issue, int index) {
  g_drop_call = g_calls + ahead - 1;
  g_drop_issue = issue;
  g_drop_index = index;
  return 0;
}

}
