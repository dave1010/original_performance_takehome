"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.ops = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.ops.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized implementation using SIMD + multi-core workload partitioning.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        hash_const1 = []
        hash_const3 = []
        for hi, (_, val1, _, _, val3) in enumerate(HASH_STAGES):
            c1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            c3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c1, self.scratch_const(val1)))
            self.add("valu", ("vbroadcast", c3, self.scratch_const(val3)))
            hash_const1.append(c1)
            hash_const3.append(c3)

        core_id = self.alloc_scratch("core_id")
        core_offset = self.alloc_scratch("core_offset")
        self.add("flow", ("coreid", core_id))
        per_core = batch_size // N_CORES
        per_core_const = self.scratch_const(per_core)
        self.add("alu", ("*", core_offset, core_id, per_core_const))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        def emit(alu=None, valu=None, load=None, store=None, flow=None):
            if alu:
                for slot in alu:
                    self.ops.append(("alu", slot))
            if valu:
                for slot in valu:
                    self.ops.append(("valu", slot))
            if load:
                for slot in load:
                    self.ops.append(("load", slot))
            if store:
                for slot in store:
                    self.ops.append(("store", slot))
            if flow:
                for slot in flow:
                    self.ops.append(("flow", slot))

        def emit_valu_chunks(slots):
            for i in range(0, len(slots), SLOT_LIMITS["valu"]):
                emit(valu=slots[i : i + SLOT_LIMITS["valu"]])

        def emit_alu_chunks(slots):
            for i in range(0, len(slots), SLOT_LIMITS["alu"]):
                emit(alu=slots[i : i + SLOT_LIMITS["alu"]])

        def emit_load_chunks(slots):
            for i in range(0, len(slots), SLOT_LIMITS["load"]):
                emit(load=slots[i : i + SLOT_LIMITS["load"]])

        def emit_hash_update(block_idx):
            emit_valu_chunks(
                [("^", val_vecs[block_idx], val_vecs[block_idx], node_vecs[block_idx])]
            )
            for stage, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                # Hash stages stay fully vectorized (valu) with vectorized constants.
                emit_valu_chunks(
                    [
                        (
                            op1,
                            tmp1_vecs[block_idx],
                            val_vecs[block_idx],
                            hash_const1[stage],
                        )
                    ]
                    + [
                        (
                            op3,
                            tmp2_vecs[block_idx],
                            val_vecs[block_idx],
                            hash_const3[stage],
                        )
                    ]
                )
                emit_valu_chunks(
                    [
                        (
                            op2,
                            val_vecs[block_idx],
                            tmp1_vecs[block_idx],
                            tmp2_vecs[block_idx],
                        )
                    ]
                )
            emit_valu_chunks(
                [("&", parity_vecs[block_idx], val_vecs[block_idx], v_one)]
            )
            emit_valu_chunks(
                [("+", choice_vecs[block_idx], parity_vecs[block_idx], v_one)]
            )
            emit_valu_chunks(
                [
                    (
                        "multiply_add",
                        idx_vecs[block_idx],
                        idx_vecs[block_idx],
                        v_two,
                        choice_vecs[block_idx],
                    )
                ]
            )
            emit_valu_chunks(
                [("<", mask_vecs[block_idx], idx_vecs[block_idx], v_n_nodes)]
            )
            emit_valu_chunks(
                [("*", idx_vecs[block_idx], idx_vecs[block_idx], mask_vecs[block_idx])]
            )

        blocks = per_core // VLEN
        group_size = 6
        block_offsets = [self.alloc_scratch(f"block_offset_{i}") for i in range(group_size)]
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(group_size)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(group_size)]
        idx_vecs = [self.alloc_scratch(f"idx_vec_{i}", VLEN) for i in range(group_size)]
        val_vecs = [self.alloc_scratch(f"val_vec_{i}", VLEN) for i in range(group_size)]
        node_vecs = [self.alloc_scratch(f"node_vec_{i}", VLEN) for i in range(group_size)]
        addr_vecs = [self.alloc_scratch(f"addr_vec_{i}", VLEN) for i in range(group_size)]
        tmp1_vecs = [self.alloc_scratch(f"tmp1_vec_{i}", VLEN) for i in range(group_size)]
        tmp2_vecs = [self.alloc_scratch(f"tmp2_vec_{i}", VLEN) for i in range(group_size)]
        parity_vecs = [self.alloc_scratch(f"parity_vec_{i}", VLEN) for i in range(group_size)]
        choice_vecs = [self.alloc_scratch(f"choice_vec_{i}", VLEN) for i in range(group_size)]
        mask_vecs = [self.alloc_scratch(f"mask_vec_{i}", VLEN) for i in range(group_size)]

        block_consts = [self.scratch_const(b * VLEN) for b in range(blocks)]

        for block in range(0, blocks, group_size):
            group = min(group_size, blocks - block)
            emit_alu_chunks(
                [
                    ("+", block_offsets[i], core_offset, block_consts[block + i])
                    for i in range(group)
                ]
            )
            emit_alu_chunks(
                [
                    ("+", idx_addrs[i], self.scratch["inp_indices_p"], block_offsets[i])
                    for i in range(group)
                ]
                + [
                    ("+", val_addrs[i], self.scratch["inp_values_p"], block_offsets[i])
                    for i in range(group)
                ]
            )
            for i in range(group):
                emit(
                    load=[
                        ("vload", idx_vecs[i], idx_addrs[i]),
                        ("vload", val_vecs[i], val_addrs[i]),
                    ]
                )
            for _ in range(rounds):
                for i in range(group):
                    emit_valu_chunks(
                        [("+", addr_vecs[i], idx_vecs[i], v_forest_base)]
                    )
                    for lane in range(VLEN):
                        emit_load_chunks(
                            [("load_offset", node_vecs[i], addr_vecs[i], lane)]
                        )
                    if i > 0:
                        emit_hash_update(i - 1)
                emit_hash_update(group - 1)
            for i in range(group):
                emit(
                    store=[
                        ("vstore", idx_addrs[i], idx_vecs[i]),
                        ("vstore", val_addrs[i], val_vecs[i]),
                    ]
                )

        # Required to match with the yield in reference_kernel2
        self.ops.append(("flow", ("pause",)))
        self.instrs = self.schedule_ops(self.ops)

    def schedule_ops(self, ops):
        def slot_reads_writes(engine, slot):
            reads = set()
            writes = set()

            if engine == "alu":
                _, dest, a1, a2 = slot
                reads.update({a1, a2})
                writes.add(dest)
            elif engine == "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        reads.add(src)
                        writes.update(range(dest, dest + VLEN))
                    case ("multiply_add", dest, a, b, c):
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                        reads.update(range(c, c + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case (op, dest, a1, a2):
                        reads.update(range(a1, a1 + VLEN))
                        reads.update(range(a2, a2 + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case _:
                        pass
            elif engine == "load":
                match slot:
                    case ("load", dest, addr):
                        reads.add(addr)
                        writes.add(dest)
                    case ("load_offset", dest, addr, offset):
                        reads.add(addr + offset)
                        writes.add(dest + offset)
                    case ("vload", dest, addr):
                        reads.add(addr)
                        writes.update(range(dest, dest + VLEN))
                    case ("const", dest, _):
                        writes.add(dest)
                    case _:
                        pass
                reads.add(("mem",))
            elif engine == "store":
                match slot:
                    case ("store", addr, src):
                        reads.update({addr, src})
                    case ("vstore", addr, src):
                        reads.add(addr)
                        reads.update(range(src, src + VLEN))
                    case _:
                        pass
                reads.add(("mem",))
                writes.add(("mem",))
            elif engine == "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        reads.update({cond, a, b})
                        writes.add(dest)
                    case ("add_imm", dest, a, _):
                        reads.add(a)
                        writes.add(dest)
                    case ("vselect", dest, cond, a, b):
                        reads.update(range(cond, cond + VLEN))
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case ("coreid", dest):
                        writes.add(dest)
                    case _:
                        pass
            return reads, writes

        def schedule_segment(segment_ops, ready_stats):
            reads_list = []
            writes_list = []
            deps = [set() for _ in segment_ops]
            dependents = [set() for _ in segment_ops]
            last_write = {}
            last_read = {}

            for idx, (engine, slot) in enumerate(segment_ops):
                reads, writes = slot_reads_writes(engine, slot)
                reads_list.append(reads)
                writes_list.append(writes)
                for addr in reads:
                    if addr in last_write:
                        deps[idx].add(last_write[addr])
                for addr in writes:
                    if addr in last_write:
                        deps[idx].add(last_write[addr])
                    if addr in last_read:
                        deps[idx].add(last_read[addr])
                for addr in reads:
                    last_read[addr] = idx
                for addr in writes:
                    last_write[addr] = idx

            indegree = [len(d) for d in deps]
            for idx, parents in enumerate(deps):
                for parent in parents:
                    dependents[parent].add(idx)

            ready = {engine: [] for engine in SLOT_LIMITS}
            ready_all = []
            for idx, (engine, slot) in enumerate(segment_ops):
                if indegree[idx] == 0:
                    ready_all.append(idx)
                    if engine in ready:
                        ready[engine].append(idx)

            instrs = []
            scheduled = [False] * len(segment_ops)

            while ready_all:
                ready_stats.append(
                    {
                        engine: len(queue)
                        for engine, queue in ready.items()
                        if engine in SLOT_LIMITS
                    }
                )
                instr = {}
                taken = set()
                for engine in ["load", "valu", "alu", "store", "flow", "debug"]:
                    if engine not in SLOT_LIMITS:
                        continue
                    limit = SLOT_LIMITS[engine]
                    slots = []
                    while ready[engine] and len(slots) < limit:
                        idx = ready[engine].pop(0)
                        if scheduled[idx]:
                            continue
                        slots.append(segment_ops[idx][1])
                        taken.add(idx)
                    if slots:
                        instr[engine] = slots
                if not taken:
                    idx = ready_all.pop(0)
                    if not scheduled[idx]:
                        engine, slot = segment_ops[idx]
                        instr[engine] = [slot]
                        taken.add(idx)
                instrs.append(instr)
                for idx in taken:
                    scheduled[idx] = True
                    if idx in ready_all:
                        ready_all.remove(idx)
                    for child in dependents[idx]:
                        indegree[child] -= 1
                        if indegree[child] == 0:
                            ready_all.append(child)
                            engine = segment_ops[child][0]
                            if engine in ready:
                                ready[engine].append(child)
            return instrs

        instrs = []
        ready_stats = []
        segment = []
        for engine, slot in ops:
            if engine == "flow" and slot and slot[0] == "pause":
                if segment:
                    instrs.extend(schedule_segment(segment, ready_stats))
                    segment = []
                instrs.append({"flow": [slot]})
            else:
                segment.append((engine, slot))
        if segment:
            instrs.extend(schedule_segment(segment, ready_stats))
        self.ready_stats = ready_stats
        return instrs

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
