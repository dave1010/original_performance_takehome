from __future__ import annotations

from dataclasses import dataclass
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from perf_takehome import KernelBuilder
from problem import (
    Machine,
    SLOT_LIMITS,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
)


@dataclass
class SlotStats:
    slots: int = 0
    bundles: int = 0


ENGINE_ORDER = ["alu", "valu", "load", "store", "flow", "debug"]


def analyze_program(instrs: list[dict]) -> dict[str, SlotStats]:
    stats = {engine: SlotStats() for engine in ENGINE_ORDER}
    for instr in instrs:
        for engine, slots in instr.items():
            stats[engine].slots += len(slots)
            stats[engine].bundles += 1
    return stats


def bundle_signature(instr: dict) -> tuple[int, int, int, int, int]:
    return (
        len(instr.get("valu", [])),
        len(instr.get("load", [])),
        len(instr.get("store", [])),
        len(instr.get("alu", [])),
        len(instr.get("flow", [])),
    )


def analyze_bundle_mix(instrs: list[dict]) -> dict[str, object]:
    histogram: dict[tuple[int, int, int, int, int], int] = {}
    valu_with_load = 0
    load_with_valu = 0
    bundles_with_load = 0
    bundles_with_valu = 0
    valu_slots_when_load = 0
    load_slots_when_valu = 0

    for instr in instrs:
        signature = bundle_signature(instr)
        histogram[signature] = histogram.get(signature, 0) + 1
        valu_slots, load_slots, _, _, _ = signature
        if load_slots > 0:
            bundles_with_load += 1
            valu_slots_when_load += valu_slots
            if valu_slots > 0:
                valu_with_load += 1
        if valu_slots > 0:
            bundles_with_valu += 1
            load_slots_when_valu += load_slots
            if load_slots > 0:
                load_with_valu += 1

    return {
        "histogram": histogram,
        "valu_with_load": valu_with_load,
        "load_with_valu": load_with_valu,
        "bundles_with_load": bundles_with_load,
        "bundles_with_valu": bundles_with_valu,
        "valu_slots_when_load": valu_slots_when_load,
        "load_slots_when_valu": load_slots_when_valu,
        "total_bundles": len(instrs),
    }


def summarize_ready_stats(ready_stats: list[dict[str, int]]) -> dict[str, float]:
    if not ready_stats:
        return {engine: 0.0 for engine in SLOT_LIMITS}
    totals = {engine: 0 for engine in SLOT_LIMITS}
    for cycle in ready_stats:
        for engine in SLOT_LIMITS:
            totals[engine] += cycle.get(engine, 0)
    return {engine: totals[engine] / len(ready_stats) for engine in SLOT_LIMITS}


def main() -> None:
    forest_height = 10
    rounds = 16
    batch_size = 256

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    stats = analyze_program(kb.instrs)
    bundle_mix = analyze_bundle_mix(kb.instrs)
    ready_stats = summarize_ready_stats(getattr(kb, "ready_stats", []))

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    for _ in reference_kernel2(mem, {}):
        machine.run()
    cycles = machine.cycle

    lines = [
        "# Kernel Instrumentation Report",
        "",
        "## Run configuration",
        f"- forest_height: {forest_height}",
        f"- rounds: {rounds}",
        f"- batch_size: {batch_size}",
        f"- n_cores: {N_CORES}",
        f"- cycles: {cycles}",
        "",
        "## Slot utilization",
        "",
        "| Engine | Slot limit | Total slots | Bundles | Avg slots/cycle | Utilization |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for engine in ENGINE_ORDER:
        if engine not in SLOT_LIMITS:
            continue
        limit = SLOT_LIMITS[engine]
        engine_stats = stats[engine]
        avg_per_cycle = engine_stats.slots / cycles if cycles else 0
        utilization = (engine_stats.slots / (cycles * limit)) if cycles else 0
        lines.append(
            f"| {engine} | {limit} | {engine_stats.slots} | {engine_stats.bundles} | {avg_per_cycle:.2f} | {utilization:.2%} |"
        )

    lines.append("")
    lines.append("## Bundle composition")
    lines.append("")
    lines.append(
        "| (valu, load, store, alu, flow) | Bundles | Share |"
    )
    lines.append("| --- | --- | --- |")
    for signature, count in sorted(
        bundle_mix["histogram"].items(), key=lambda item: item[1], reverse=True
    ):
        share = count / bundle_mix["total_bundles"] if bundle_mix["total_bundles"] else 0
        lines.append(f"| {signature} | {count} | {share:.2%} |")
    lines.append("")
    bundles_with_load = bundle_mix["bundles_with_load"]
    bundles_with_valu = bundle_mix["bundles_with_valu"]
    total_bundles = bundle_mix["total_bundles"]
    valu_with_load = bundle_mix["valu_with_load"]
    load_with_valu = bundle_mix["load_with_valu"]
    avg_valu_when_load = (
        bundle_mix["valu_slots_when_load"] / bundles_with_load if bundles_with_load else 0
    )
    avg_load_when_valu = (
        bundle_mix["load_slots_when_valu"] / bundles_with_valu if bundles_with_valu else 0
    )
    lines.append("## Bundle overlap summary")
    lines.append("")
    lines.append(f"- Total bundles: {total_bundles}")
    lines.append(
        f"- Bundles with both valu+load: {valu_with_load} ({valu_with_load / total_bundles:.2%})"
        if total_bundles
        else "- Bundles with both valu+load: 0 (0.00%)"
    )
    lines.append(
        f"- Avg valu slots when load>0: {avg_valu_when_load:.2f}"
    )
    lines.append(
        f"- Avg load slots when valu>0: {avg_load_when_valu:.2f}"
    )
    lines.append("")
    lines.append("## Ready-ops summary")
    lines.append("")
    lines.append(
        "- Avg ready ops per cycle: "
        + ", ".join(f"{engine}={ready_stats[engine]:.2f}" for engine in ENGINE_ORDER if engine in ready_stats)
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Slot utilization is derived from static instruction bundles vs. measured cycles.")
    lines.append("- Debug slots are tracked but do not contribute to cycles in the simulator.")

    with open("REPORT.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
