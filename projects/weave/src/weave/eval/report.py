"""Render the Phase 0 measurements from stamped audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def phase0_report(results: Path) -> str:
    records = [json.loads(path.read_text()) for path in sorted(results.glob("*.json"))]
    identities = {
        (rec["git_sha"], rec["config_sha256"], rec["implementation_sha256"])
        for rec in records
    }
    if len(identities) > 1:
        raise ValueError(
            "Audit files have different Git, config, or implementation hashes."
        )

    lines = ["# WEAVE Phase 0 measurements", ""]
    if not identities:
        return "\n".join(lines + ["No audit artifacts found.", ""])
    git_sha, config_sha, implementation_sha = next(iter(identities))
    lines += [
        f"Git `{git_sha}` · config `{config_sha}` · implementation `{implementation_sha}`",
        "",
    ]
    for record in records:
        result = record["result"]
        label = record["mode"]
        if "error" in result:
            lines += [f"**{label} failed.** `{result['error']}`", ""]
        elif label == "inventory":
            lines += [
                "## Recording inventory",
                "",
                "| Task | Recorded ID | Arms | Train / validation / test | RGB streams | State snapshot |",
                "| --- | --- | ---: | --- | ---: | --- |",
            ]
            for task, info in result["tasks"].items():
                splits = info["split_sizes"]
                paths = info["hdf5_datasets"]
                arms = sum(
                    name == "actions" or name.startswith("actions/") for name in paths
                )
                rgb = sum(name.endswith("/rgb") for name in paths)
                states = any(
                    name == "env_states" or name.startswith("env_states/")
                    for name in paths
                )
                sizes = " / ".join(
                    str(splits[s]) for s in ("train", "validation", "test")
                )
                lines.append(
                    f"| {task} | {info['recording']['env_id']} | {arms} | {sizes} | {rgb} | "
                    f"{'present' if states else 'absent'} |"
                )
            lines.append("")
        elif label == "replay":
            lines += [
                "## Native open-loop replay",
                "",
                "| Task | Raw success | Delta success | Repeat agreement | BC precheck | Control Hz |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
            for task, info in result["tasks"].items():
                if "error" in info:
                    lines.append(f"| {task} | Error | — | — | — | — |")
                    continue
                lines.append(
                    f"| {task} | {info['raw_success_rate']:.1%} | "
                    f"{info['delta_success_rate']:.1%} | {info['repeat_agreement_rate']:.1%} | "
                    f"{info['bc_precheck']} | {info['control_hz']} |"
                )
            lines.append("")
            for task, info in result["tasks"].items():
                if "error" in info:
                    lines += [f"- {task} replay error. `{info['error']}`"]
            lines.append("")
    bench = [r["result"] for r in records if r["mode"] == "benchmark"]
    if bench:
        lines += [
            "## Native two-Panda throughput",
            "",
            "| Task | Backend | Envs | Env steps/s | Process MiB | Incremental MiB/env | Contact probe |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for result in sorted(
            bench, key=lambda r: (r["task"], r["backend"], r["num_envs"])
        ):
            prefix = (
                f"| {result['task']} | {result['backend']} | {result['num_envs']} |"
            )
            if "error" in result:
                lines.append(prefix + " Error | — | — | — |")
                continue
            memory = result["process_gpu_mib"]
            per_env = result["incremental_gpu_mib_per_env"]
            contact = result.get("contact_probe")
            probe = (
                "—" if contact is None else ("pass" if contact["passed"] else "fail")
            )
            lines.append(
                f"{prefix} {result['median_env_steps_per_second']:.0f} | "
                f"{memory if memory is not None else '—'} | "
                f"{f'{per_env:.2f}' if per_env is not None else '—'} | {probe} |"
            )
        lines.append("")
    for mode in ("inventory", "replay", "benchmark"):
        if not any(record["mode"] == mode for record in records):
            lines += [f"{mode.capitalize()} has no measurements.", ""]
    plan = records[0].get("benchmark_plan")
    if plan and bench:
        completed = {(r["task"], r["num_envs"]) for r in bench if r["backend"] == "gpu"}
        missing = [
            f"{task} × {count}"
            for task in plan["tasks"]
            for count in plan["env_counts"]
            if (task, count) not in completed
        ]
        if missing:
            lines += [
                "Missing GPU benchmark artifacts",
                "",
                *(f"- {cell}" for cell in missing),
                "",
            ]
    lines += ["G0 requires a reviewed environment branch decision in the audit.", ""]
    return "\n".join(lines)
