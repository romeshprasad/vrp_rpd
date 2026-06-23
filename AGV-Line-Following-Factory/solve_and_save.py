#!/usr/bin/env python3
"""
solve_and_save.py — Run the VRP-RPD (BRKGA) + MAPF pipeline ONCE on the
physical workstations.json layout, and save everything downstream consumers
need (grid layout, per-agent MAPF timed paths, per-agent dropoff/pickup
events) to a single JSON file.

This exists so we never have to re-solve to regenerate robot command
scripts or inspect the dropoff/pickup assignment — solve once, reuse the
saved result as many times as needed.

Usage:
    python3 solve_and_save.py --output solution.json
    python3 solve_and_save.py --num-agents 4 --resources-per-agent 2 --seed 42
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from agv_testbed.grid_env import load_physical_grid
from agv_testbed.pipeline import run_pipeline


def build_solution_dict(grid, pipeline_result) -> dict:
    """Flatten grid + pipeline result into plain JSON-serializable data."""
    vrp_result = pipeline_result.vrp_result
    mapf_result = pipeline_result.mapf_result

    ws_id_to_name = {
        grid.workstation_ids[idx]: f"WS{idx+1:02d}"
        for idx in range(len(grid.between_nodes))
    }

    agents = {}
    for aid, plan in vrp_result.agents.items():
        tp = mapf_result.paths.get(aid)
        timed_path = [[int(n), int(t)] for n, t in tp.path] if tp and tp.path else []

        events = [
            {
                "workstation_id": int(gnode),
                "workstation_name": ws_id_to_name.get(gnode),
                "operation": "DROPOFF" if op == "D" else "PICKUP",
                "solver_idx": int(solver_idx),
            }
            for gnode, op, solver_idx in plan.events
        ]

        agents[str(aid)] = {
            "completion_time": plan.completion_time,
            "events": events,
            "mapf_timed_path": timed_path,
        }

    return {
        "grid": {
            "rows": grid.topology.rows,
            "cols": grid.topology.cols,
            "between_nodes": [list(p) for p in grid.between_nodes],
            "spur_entry_ids": list(grid.spur_entry_ids),
            "workstation_ids": list(grid.workstation_ids),
            "processing_times": list(grid.processing_times),
            "depot": grid.depot,
        },
        "vrp": {
            "heuristic": vrp_result.heuristic,
            "makespan": vrp_result.makespan,
            "priority_order": vrp_result.priority_order(),
        },
        "mapf": {
            "success": mapf_result.success,
            "max_timestep": mapf_result.max_timestep(),
            "conflicts_remaining": len(mapf_result.conflicts),
        },
        "pipeline": {
            "iterations": pipeline_result.iterations,
            "converged": pipeline_result.converged,
        },
        "agents": agents,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="solution.json", help="Output JSON path")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--resources-per-agent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Workstation processing-time RNG seed")
    parser.add_argument("--max-iterations", type=int, default=5, help="MAPF conflict retry limit")
    parser.add_argument("--total-generations", type=int, default=500)
    parser.add_argument("--gens-per-cycle", type=int, default=100)
    parser.add_argument("--num-gpus", type=int, default=None,
                         help="Override GPU worker count (use 0 if CUDA driver is unavailable)")
    parser.add_argument("--num-cpu-workers", type=int, default=2)
    parser.add_argument("--use-gp", action="store_true", default=False)
    args = parser.parse_args()

    grid = load_physical_grid(rng=np.random.default_rng(args.seed))

    brkga_kwargs = dict(
        total_generations=args.total_generations,
        gens_per_cycle=args.gens_per_cycle,
        use_gp=args.use_gp,
        num_cpu_workers=args.num_cpu_workers,
    )
    if args.num_gpus is not None:
        brkga_kwargs["num_gpus"] = args.num_gpus

    pipeline_result = run_pipeline(
        grid,
        num_agents=args.num_agents,
        resources_per_agent=args.resources_per_agent,
        max_iterations=args.max_iterations,
        use_brkga=True,
        brkga_kwargs=brkga_kwargs,
    )
    pipeline_result.summary()

    if not pipeline_result.converged:
        raise RuntimeError(
            "MAPF did not converge after all retries — refusing to save a "
            "solution with unresolved collisions. Re-run, raise "
            "--max-iterations, or inspect the remaining conflict."
        )

    solution = build_solution_dict(grid, pipeline_result)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(solution, indent=2))
    print(f"\nSaved solution to {out_path} "
          f"(makespan={solution['vrp']['makespan']:.1f}, "
          f"mapf_max_t={solution['mapf']['max_timestep']})")


if __name__ == "__main__":
    main()
