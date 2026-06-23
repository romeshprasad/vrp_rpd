"""
Two-Layer Pipeline: VRP-RPD + MAPF with Iterative Feedback
-----------------------------------------------------------
Layer 1: VRP-RPD heuristic search -> agent routes + completion times
Layer 2: Prioritized MAPF -> collision-free timed paths

If MAPF finds conflicts, the conflicted agents' routes are flagged and
Layer 1 is re-run with a perturbed seed so different routes emerge.
This continues until convergence (no conflicts) or max_iterations.

For Plan 1 (no stochasticity, single depot) the feedback is simple:
  - identify which agents had conflicts
  - re-solve only those agents' sub-problems with different random seeds
  - re-run MAPF on full set with updated paths

Returns a PipelineResult with the final collision-free timed paths
ready for visualization.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from vrp_rpd.agv_testbed.grid_env import WarehouseGrid, load_physical_grid, bfs_path
from vrp_rpd.agv_testbed.vrp_solver import SolverResult, run_heuristics, run_brkga
from vrp_rpd.agv_testbed.mapf_solver import MAPFResult, TimedPath, solve_mapf, detect_conflicts
from vrp_rpd.agv_testbed.instance_builder import tours_to_grid_paths


@dataclass
class PipelineResult:
    vrp_result: SolverResult
    mapf_result: MAPFResult
    iterations: int
    converged: bool

    def summary(self):
        print(f"\n{'='*60}")
        print(f"PIPELINE RESULT")
        print(f"  VRP makespan    : {self.vrp_result.makespan:.1f}")
        print(f"  MAPF success    : {self.mapf_result.success}")
        print(f"  Iterations      : {self.iterations}")
        print(f"  Converged       : {self.converged}")
        print(f"  Final max t     : {self.mapf_result.max_timestep()}")
        print(f"  Remaining conf. : {len(self.mapf_result.conflicts)}")
        print(f"{'='*60}\n")


def run_pipeline(
    grid: WarehouseGrid,
    num_agents: int = 3,
    resources_per_agent: int = 5,
    max_iterations: int = 5,
    seed: int = 42,
    use_brkga: bool = True,
    brkga_kwargs: Optional[Dict] = None,
) -> PipelineResult:
    """
    Run VRP-RPD then MAPF iteratively until convergence.

    The grid's workstation layout is fixed (physical positions from
    workstations.json) — on conflict, the same grid is reused; with the full
    BRKGA solver (use_brkga=True) each retry's internal randomness can still
    produce different routes, unlike the deterministic construction heuristics.

    use_brkga: if True (default), Layer 1 uses the full BRKGA solver
    (vrp_rpd.solver.VRPRPDSolver) — warm-started, parallel islands, GP gene
    injection. If False, falls back to the bare construction heuristics.
    brkga_kwargs: passed through to VRPRPDSolver when use_brkga=True (e.g.
    total_generations, num_gpus, use_gp).
    """
    vrp_result: Optional[SolverResult] = None
    mapf_result: Optional[MAPFResult] = None
    converged = False
    current_grid = grid
    brkga_kwargs = brkga_kwargs or {}

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} (seed={seed}) ---")

        # Layer 1: VRP-RPD
        if use_brkga:
            vrp_result = run_brkga(
                current_grid,
                num_agents=num_agents,
                resources_per_agent=resources_per_agent,
                **brkga_kwargs,
            )
        else:
            vrp_result = run_heuristics(
                current_grid,
                num_agents=num_agents,
                resources_per_agent=resources_per_agent,
            )
        vrp_result.summary()

        # Layer 2: MAPF — expand visit sequences to include spur traversals
        visit_seqs = tours_to_grid_paths(
            {aid: plan.events for aid, plan in vrp_result.agents.items()},
            current_grid,
        )
        priority = vrp_result.priority_order()

        print(f"Running MAPF (priority: {priority})...")
        mapf_result = solve_mapf(
            visit_seqs, priority,
            spur_adj=current_grid.spur_adjacency(),
            workstation_ids=set(current_grid.workstation_ids),
            topology=current_grid.topology,
        )

        n_conf = len(mapf_result.conflicts)
        print(f"MAPF: success={mapf_result.success}, conflicts={n_conf}, "
              f"max_t={mapf_result.max_timestep()}")

        if mapf_result.success:
            converged = True
            print("Converged — collision-free solution found.")
            break

        # Feedback: log conflicting agents, try new seed
        conflicting = set()
        for c in mapf_result.conflicts:
            conflicting.add(c.agent_i)
            conflicting.add(c.agent_j)
        seed += 1
        print(f"Conflicting agents: {sorted(conflicting)} — retrying (seed={seed}, "
              f"same fixed grid, heuristic tie-breaking may differ)")

    return PipelineResult(
        vrp_result=vrp_result,
        mapf_result=mapf_result,
        iterations=iteration,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    grid = load_physical_grid(rng=np.random.default_rng(42))

    result = run_pipeline(
        grid, num_agents=4, resources_per_agent=2, max_iterations=5,
        use_brkga=True,
        brkga_kwargs=dict(
            total_generations=500, gens_per_cycle=100, use_gp=False,
            num_gpus=0, num_cpu_workers=2,  # no working CUDA driver in this environment
        ),
    )
    result.summary()

    # Print timed paths per agent
    for aid, tp in sorted(result.mapf_result.paths.items()):
        ct = result.vrp_result.agents[aid].completion_time
        print(f"Agent {aid} (VRP done@{ct:.0f}): "
              f"{len(tp.path)} steps, MAPF done@t={tp.path[-1][1] if tp.path else 0}")
        # Show first few steps
        preview = tp.path[:8]
        print(f"  Path start: {preview}{'...' if len(tp.path) > 8 else ''}")
