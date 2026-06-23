#!/usr/bin/env python3
"""
mapf_path_to_commands.py — Convert a real MAPF TimedPath (transit nodes +
virtual spur-entry/workstation IDs, with possible waits) into a per-agent
juan_supervisor.py command script, using path_to_commands.py's building
blocks (grid routing + fixed workstation-visit template).

This is the missing link between agv_testbed's pipeline output and the
physical Alvik robots: pipeline.py -> MAPFResult.paths[agent_id] -> here.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from path_to_commands import edge_direction, turn_token, workstation_visit_commands, COLS


def collapse_waits(node_path: list[int]) -> list[int]:
    """
    MAPF paths can repeat the same node across consecutive timesteps (a wait
    action, e.g. for collision avoidance or DWELL at a workstation). Collapse
    consecutive duplicates — the command bridge only cares about distinct
    node-to-node moves; DWELL/processing time is handled by the fixed
    workstation-visit template, not by literal wait-step counting (per
    earlier decision: wait/timing sync is deferred, not needed yet).
    """
    collapsed = [node_path[0]]
    for n in node_path[1:]:
        if n != collapsed[-1]:
            collapsed.append(n)
    return collapsed


def mapf_path_to_commands(
    node_path: list[int],
    spur_entry_ids: set[int],
    workstation_ids: set[int],
    spur_adj: dict[int, list[int]],
    start_heading: str = "N",
    cols: int = COLS,
) -> tuple[list[str], str]:
    """
    Convert a raw MAPF node path (may include waits + virtual spur/workstation
    IDs) into a flat juan_supervisor.py command list.

    node_path       : [node_id, ...] as returned by TimedPath (node_at each t)
                      or directly the list of (node_id, t) pairs' first elements.
    spur_entry_ids  : grid.spur_entry_ids, as a set
    workstation_ids : grid.workstation_ids, as a set
    spur_adj        : grid.spur_adjacency() — used to find a spur entry's two
                       real transit-edge endpoints (its non-workstation neighbors)
    cols            : grid width (node_id = row*cols + col) — must match the
                      grid the path was solved on; defaults to the physical
                      8x8 Alvik grid, pass grid.topology.cols for other sizes.

    Returns (commands, heading_after) — heading_after is the robot's facing
    on arrival at the path's final node (e.g. depot), needed by callers that
    splice on a fixed-heading template afterward (like garage_exit_commands,
    which assumes arrival facing South).
    """
    path = collapse_waits(node_path)

    # Map each spur entry to its two transit-edge endpoints (excluding the
    # workstation neighbor itself), so we know which real edge it sits on.
    entry_to_transit_endpoints: dict[int, list[int]] = {}
    for entry in spur_entry_ids:
        neighbors = spur_adj.get(entry, [])
        entry_to_transit_endpoints[entry] = [n for n in neighbors if n not in workstation_ids]

    commands: list[str] = []
    heading = start_heading
    i = 0
    while i < len(path) - 1:
        a, b = path[i], path[i + 1]

        if b in spur_entry_ids and a not in spur_entry_ids and a not in workstation_ids:
            # Transit node -> spur entry: turn onto the edge (entry_heading).
            # The spur entry sits on the edge (a, other_endpoint) -- use those
            # real transit nodes as the direction reference throughout, since
            # the virtual spur-entry ID itself has no (row, col).
            endpoints = entry_to_transit_endpoints[b]
            other_endpoint = endpoints[0] if endpoints[1] == a else endpoints[1]
            entry_heading = edge_direction(a, other_endpoint, cols=cols)

            tok = turn_token(heading, entry_heading)
            if tok:
                commands.append(tok)
            heading = entry_heading

            # Count consecutive workstation visits at this same spur entry
            # (one MAPF "spur_entry -> workstation -> spur_entry" triple per
            # visit; multiple visits happen when an agent has more than one
            # dropoff/pickup event at the same workstation in a row) — pure
            # lookahead, no commands emitted yet.
            j = i + 1  # currently at b (spur entry)
            n_visits = 0
            while (j + 2 < len(path) and path[j] == b
                   and path[j + 1] in workstation_ids and path[j + 2] == b):
                n_visits += 1
                j += 2  # consumed workstation -> spur entry

            if n_visits == 0:
                raise ValueError(
                    f"Spur entry {b} reached from {a} but next node "
                    f"({path[i+2] if i+2 < len(path) else 'END'}) is not its workstation"
                )

            exit_transit = path[j + 1] if j + 1 < len(path) else None
            if exit_transit == a:
                exit_heading = edge_direction(other_endpoint, a, cols=cols)
            elif exit_transit == other_endpoint:
                exit_heading = edge_direction(a, other_endpoint, cols=cols)
            elif exit_transit is None:
                exit_heading = entry_heading  # last stop — nothing further to infer from
            else:
                raise ValueError(
                    f"Spur entry {b}'s edge is ({a},{other_endpoint}) "
                    f"but path continues to unrelated node {exit_transit}"
                )

            # Stay parked at the workstation for all n_visits events (e.g. a
            # dropoff immediately followed by a pickup at the same station) —
            # DWELL once per visit rather than exiting/re-entering the spur
            # for each one.
            commands.extend(workstation_visit_commands(entry_heading, exit_heading, num_dwells=n_visits))

            commands.append("FORWARD_UNTIL_RED")  # spur-entry midpoint -> next transit node
            heading = exit_heading
            i = j + 1  # position at the spur entry node just before exit_transit
            continue

        else:
            # Plain transit-to-transit move.
            desired = edge_direction(a, b, cols=cols)
            tok = turn_token(heading, desired)
            if tok:
                commands.append(tok)
                heading = desired
            commands.append("FORWARD_UNTIL_RED")
            i += 1

    return commands, heading


def spur_adjacency_from_between_nodes(
    between_nodes: list[tuple[int, int]],
    spur_entry_ids: list[int],
    workstation_ids: list[int],
) -> dict[int, list[int]]:
    """Rebuild grid.spur_adjacency()'s output from saved plain data (no WarehouseGrid needed)."""
    adj: dict[int, list[int]] = {}
    for entry_id, ws_id, (a, b) in zip(spur_entry_ids, workstation_ids, between_nodes):
        adj.setdefault(a, []).append(entry_id)
        adj.setdefault(entry_id, []).append(a)
        adj.setdefault(b, []).append(entry_id)
        adj.setdefault(entry_id, []).append(b)
        adj.setdefault(entry_id, []).append(ws_id)
        adj.setdefault(ws_id, []).append(entry_id)
    return adj


def print_dropoff_pickup_table(solution: dict) -> None:
    print("\n=== Per-agent workstation visits (dropoff/pickup) ===")
    for aid_str, agent in sorted(solution["agents"].items(), key=lambda kv: int(kv[0])):
        robot_number = int(aid_str) + 1
        print(f"\nAgent {aid_str} (Alvik{robot_number}), done @ {agent['completion_time']:.1f}:")
        for ev in agent["events"]:
            print(f"  {ev['operation']:8s} at {ev['workstation_name']} "
                  f"(ws_id={ev['workstation_id']}, solver_idx={ev['solver_idx']})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate robot command scripts from a saved solve_and_save.py solution."
    )
    parser.add_argument("--input", default="solution.json",
                         help="Solution JSON produced by solve_and_save.py")
    parser.add_argument("--output-dir", default=".",
                         help="Directory to write generated_alvikN.txt files into")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from path_to_commands import garage_entry_commands, garage_exit_commands

    solution = json.loads(Path(args.input).read_text())

    print_dropoff_pickup_table(solution)

    if not solution["pipeline"]["converged"]:
        raise RuntimeError(
            f"{args.input} was saved from a non-converged MAPF run "
            f"({solution['mapf']['conflicts_remaining']} conflicts remaining) — "
            f"cannot generate a collision-free command script from it."
        )

    grid_data = solution["grid"]
    spur_entry_set = set(grid_data["spur_entry_ids"])
    workstation_set = set(grid_data["workstation_ids"])
    spur_adj = spur_adjacency_from_between_nodes(
        [tuple(p) for p in grid_data["between_nodes"]],
        grid_data["spur_entry_ids"],
        grid_data["workstation_ids"],
    )

    out_dir = Path(args.output_dir)
    for aid_str, agent in sorted(solution["agents"].items(), key=lambda kv: int(kv[0])):
        robot_number = int(aid_str) + 1
        timed_path = agent["mapf_timed_path"]
        if not timed_path:
            print(f"Agent {aid_str}: no path, skipping")
            continue

        node_path = [n for n, _t in timed_path]
        grid_commands, heading_at_depot = mapf_path_to_commands(
            node_path, spur_entry_set, workstation_set, spur_adj,
            start_heading="N", cols=grid_data.get("cols", COLS),
        )

        # garage_exit_commands assumes the robot arrives at depot (0,0)
        # facing South. The MAPF path can arrive heading E, W, or S
        # depending on which direction the last grid edge into depot came
        # from — insert one turn here if it didn't already land facing S.
        final_turn = turn_token(heading_at_depot, "S")
        depot_turn_commands = [final_turn] if final_turn else []

        full_script = (
            garage_entry_commands()
            + grid_commands
            + depot_turn_commands
            + garage_exit_commands(robot_number)
        )

        out_path = out_dir / f"generated_alvik{robot_number}.txt"
        out_path.write_text("\n".join(full_script) + "\n")
        print(f"\nAgent {aid_str} -> Alvik{robot_number}: "
              f"{len(full_script)} commands written to {out_path}")
