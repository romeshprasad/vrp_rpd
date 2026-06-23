#!/usr/bin/env python3
"""
build_full_route.py — Assemble a full juan_supervisor.py command script for a
workstation visit sequence (WS01 -> WS03 -> WS04 -> WS06 -> WS08 -> WS07 ->
depot), using shortest-path BFS for grid legs and the fixed workstation-visit
template for each stop.

Scope: grid + workstation visits only (no depot ramp — route starts and ends
at node (0,0), per spec: starts facing N, ends facing S).
"""

from __future__ import annotations
import json
from collections import deque
from pathlib import Path

from path_to_commands import edge_direction, turn_token, workstation_visit_commands, COLS

ROWS = 8

WORKSTATIONS_JSON = Path(__file__).parent / "workstations.json"


def load_ws_edges(path: Path = WORKSTATIONS_JSON) -> dict[str, tuple[int, int]]:
    """
    workstations.json's between_nodes are 1-indexed (node 1 = grid (0,0)),
    while this module's node IDs are 0-indexed (node_id = row*COLS + col,
    starting at 0) — subtract 1 on load so everything downstream is 0-indexed.
    """
    data = json.loads(path.read_text())
    return {
        ws["id"]: (ws["between_nodes"][0] - 1, ws["between_nodes"][1] - 1)
        for ws in data["workstations"]
    }


WS_EDGES = load_ws_edges()

ROUTE_ORDER = ["WS01", "WS03", "WS04", "WS06", "WS08", "WS07"]


def neighbors(n: int) -> list[int]:
    r, c = n // COLS, n % COLS
    out = []
    if r > 0: out.append(n - COLS)
    if r < ROWS - 1: out.append(n + COLS)
    if c > 0: out.append(n - 1)
    if c < COLS - 1: out.append(n + 1)
    return out


def bfs_path(src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in neighbors(u):
            if v not in prev:
                prev[v] = u
                q.append(v)
    if dst not in prev:
        raise ValueError(f"No path {src} -> {dst}")
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def drive_grid_segment(node_ids: list[int], heading: str) -> tuple[list[str], str]:
    """Pure grid routing over a list of consecutive node ids. Returns (commands, heading_after)."""
    commands = []
    for a, b in zip(node_ids, node_ids[1:]):
        desired = edge_direction(a, b)
        tok = turn_token(heading, desired)
        if tok:
            commands.append(tok)
            heading = desired
        commands.append("FORWARD_UNTIL_RED")
    return commands, heading


def build_full_route(start: int, ws_order: list[str], ws_edges: dict, end: int,
                      start_heading: str = "N") -> tuple[list[str], str]:
    """Returns (commands, heading_after) — heading_after is the robot's facing on arrival at `end`."""
    commands = []
    heading = start_heading
    current = start

    for i, ws_id in enumerate(ws_order):
        a, b = ws_edges[ws_id]

        # Nearer endpoint of the workstation edge is the spur-entry approach.
        dist_a, dist_b = len(bfs_path(current, a)), len(bfs_path(current, b))
        entry, far = (a, b) if dist_a <= dist_b else (b, a)

        # Drive from current position to the spur-entry node.
        seg = bfs_path(current, entry)
        if len(seg) > 1:
            seg_cmds, heading = drive_grid_segment(seg, heading)
            commands.extend(seg_cmds)

        # The spur entry sits at the midpoint of the (entry, far) edge, which
        # always runs E/W — so the robot must be heading E or W (i.e. already
        # turned onto that edge, traveling toward `far`) before the fixed
        # workstation-visit template applies. If the last grid move that got
        # us to `entry` wasn't already E/W, turn onto the edge now.
        entry_heading = edge_direction(entry, far)
        tok = turn_token(heading, entry_heading)
        if tok:
            commands.append(tok)
        heading = entry_heading

        # Decide exit heading: whichever of the edge's two endpoints gets us
        # closer to the next target (next workstation, or depot on the last stop).
        if i + 1 < len(ws_order):
            next_a, next_b = ws_edges[ws_order[i + 1]]
            lookahead_targets = (next_a, next_b)
        else:
            lookahead_targets = (end,)

        cost_continue = min(len(bfs_path(far, t)) for t in lookahead_targets)
        cost_back = min(len(bfs_path(entry, t)) for t in lookahead_targets)
        exit_node = far if cost_continue <= cost_back else entry
        exit_heading = edge_direction(entry, far) if exit_node == far else edge_direction(far, entry)

        commands.extend(workstation_visit_commands(entry_heading, exit_heading))
        # workstation_visit_commands leaves the robot at the spur-entry
        # midpoint, facing exit_heading — still needs to drive the remaining
        # half of the edge out to the chosen grid node (entry or far).
        commands.append("FORWARD_UNTIL_RED")
        heading = exit_heading
        current = exit_node

    # Final leg back to depot.
    seg = bfs_path(current, end)
    if len(seg) > 1:
        seg_cmds, heading = drive_grid_segment(seg, heading)
        commands.extend(seg_cmds)

    return commands, heading


if __name__ == "__main__":
    from path_to_commands import garage_entry_commands, garage_exit_commands

    depot = 0
    robot_number = 1  # Alvik1

    grid_commands, heading_at_depot = build_full_route(
        depot, ROUTE_ORDER, WS_EDGES, depot, start_heading="N"
    )

    if heading_at_depot != "S":
        raise ValueError(
            f"Route returns to depot facing {heading_at_depot}, expected S "
            f"(garage_exit_commands assumes arrival at (0,0) facing South)"
        )

    full_script = (
        garage_entry_commands()
        + grid_commands
        + garage_exit_commands(robot_number)
    )

    print(f"Full Alvik{robot_number} script ({len(full_script)} lines): "
          f"garage -> {' -> '.join(ROUTE_ORDER)} -> garage\n")
    for i, c in enumerate(full_script, 1):
        print(f"{i:3d}  {c}")
