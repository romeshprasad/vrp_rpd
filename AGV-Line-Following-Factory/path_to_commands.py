#!/usr/bin/env python3
"""
path_to_commands.py — Translate a MAPF/grid node path into juan_supervisor.py commands.

Two pieces, kept separate:

1. Grid routing (scalable, no special cases): given consecutive node IDs,
   compute the world-frame direction of each move, turn (R/L) only if the
   heading changed, then FORWARD_UNTIL_RED. This is the only logic that
   needs to know about the grid/MAPF path.

2. Workstation visit (fixed constant template): entering a workstation is
   always the same 7-line shape —
       FORWARD_UNTIL_YELLOW         (arrive at spur entry)
       <turn in>                    (face the spur — depends on incoming heading)
       FORWARD_UNTIL_YELLOW         (arrive at workstation)
       ROTATE_180
       DWELL
       FORWARD_UNTIL_YELLOW         (back to spur entry)
       <turn out>                   (face wherever the next leg requires)
   Only the two turn directions vary call to call; everything else is fixed.
"""

from __future__ import annotations

COLS = 8
ORDER = ["N", "E", "S", "W"]  # clockwise order


def edge_direction(a: int, b: int, cols: int = COLS) -> str:
    """World-frame direction of the move from node a to node b. Raises if not grid-adjacent."""
    ra, ca = a // cols, a % cols
    rb, cb = b // cols, b % cols
    if rb == ra + 1 and cb == ca:
        return "N"
    if rb == ra - 1 and cb == ca:
        return "S"
    if rb == ra and cb == ca + 1:
        return "E"
    if rb == ra and cb == ca - 1:
        return "W"
    raise ValueError(f"{a} -> {b} is not a single grid-adjacent move")


def turn_token(current: str, desired: str) -> str | None:
    """Command needed to rotate from `current` heading to `desired`, or None if already facing it."""
    if current == desired:
        return None
    diff = (ORDER.index(desired) - ORDER.index(current)) % 4
    if diff == 1:
        return "RIGHT_UNTIL_COLOR"
    if diff == 3:
        return "LEFT_UNTIL_COLOR"
    raise ValueError(
        f"180-degree turn requested for plain grid move {current} -> {desired} "
        f"(ROTATE_180 only happens inside a workstation visit, not on the grid)"
    )


def path_to_commands(node_path: list[int], start_heading: str = "N") -> list[str]:
    """
    Convert a list of consecutive grid node IDs into a juan_supervisor.py
    command list. Pure routing — no workstation awareness.
    """
    commands = []
    heading = start_heading
    for a, b in zip(node_path, node_path[1:]):
        desired = edge_direction(a, b)
        tok = turn_token(heading, desired)
        if tok:
            commands.append(tok)
            heading = desired
        commands.append("FORWARD_UNTIL_RED")
    return commands


# ---------------------------------------------------------------------------
# Workstation visit — fixed constant template
# ---------------------------------------------------------------------------

def workstation_visit_commands(entry_heading: str, exit_heading: str, num_dwells: int = 1) -> list[str]:
    """
    The constant 7-line shape for entering and exiting a workstation.
    entry_heading : robot's heading on arrival at the spur-entry midpoint (E or W)
    exit_heading  : heading the robot needs after coming back out, to continue
                    toward whatever node comes next in the path (E or W)
    num_dwells    : number of DWELL commands issued while parked at the
                    workstation (>1 for back-to-back events at the same
                    workstation, e.g. a dropoff immediately followed by a
                    pickup — stay parked and DWELL again rather than
                    exiting/re-entering the spur for each event).
    Returns the command list; robot ends facing `exit_heading`.
    """
    if num_dwells < 1:
        raise ValueError(f"num_dwells must be >= 1, got {num_dwells}")

    turn_in = turn_token(entry_heading, "S")
    turn_out = turn_token("N", exit_heading)

    commands = ["FORWARD_UNTIL_YELLOW"]
    if turn_in:
        commands.append(turn_in)
    commands += ["FORWARD_UNTIL_YELLOW", "ROTATE_180"]
    commands += ["DWELL"] * num_dwells
    commands.append("FORWARD_UNTIL_YELLOW")
    if turn_out:
        commands.append(turn_out)
    return commands


# ---------------------------------------------------------------------------
# Garage entry / exit — fixed templates, parameterized only by robot number
# ---------------------------------------------------------------------------
# All robots start parked facing North, in parallel stalls, sharing one lane
# out to the grid. Entry is identical for every robot. Exit is identical in
# shape but AlvikN passes N-1 other stalls' blue markers before turning into
# its own (the Nth) and docking.

def garage_entry_commands() -> list[str]:
    """
    Drive out of the garage stall onto the grid, arriving at node (0,0)
    facing North. Same for every robot.
    """
    return [
        "FORWARD_UNTIL_BLUE",   # out of the stall to the garage-exit line
        "RIGHT_UNTIL_COLOR",    # turn to face the depot lane
        "FORWARD_UNTIL_RED",    # depot's own red marker
        "RIGHT_UNTIL_COLOR",    # turn onto the grid
        "FORWARD_UNTIL_RED",    # arrive at (0,0), facing N
    ]


def garage_exit_commands(robot_number: int) -> list[str]:
    """
    From (0,0) facing South, drive back to AlvikN's own parking stall and
    park facing North again. robot_number is 1-indexed (Alvik1, Alvik2, ...).
    AlvikN passes (robot_number - 1) other stalls' blue markers before
    turning into its own.
    """
    if robot_number < 1:
        raise ValueError(f"robot_number must be >= 1, got {robot_number}")

    commands = [
        "FORWARD_UNTIL_RED",   # leave (0,0) heading south
        "LEFT_UNTIL_COLOR",    # turn onto the shared garage lane
    ]
    commands += ["FORWARD_UNTIL_BLUE"] * robot_number  # pass N-1 stalls, reach the Nth marker
    commands += [
        "LEFT_UNTIL_COLOR",    # turn into its own stall
        "FORWARD_UNTIL_BLUE",  # dock fully inside
        "ROTATE_180",          # face North again, ready for next dispatch
    ]
    return commands


if __name__ == "__main__":
    # Hand-picked test path on the 8x8 grid (row*8+col):
    #   0 -> 1 -> 2   : straight east along row 0
    #   2 -> 10        : turn to face north, go straight
    #   10 -> 11       : turn to face east again
    node_path = [0, 1, 2, 10, 11]

    print(f"Input node path : {node_path}")
    coords = [(n // COLS, n % COLS) for n in node_path]
    print(f"As (row, col)   : {coords}")

    commands = path_to_commands(node_path, start_heading="N")
    print("\nOutput commands (pure routing):")
    for c in commands:
        print(f"  {c}")

    print("\n--- Workstation visit example ---")
    print("Entering heading E, need to exit heading W:")
    for c in workstation_visit_commands("E", "W"):
        print(f"  {c}")
