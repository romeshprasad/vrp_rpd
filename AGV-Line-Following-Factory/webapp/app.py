#!/usr/bin/env python3
"""
webapp/app.py — Hosted UI for configuring a grid/workstation layout,
running the BRKGA + MAPF pipeline, and downloading per-robot command
scripts. Wraps existing backend code only — no solver/bridge logic lives
here.

Usage:
    python3 app.py --host 0.0.0.0 --port 5050

Then share http://<server-ip>:5050 with whoever needs to use it (e.g. over
VPN, same as agv_testbed/web_viewer.py's existing sharing approach).
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

# AGV-Line-Following-Factory/ (for path_to_commands, mapf_path_to_commands,
# solve_and_save) and its parent vrp_rpd/ (for agv_testbed) both need to be
# importable, matching the sys.path setup already used by those scripts.
FACTORY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FACTORY_DIR))
sys.path.insert(0, str(FACTORY_DIR.parent))

from flask import Flask, render_template, request, jsonify, send_file
import io
import zipfile

import numpy as np

from agv_testbed.grid_env import build_grid_from_ui
from agv_testbed.pipeline import run_pipeline

from solve_and_save import build_solution_dict
from mapf_path_to_commands import mapf_path_to_commands, print_dropoff_pickup_table
from path_to_commands import garage_entry_commands, garage_exit_commands, turn_token


app = Flask(__name__)


def parse_workstation_edges(raw: list) -> list[tuple[int, int]]:
    """raw: list of [a, b] pairs (already 0-indexed node ids) from the UI."""
    edges = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Malformed workstation edge: {pair!r}")
        a, b = int(pair[0]), int(pair[1])
        edges.append((a, b))
    return edges


def generate_robot_scripts(solution: dict) -> dict[int, list[str]]:
    """Returns {robot_number: [command, ...]} for every agent in the solution."""
    grid_data = solution["grid"]
    spur_entry_set = set(grid_data["spur_entry_ids"])
    workstation_set = set(grid_data["workstation_ids"])

    from mapf_path_to_commands import spur_adjacency_from_between_nodes
    spur_adj = spur_adjacency_from_between_nodes(
        [tuple(p) for p in grid_data["between_nodes"]],
        grid_data["spur_entry_ids"],
        grid_data["workstation_ids"],
    )
    cols = grid_data["cols"]

    scripts: dict[int, list[str]] = {}
    for aid_str, agent in solution["agents"].items():
        robot_number = int(aid_str) + 1
        timed_path = agent["mapf_timed_path"]
        if not timed_path:
            continue

        node_path = [n for n, _t in timed_path]
        grid_commands, heading_at_depot = mapf_path_to_commands(
            node_path, spur_entry_set, workstation_set, spur_adj,
            start_heading="N", cols=cols,
        )

        final_turn = turn_token(heading_at_depot, "S")
        depot_turn_commands = [final_turn] if final_turn else []

        scripts[robot_number] = (
            garage_entry_commands()
            + grid_commands
            + depot_turn_commands
            + garage_exit_commands(robot_number)
        )
    return scripts


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json(force=True)

    try:
        rows = int(data["rows"])
        cols = int(data["cols"])
        workstation_edges = parse_workstation_edges(data["workstation_edges"])
        num_agents = int(data["num_agents"])
        resources_per_agent = int(data["resources_per_agent"])
        seed = int(data.get("seed", 42))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    try:
        grid = build_grid_from_ui(
            rows=rows, cols=cols,
            workstation_edges=workstation_edges,
            rng=np.random.default_rng(seed),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if num_agents < 1:
        return jsonify({"error": "Number of agents must be at least 1"}), 400
    if resources_per_agent < 1:
        return jsonify({"error": "Resources per agent must be at least 1"}), 400

    try:
        pipeline_result = run_pipeline(
            grid,
            num_agents=num_agents,
            resources_per_agent=resources_per_agent,
            max_iterations=5,
            use_brkga=True,
            brkga_kwargs=dict(
                total_generations=500, gens_per_cycle=100, use_gp=False,
                num_gpus=0, num_cpu_workers=2,
            ),
        )
    except Exception as e:
        return jsonify({"error": f"Solver failed: {e}"}), 500

    if not pipeline_result.converged:
        return jsonify({
            "error": (
                f"MAPF did not converge after {pipeline_result.iterations} attempts "
                f"({len(pipeline_result.mapf_result.conflicts)} conflicts remaining). "
                f"Try different agent/resource counts or workstation placement."
            )
        }), 409

    solution = build_solution_dict(grid, pipeline_result)
    scripts = generate_robot_scripts(solution)

    # Stash the solution in-process keyed by a token so /download can
    # regenerate without re-solving. Simplest viable approach for a
    # synchronous, single-result-at-a-time flow (no shared session state
    # needed beyond this request's lifetime for the results page).
    import uuid
    token = uuid.uuid4().hex
    _SOLUTIONS[token] = solution

    agents_view = []
    for aid_str, agent in sorted(solution["agents"].items(), key=lambda kv: int(kv[0])):
        robot_number = int(aid_str) + 1
        agents_view.append({
            "agent_id": int(aid_str),
            "robot_number": robot_number,
            "completion_time": round(agent["completion_time"], 1),
            "events": agent["events"],
            "command_count": len(scripts.get(robot_number, [])),
        })

    return jsonify({
        "token": token,
        "makespan": round(solution["vrp"]["makespan"], 1),
        "mapf_max_timestep": solution["mapf"]["max_timestep"],
        "iterations": solution["pipeline"]["iterations"],
        "agents": agents_view,
    })


# In-memory store: token -> solution dict. Cleared on process restart.
# Fine for a synchronous single-classroom-session tool; not durable storage.
_SOLUTIONS: dict[str, dict] = {}


@app.route("/download/<token>/<int:robot_number>")
def download_one(token: str, robot_number: int):
    solution = _SOLUTIONS.get(token)
    if solution is None:
        return jsonify({"error": "Unknown or expired solution token — please solve again."}), 404

    scripts = generate_robot_scripts(solution)
    commands = scripts.get(robot_number)
    if commands is None:
        return jsonify({"error": f"No script for robot {robot_number}"}), 404

    buf = io.BytesIO(("\n".join(commands) + "\n").encode("utf-8"))
    return send_file(
        buf, mimetype="text/plain", as_attachment=True,
        download_name=f"generated_alvik{robot_number}.txt",
    )


@app.route("/download/<token>/all")
def download_all(token: str):
    solution = _SOLUTIONS.get(token)
    if solution is None:
        return jsonify({"error": "Unknown or expired solution token — please solve again."}), 404

    scripts = generate_robot_scripts(solution)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for robot_number, commands in sorted(scripts.items()):
            zf.writestr(f"generated_alvik{robot_number}.txt", "\n".join(commands) + "\n")
    buf.seek(0)
    return send_file(
        buf, mimetype="application/zip", as_attachment=True,
        download_name="generated_alvik_scripts.zip",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
