# AGV Grid Solver — Web UI

A hosted web app for configuring a grid/workstation layout, running the
BRKGA + MAPF pipeline, and downloading the generated per-robot command
scripts (`generated_alvikN.txt`). Built for handing off testing to someone
who shouldn't need to touch the Python code.

## What it does

1. Pick a grid size (rows x cols).
2. Click pairs of adjacent cells (same row only — workstations require an
   east-west edge so the spur can run south, matching the physical Alvik
   grid's convention) to place workstations.
3. Set number of agents and resources per agent.
4. Click **Solve** — runs the full pipeline (`agv_testbed.pipeline.run_pipeline`,
   BRKGA + MAPF with conflict retry) synchronously, in-process.
5. See the makespan and per-agent dropoff/pickup table.
6. Download each robot's command script individually, or all of them as a
   zip.

This is a thin wrapper — all solver/bridge logic lives in the existing
modules (`agv_testbed/`, `solve_and_save.py`, `mapf_path_to_commands.py`,
`path_to_commands.py`); the webapp only translates between HTTP requests
and those functions. Nothing here duplicates solver logic.

## Running it

```bash
cd AGV-Line-Following-Factory/webapp
python3 app.py --host 0.0.0.0 --port 5050
```

`--host 0.0.0.0` is required for anyone other than you (on the same
machine) to reach it — binding to `127.0.0.1` (the default) only accepts
local connections.

Then share `http://<server-ip-or-hostname>:5050` with whoever needs access
(student, collaborator, etc.) — same kind of network exposure as
`agv_testbed/web_viewer.py` already assumes (VPN, firewall rule, or
whatever you already use to reach this server). No additional
infrastructure (reverse proxy, auth, HTTPS) is set up here; add it if the
network this runs on isn't already trusted/private.

## Notes / limitations

- **Synchronous solve**: clicking Solve blocks until the pipeline finishes
  (typically 15-60+ seconds depending on agent/workstation count and BRKGA
  generation count). There's no background job queue — if you need
  multiple people running solves concurrently without blocking each other,
  Flask's dev server handles concurrent requests via separate threads/
  processes already, but each request still waits for its own solve.
- **No persistence across restarts**: solved results are kept in an
  in-memory dict (`_SOLUTIONS` in `app.py`) keyed by a one-time token handed
  back to the browser. Restarting the Flask process clears all of them —
  any download links from a previous session stop working. This is
  intentional for a lightweight classroom/testing tool, not meant as
  durable storage.
- **GPU**: BRKGA runs with `num_gpus=0, num_cpu_workers=2` hardcoded in
  `app.py`'s `/solve` handler, since this session found CUDA driver
  initialization unreliable in this dev environment even when
  `torch.cuda.device_count()` reports GPUs present. If running on a machine
  with a confirmed-working CUDA setup, edit the `brkga_kwargs` in
  `app.py::solve()` to use real GPU workers for faster/larger solves.
- **Workstation edges must be horizontal (same row)** — this isn't an
  arbitrary UI restriction, it's because the physical robot command bridge
  (`path_to_commands.py`) assumes every workstation's spur runs south from
  an east-west edge. A north-south edge has no valid spur direction and is
  rejected both client-side (JS) and server-side (`build_grid_from_ui` in
  `agv_testbed/grid_env.py`).
- **This UI's outputs are real robot command scripts** regardless of
  whether the grid size/layout matches the actual physical Alvik test grid
  — useful for testing routing logic on hypothetical layouts, but only the
  real 8x8 `workstations.json` layout corresponds to the taped-out physical
  setup.
