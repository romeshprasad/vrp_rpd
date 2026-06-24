# Multi-Agent Traffic Control

Centralized fleet launcher and supervisor for a 4-robot Arduino Alvik AGV system running on ROS 2.

`multi_agent_traffic_control_v1.py` replaces the original single-robot `juan_supervisor.py` workflow. One Python process owns all robot publishers and subscribers, staggers robot launches to avoid depot collisions, and logs per-command timing data to CSV.

---

## Prerequisites

| Requirement | Version tested |
|---|---|
| Ubuntu 24.04 (or WSL2) | 24.04 LTS |
| ROS 2 Jazzy | jazzy |
| Python 3.10+ | 3.10 |
| micro-ROS agent | see main README |

All four Alviks must be powered on, connected to WiFi, and the micro-ROS agent must be running before launching this script.

---

## Quick Start

### 1. Start the micro-ROS agent (own terminal)

```bash
micro-ros-agent udp4 --port 8888
```

### 2. Source ROS 2 and navigate to this directory

```bash
source /opt/ros/jazzy/setup.bash
cd ORACLE_VM/
```

### 3. Run the traffic controller

```bash
python3 multi_agent_traffic_control_v1.py
```

---

## Interactive Launch Flow

The script is fully interactive — no command-line flags required.

**Step 1 — Route selection**

```
Use generated_alvik#.txt route files from the current directory? [Y/n]:
```

- Press **Enter** (or type `y`) to auto-discover `generated_alvik1.txt` through `generated_alvik4.txt` in the current directory.
- Type `n` to enter each robot's route file path manually.

**Step 2 — Launch plan preview**

The script prints the staggered launch schedule before any robot moves:

```
Launch plan:
  Wave 1 at t=0.0s : Alvik1, Alvik4
  Wave 2 at t=12.0s: Alvik2
  Wave 3 at t=18.0s: Alvik3
```

**Step 3 — Two-stage confirmation**

```
Proceed to launch the fleet? [y/N]:
Type "LAUNCH" to confirm robot motion:
```

Both prompts must be confirmed before any robot receives a command. Press **Ctrl+C** at any time during a run to emergency-stop all robots.

---

## Route Files

Route files (`generated_alvik#.txt`) contain one primitive command per line:

```
FORWARD_UNTIL_BLUE
RIGHT_UNTIL_COLOR
FORWARD_UNTIL_RED
DWELL
ROTATE_180
...
```

The auto-discovery expects files named exactly `generated_alvik1.txt`, `generated_alvik2.txt`, `generated_alvik3.txt`, and `generated_alvik4.txt` in the directory where you run the script.

To use custom route files, answer `n` at the first prompt and enter each file path manually.

---

## Output

**Terminal** — live per-robot command progress with timestamps:

```
[Alvik1] CMD  1/18 FORWARD_UNTIL_BLUE      → sent at t=0.12s
[Alvik1] DONE 1/18 FORWARD_UNTIL_BLUE      ← ack  at t=2.38s  (2.26s)
```

**CSV log** — `fleet_command_timings.csv` is written in the working directory after each run:

```
robot,command_index,command,sent_at_sec,ack_at_sec,duration_sec
Alvik1,1,FORWARD_UNTIL_BLUE,0.12,2.38,2.26
...
```

**Summary** — printed at the end of the run:

```
=== Run summary ===
Alvik1   18/18 commands  OK   total=47.3s
Alvik2   18/18 commands  OK   total=49.1s
Alvik3   18/18 commands  OK   total=51.4s
Alvik4   18/18 commands  FAIL total=22.0s  (timeout on cmd 7)
```

---

## Stopping Mid-Run

Press **Ctrl+C** in the terminal running the script. The controller sends a `STOP` command to every robot before shutting down.

---

## Differences from `juan_supervisor.py`

| Feature | `juan_supervisor.py` | `multi_agent_traffic_control_v1.py` |
|---|---|---|
| Robots per process | 1 | Up to 4 |
| Launch staggering | Manual (separate terminals) | Automatic wave schedule |
| Timing CSV | No | Yes |
| Depot collision avoidance | None | Wave-based gap enforcement |

---

## Related Files

| File | Purpose |
|---|---|
| `generated_alvik1-4.txt` | Pre-computed route files for each robot |
| `agv_robots.yaml` | Robot names and MAC addresses |
| `workstations.json` | Workstation positions on the 8×8 grid |
| `fleet_command_timings.csv` | Output timing log from the last run |
| `dispatch_node.py` | Lower-level ROS 2 dispatch node (used separately) |

See the main [README.md](README.md) for full system setup, Arduino flashing, and network configuration.
