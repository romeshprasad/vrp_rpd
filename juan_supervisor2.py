#!/usr/bin/env python3
"""
juan_supervisor.py  —  Step-by-step command supervisor for juan_code.ino


juan_code.ino is a thin executor: it accepts ONE atomic command at a time on
<ROBOT_NAME>_cmd, executes it, and publishes status updates on
<ROBOT_NAME>_status (ROBOT_NAME is e.g. Alvik1, Alvik2, ... — each robot picks
its own name from its WiFi MAC address). The PC (this script) is the brain —
it holds the full route and sends the next command only after the robot
reports it has finished the current one.


Usage:
    python3 juan_supervisor.py --robot Alvik1 FORWARD_UNTIL_RED RIGHT_UNTIL_COLOR ...


    # or load a route from route_planner-style tokens (R/L/RED -> juan commands):
    python3 juan_supervisor.py --robot Alvik1 --tokens R,RED,RED,L,RED


    # or load a route from a file (one command per line, '#' comments allowed):
    python3 juan_supervisor.py --robot Alvik1 --file full_route_juan.txt


Timing options (optional, can appear anywhere on the command line):
    --csv PATH     Append one row per command (timestamp, robot, label, seq,
                   command, duration_s) to PATH. Point every run at the same
                   file to pool a whole session automatically.
    --label TEXT   A tag written into each CSV row, e.g. round01. Handy for
                   reconstructing session order after pooling.


At the end of every run the supervisor prints a per-command timing table
(n, mean, median, std, min, max). Durations are measured from dispatch to the
command's done-marker, i.e. the IDLE-to-IDLE interval.


Run one instance of this script per robot (each with its own --robot name and
route) to control multiple AGVs at once.


Recognized juan_code.ino commands (sent verbatim on <ROBOT_NAME>_cmd):
    FORWARD_UNTIL_RED     FORWARD_UNTIL_COLOR    BACKWARD_UNTIL_COLOR
    FORWARD_UNTIL_YELLOW  BACKWARD_UNTIL_YELLOW
    FORWARD_UNTIL_BLUE    BACKWARD_UNTIL_BLUE
    RIGHT_UNTIL_COLOR     LEFT_UNTIL_COLOR       ROTATE_180
    DWELL                 STOP                   RESET_POSE    GET_STATUS


The robot reports "BUSY ..." while executing and "IDLE" (or "DETECTED ...",
"TURN COMPLETE", "STOPPED", "POSE_RESET") when ready for the next command.
This script waits for one of the "done" markers below before sending the
next command in the sequence.
"""


from __future__ import annotations


import os
import sys
import csv
import time
import statistics
import threading
from datetime import datetime


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


# Status strings that mean "robot is ready for the next command".
# "TURN COMPLETE" / "ROTATE_180 COMPLETE" are informational only — juan_code.ino
# always publishes "IDLE" right after them in the same step, so "IDLE" alone
# is the real done signal. Including the others here causes a double-trigger:
# the first message advances to the next command, then the still-in-flight
# "IDLE" gets misread as that *next* command's done-marker.
DONE_MARKERS = ("IDLE", "STOPPED", "POSE_RESET", "ERROR")


# Map route_planner-style tokens to juan_code.ino commands (best-effort —
# only covers tokens juan_code.ino actually understands).
TOKEN_TO_CMD = {
    "RED":    "FORWARD_UNTIL_RED",
    "YELLOW": "FORWARD_UNTIL_YELLOW",
    "BLUE":   "FORWARD_UNTIL_BLUE",
    "R":      "RIGHT_UNTIL_COLOR",
    "L":      "LEFT_UNTIL_COLOR",
}




class JuanSupervisor(Node):
    def __init__(self, robot_name: str, commands: list[str],
                 csv_path: str | None = None, label: str | None = None):
        super().__init__(f"juan_supervisor_{robot_name.lower()}")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


        self._robot_name = robot_name
        self._commands = commands
        self._index = 0
        self._done_event = threading.Event()
        self._skip_next_idle = False


        # --- timing state ---
        self._durations: dict[str, list[float]] = {}   # command -> [seconds]
        self._seq = 0
        self._csv_path = csv_path
        self._label = label


        cmd_topic = f"{robot_name}_cmd"
        status_topic = f"{robot_name}_status"
        self._cmd_pub = self.create_publisher(String, cmd_topic, qos)
        self._status_sub = self.create_subscription(
            String, status_topic, self._on_status, qos)


        self.get_logger().info(
            f"Loaded {len(commands)} command(s) for {robot_name} "
            f"(cmd={cmd_topic}, status={status_topic})")
        if csv_path:
            self.get_logger().info(
                f"Appending per-command timings to {csv_path}"
                + (f" (label={label})" if label else ""))


    def _on_status(self, msg: String):
        text = msg.data.strip()
        self.get_logger().info(f"status: {text}")


        if text.startswith("ERROR"):
            self.get_logger().error(f"Robot reported error: {text}")
            self._done_event.set()
            return


        # The robot sometimes double-publishes "IDLE" right after
        # "TURN COMPLETE"/"ROTATE_180 COMPLETE", and the duplicate can
        # arrive just after we've sent the next command. A bare "IDLE"
        # immediately after sending can't be a real completion (no
        # command finishes in milliseconds), so discard exactly one.
        if self._skip_next_idle and text == "IDLE":
            self._skip_next_idle = False
            return
        self._skip_next_idle = False


        if any(text == m or text.startswith(m + " ") for m in DONE_MARKERS):
            self._done_event.set()


    def _send(self, cmd: str):
        self.get_logger().info(f"-> {cmd}")
        self._done_event.clear()
        self._skip_next_idle = True
        msg = String()
        msg.data = cmd
        self._cmd_pub.publish(msg)


    def _append_csv(self, seq: int, cmd: str, dur: float):
        new_file = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "robot", "label", "seq",
                            "command", "duration_s"])
            w.writerow([datetime.now().isoformat(timespec="milliseconds"),
                        self._robot_name, self._label or "", seq, cmd,
                        f"{dur:.4f}"])


    def _print_stats(self):
        if not self._durations:
            return
        log = self.get_logger()
        log.info("================= TIMING SUMMARY =================")
        log.info(f"{'command':<22}{'n':>3}{'mean':>9}{'median':>9}"
                 f"{'std':>9}{'min':>8}{'max':>8}")
        for cmd, ds in self._durations.items():
            n = len(ds)
            std = statistics.pstdev(ds) if n > 1 else 0.0
            log.info(f"{cmd:<22}{n:>3}{statistics.mean(ds):>9.3f}"
                     f"{statistics.median(ds):>9.3f}{std:>9.3f}"
                     f"{min(ds):>8.3f}{max(ds):>8.3f}")
        total = sum(sum(ds) for ds in self._durations.values())
        log.info(f"total measured command time: {total:.3f} s")
        log.info("=================================================")


    def run(self, timeout_sec: float = 30.0) -> bool:
        """Send each command in order, waiting for a done marker after each.
        Returns True if all commands completed, False on timeout/error."""
        for cmd in self._commands:
            t0 = time.monotonic()
            self._send(cmd)
            if not self._done_event.wait(timeout=timeout_sec):
                self.get_logger().error(
                    f"Timed out waiting for robot to finish: {cmd}")
                self._print_stats()
                return False
            dur = time.monotonic() - t0
            self._seq += 1
            self._durations.setdefault(cmd, []).append(dur)
            if self._csv_path:
                self._append_csv(self._seq, cmd, dur)
            self.get_logger().info(f"[timing] {cmd}: {dur:.3f} s")
        self.get_logger().info("All commands completed.")
        self._print_stats()
        return True




def commands_from_file(path: str) -> list[str]:
    """Read one command per line. Blank lines and lines starting with
    '#' (comments, e.g. '# --- WS visit #1 ---') are skipped."""
    commands = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            commands.append(line.upper())
    return commands




def tokens_to_commands(token_csv: str) -> list[str]:
    tokens = [t.strip().upper() for t in token_csv.split(",") if t.strip()]
    commands = []
    for tok in tokens:
        if tok not in TOKEN_TO_CMD:
            raise ValueError(
                f"Token '{tok}' has no juan_code.ino equivalent "
                f"(supported: {sorted(TOKEN_TO_CMD)})")
        commands.append(TOKEN_TO_CMD[tok])
    return commands




def _pop_flag(args: list[str], name: str) -> str | None:
    """Remove '--flag value' from args anywhere it appears; return value."""
    if name in args:
        i = args.index(name)
        if i + 1 >= len(args):
            print(f"{name} needs a value")
            sys.exit(1)
        val = args[i + 1]
        del args[i:i + 2]
        return val
    return None




def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)


    # Optional timing flags may appear anywhere; strip them out first.
    csv_path = _pop_flag(args, "--csv")
    label = _pop_flag(args, "--label")


    if not args or args[0] != "--robot" or len(args) < 3:
        print("usage: juan_supervisor.py --robot AlvikN "
              "<commands | --tokens ... | --file ...> [--csv PATH] [--label TEXT]")
        sys.exit(1)
    robot_name = args[1]
    args = args[2:]


    if args[0] == "--tokens":
        if len(args) != 2:
            print("usage: juan_supervisor.py --robot AlvikN --tokens R,RED,RED,L,RED")
            sys.exit(1)
        commands = tokens_to_commands(args[1])
    elif args[0] == "--file":
        if len(args) != 2:
            print("usage: juan_supervisor.py --robot AlvikN --file route.txt")
            sys.exit(1)
        commands = commands_from_file(args[1])
    else:
        commands = [a.upper() for a in args]


    rclpy.init()
    node = JuanSupervisor(robot_name, commands, csv_path=csv_path, label=label)


    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()


    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


    sys.exit(0 if ok else 1)




if __name__ == "__main__":
    main()



