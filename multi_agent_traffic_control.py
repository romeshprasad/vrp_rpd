#!/usr/bin/env python3
"""
multi_agent_traffic_control.py

Centralized fleet launcher/supervisor for Alvik primitive-command route files.

This is the next step after running one juan_supervisor.py terminal per robot:
one Python process owns all robot publishers/subscribers and launches the fleet
with a safe depot schedule.

Important scope note:
    This version centralizes execution and launch staggering, but it does not
    yet perform true graph-level MAPF. The current generated_alvik#.txt files
    contain primitive commands, not annotated graph actions such as:

        from_node,to_node,edge_id,expected_duration,command

    Full MAPF reservations should be added once the route compiler preserves
    graph metadata for each primitive command or command bundle.
"""

from __future__ import annotations

import csv
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


ROBOT_NAMES = [f"Alvik{i}" for i in range(1, 5)]

# Alvik1 and Alvik4 may leave together based on the current depot geometry.
# Alvik2 and Alvik3 are staggered to avoid depot/node-0/first-turn conflicts.
DEFAULT_LAUNCH_WAVES = [
    ["Alvik1", "Alvik4"],
    ["Alvik2"],
    ["Alvik3"],
]

DEFAULT_COMMAND_TIMEOUT_SEC = 45.0
IGNORE_STALE_IDLE_SEC = 0.35
DEFAULT_MIN_SHARED_DEPOT_GAP_SEC = 12.0
DEFAULT_DEPOT_CLEAR_BUFFER_SEC = 1.0

# Direct measurements from Terminal Output.docx where full logs were available.
# This is the time from first launch command through the first completed
# FORWARD_UNTIL_RED, which is the first shared-risk depot/grid-entry segment.
FIRST_RED_RESERVATION_SEC = {
    "Alvik1": 7.4,
    "Alvik2": 9.1,
    "Alvik3": 10.8,
    "Alvik4": 12.0,
}

# Robot-specific command duration estimates from the current timing data.
# These are used for prediction only; actual IDLE feedback still controls
# command progression.
COMMAND_DURATION_ESTIMATES_SEC = {
    "Alvik1": {
        "FORWARD_UNTIL_BLUE": 2.2,
        "RIGHT_UNTIL_COLOR": 2.7,
        "FORWARD_UNTIL_RED": 3.2,
        "FORWARD_UNTIL_YELLOW": 1.7,
        "LEFT_UNTIL_COLOR": 2.7,
        "ROTATE_180": 5.0,
        "DWELL": 2.1,
    },
    "Alvik2": {
        "FORWARD_UNTIL_BLUE": 2.1,
        "RIGHT_UNTIL_COLOR": 2.7,
        "FORWARD_UNTIL_RED": 3.3,
        "FORWARD_UNTIL_YELLOW": 1.6,
        "LEFT_UNTIL_COLOR": 2.7,
        "ROTATE_180": 5.0,
        "DWELL": 2.2,
    },
    "Alvik3": {
        "FORWARD_UNTIL_BLUE": 2.2,
        "RIGHT_UNTIL_COLOR": 2.7,
        "FORWARD_UNTIL_RED": 3.3,
        "FORWARD_UNTIL_YELLOW": 2.1,
        "LEFT_UNTIL_COLOR": 2.7,
        "ROTATE_180": 5.0,
        "DWELL": 2.1,
    },
    "Alvik4": {
        "FORWARD_UNTIL_BLUE": 2.0,
        "RIGHT_UNTIL_COLOR": 2.7,
        "FORWARD_UNTIL_RED": 3.3,
        "FORWARD_UNTIL_YELLOW": 2.4,
        "LEFT_UNTIL_COLOR": 2.7,
        "ROTATE_180": 5.0,
        "DWELL": 2.2,
    },
}

FALLBACK_COMMAND_DURATION_SEC = {
    "FORWARD_UNTIL_BLUE": 2.2,
    "RIGHT_UNTIL_COLOR": 2.7,
    "FORWARD_UNTIL_RED": 3.3,
    "FORWARD_UNTIL_YELLOW": 2.2,
    "LEFT_UNTIL_COLOR": 2.7,
    "ROTATE_180": 5.0,
    "DWELL": 2.2,
}

# Same command token can correspond to different physical edge lengths.
# These sequence-specific overrides keep early launch predictions realistic
# while the route files are still raw primitive commands instead of graph
# annotated moves.
COMMAND_SEQUENCE_DURATION_OVERRIDES_SEC = {
    "Alvik4": {
        1: 2.18,  # FORWARD_UNTIL_BLUE, measured 2.176 s
        2: 2.57,  # RIGHT_UNTIL_COLOR, measured 2.569 s
        3: 7.18,  # first FORWARD_UNTIL_RED, measured 7.177 s
        4: 2.54,  # RIGHT_UNTIL_COLOR, measured 2.537 s
        5: 4.48,  # second FORWARD_UNTIL_RED, measured 4.484 s
    },
}

DONE_MARKERS = ("IDLE", "STOPPED", "POSE_RESET")
ERROR_MARKERS = ("ERROR", "EMERGENCY_STOP")


@dataclass
class RobotRuntime:
    name: str
    route_file: Path
    commands: list[str]
    done_event: threading.Event = field(default_factory=threading.Event)
    launch_event: threading.Event = field(default_factory=threading.Event)
    last_status: str = ""
    active_command: str | None = None
    command_start_time: float = 0.0
    active_expected_duration: float = 0.0
    ignore_idle_until: float = 0.0
    completed: bool = False
    failed: bool = False
    durations: dict[str, list[float]] = field(default_factory=dict)
    estimated_route_elapsed_sec: float = 0.0
    actual_route_elapsed_sec: float = 0.0


@dataclass(frozen=True)
class LaunchPlanItem:
    wave_number: int
    robots: list[str]
    release_at_sec: float
    wait_after_previous_sec: float
    basis: str


def read_route_file(path: Path) -> list[str]:
    commands: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            commands.append(line.upper())
    return commands


def discover_generated_routes(cwd: Path) -> dict[str, Path]:
    routes: dict[str, Path] = {}
    for i in range(1, 5):
        path = cwd / f"generated_alvik{i}.txt"
        if path.exists():
            routes[f"Alvik{i}"] = path
    return routes


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def prompt_routes() -> dict[str, Path]:
    cwd = Path.cwd()
    use_generated = prompt_yes_no(
        "Use generated_alvik#.txt route files from the current directory?",
        default=True,
    )

    routes: dict[str, Path] = {}

    if use_generated:
        routes = discover_generated_routes(cwd)
        if not routes:
            print("No generated_alvik#.txt files were found in the current directory.")
            print("Switching to manual route entry.")
            use_generated = False

    if not use_generated:
        print("Enter route files for each robot. Leave blank to skip that robot.")
        for robot in ROBOT_NAMES:
            raw = input(f"{robot} route file: ").strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = cwd / path
            routes[robot] = path

    return routes


def validate_routes(route_paths: dict[str, Path]) -> dict[str, RobotRuntime]:
    robots: dict[str, RobotRuntime] = {}
    for robot, path in sorted(route_paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{robot} route file not found: {path}")
        commands = read_route_file(path)
        if not commands:
            raise ValueError(f"{robot} route file is empty: {path}")
        robots[robot] = RobotRuntime(robot, path, commands)
    return robots


def build_launch_waves(robots: dict[str, RobotRuntime]) -> list[list[str]]:
    waves: list[list[str]] = []
    already: set[str] = set()

    for wave in DEFAULT_LAUNCH_WAVES:
        present = [robot for robot in wave if robot in robots]
        if present:
            waves.append(present)
            already.update(present)

    for robot in sorted(robots):
        if robot not in already:
            waves.append([robot])

    return waves


def command_duration_estimate(robot_name: str, command: str, seq: int | None = None) -> float:
    if seq is not None:
        override = COMMAND_SEQUENCE_DURATION_OVERRIDES_SEC.get(robot_name, {}).get(seq)
        if override is not None:
            return override

    robot_estimates = COMMAND_DURATION_ESTIMATES_SEC.get(robot_name, {})
    return robot_estimates.get(command, FALLBACK_COMMAND_DURATION_SEC.get(command, 3.0))


def route_duration_estimate(robot: RobotRuntime) -> float:
    return sum(
        command_duration_estimate(robot.name, command, seq)
        for seq, command in enumerate(robot.commands, start=1)
    )


def first_red_reservation(robot_name: str) -> float:
    return FIRST_RED_RESERVATION_SEC.get(robot_name, DEFAULT_MIN_SHARED_DEPOT_GAP_SEC)


def build_timed_launch_plan(waves: list[list[str]]) -> list[LaunchPlanItem]:
    plan: list[LaunchPlanItem] = []
    release_at_sec = 0.0

    for i, wave in enumerate(waves, start=1):
        if i == 1:
            wait_after_previous_sec = 0.0
            basis = "Initial release wave"
        else:
            previous_wave = waves[i - 2]
            previous_clear = max(first_red_reservation(robot) for robot in previous_wave)
            wait_after_previous_sec = max(
                DEFAULT_MIN_SHARED_DEPOT_GAP_SEC,
                previous_clear + DEFAULT_DEPOT_CLEAR_BUFFER_SEC,
            )
            release_at_sec += wait_after_previous_sec
            basis = (
                f"Previous wave first-red reservation max={previous_clear:.1f}s "
                f"+ buffer={DEFAULT_DEPOT_CLEAR_BUFFER_SEC:.1f}s, "
                f"minimum shared-depot gap={DEFAULT_MIN_SHARED_DEPOT_GAP_SEC:.1f}s"
            )

        plan.append(
            LaunchPlanItem(
                wave_number=i,
                robots=wave,
                release_at_sec=release_at_sec,
                wait_after_previous_sec=wait_after_previous_sec,
                basis=basis,
            )
        )

    return plan


class MultiAgentTrafficControl(Node):
    def __init__(
        self,
        robots: dict[str, RobotRuntime],
        csv_path: Path | None = None,
        command_timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
    ):
        super().__init__("multi_agent_traffic_control")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.robots = robots
        self.csv_path = csv_path
        self.command_timeout_sec = command_timeout_sec
        self._lock = threading.Lock()
        self._publishers = {}
        self._subscriptions = []

        for robot in robots.values():
            cmd_topic = f"{robot.name}_cmd"
            status_topic = f"{robot.name}_status"
            self._publishers[robot.name] = self.create_publisher(String, cmd_topic, qos)
            self._subscriptions.append(
                self.create_subscription(
                    String,
                    status_topic,
                    self._make_status_callback(robot.name),
                    qos,
                )
            )
            self.get_logger().info(
                f"{robot.name}: cmd={cmd_topic}, status={status_topic}, "
                f"commands={len(robot.commands)}, file={robot.route_file}"
            )

    def _make_status_callback(self, robot_name: str) -> Callable[[String], None]:
        def callback(msg: String):
            text = msg.data.strip()
            now = time.monotonic()
            robot = self.robots[robot_name]

            with self._lock:
                robot.last_status = text

                if any(text.startswith(prefix) for prefix in ERROR_MARKERS):
                    self.get_logger().error(f"{robot_name} status: {text}")
                    robot.failed = True
                    robot.done_event.set()
                    return

                if text == "IDLE" and now < robot.ignore_idle_until:
                    return

                if any(text == marker or text.startswith(marker + " ") for marker in DONE_MARKERS):
                    robot.done_event.set()

            self.get_logger().info(f"{robot_name} status: {text}")

        return callback

    def send_command(self, robot: RobotRuntime, seq: int, command: str):
        msg = String()
        msg.data = command
        expected_duration = command_duration_estimate(robot.name, command, seq)

        with self._lock:
            robot.active_command = command
            robot.command_start_time = time.monotonic()
            robot.active_expected_duration = expected_duration
            robot.ignore_idle_until = robot.command_start_time + IGNORE_STALE_IDLE_SEC
            robot.done_event.clear()

        predicted_finish = robot.actual_route_elapsed_sec + expected_duration
        self.get_logger().info(
            f"{robot.name} -> {command} "
            f"(expected={expected_duration:.3f}s, predicted_route_elapsed={predicted_finish:.3f}s)"
        )
        self._publishers[robot.name].publish(msg)

    def run_robot_route(self, robot_name: str):
        robot = self.robots[robot_name]
        self.get_logger().info(f"{robot.name}: waiting for launch release")
        robot.launch_event.wait()
        self.get_logger().info(f"{robot.name}: launched")

        for seq, command in enumerate(robot.commands, start=1):
            if robot.failed:
                break

            self.send_command(robot, seq, command)
            finished = robot.done_event.wait(timeout=self.command_timeout_sec)
            duration = time.monotonic() - robot.command_start_time

            if not finished:
                robot.failed = True
                self.get_logger().error(
                    f"{robot.name}: timeout on seq={seq}, command={command}, "
                    f"duration={duration:.3f}s"
                )
                self.send_stop(robot)
                break

            robot.durations.setdefault(command, []).append(duration)
            robot.estimated_route_elapsed_sec += robot.active_expected_duration
            robot.actual_route_elapsed_sec += duration
            schedule_error = robot.actual_route_elapsed_sec - robot.estimated_route_elapsed_sec

            if self.csv_path:
                self.append_csv(
                    robot,
                    seq,
                    command,
                    robot.active_expected_duration,
                    duration,
                    schedule_error,
                )
            self.get_logger().info(
                f"{robot.name}: seq={seq}/{len(robot.commands)} "
                f"{command} completed in {duration:.3f}s "
                f"(expected={robot.active_expected_duration:.3f}s, "
                f"route_error={schedule_error:+.3f}s)"
            )

        robot.completed = not robot.failed
        if robot.completed:
            self.get_logger().info(f"{robot.name}: route complete")
        else:
            self.get_logger().error(f"{robot.name}: route failed")

    def send_stop(self, robot: RobotRuntime):
        msg = String()
        msg.data = "STOP"
        self._publishers[robot.name].publish(msg)

    def append_csv(
        self,
        robot: RobotRuntime,
        seq: int,
        command: str,
        expected_duration: float,
        actual_duration: float,
        schedule_error: float,
    ):
        if not self.csv_path:
            return
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow([
                    "timestamp",
                    "robot",
                    "route_file",
                    "seq",
                    "command",
                    "expected_duration_s",
                    "actual_duration_s",
                    "command_error_s",
                    "estimated_route_elapsed_s",
                    "actual_route_elapsed_s",
                    "route_error_s",
                ])
            writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                robot.name,
                str(robot.route_file),
                seq,
                command,
                f"{expected_duration:.4f}",
                f"{actual_duration:.4f}",
                f"{actual_duration - expected_duration:.4f}",
                f"{robot.estimated_route_elapsed_sec:.4f}",
                f"{robot.actual_route_elapsed_sec:.4f}",
                f"{schedule_error:.4f}",
            ])

    def print_summary(self):
        log = self.get_logger()
        log.info("================= FLEET SUMMARY =================")
        for robot in self.robots.values():
            status = "OK" if robot.completed else "FAILED"
            total = sum(sum(ds) for ds in robot.durations.values())
            log.info(
                f"{robot.name}: {status}, measured command time={total:.3f}s, "
                f"estimated={robot.estimated_route_elapsed_sec:.3f}s, "
                f"route_error={robot.actual_route_elapsed_sec - robot.estimated_route_elapsed_sec:+.3f}s"
            )
            for command, ds in robot.durations.items():
                n = len(ds)
                std = statistics.pstdev(ds) if n > 1 else 0.0
                log.info(
                    f"  {command:<24} n={n:>3} mean={statistics.mean(ds):>7.3f} "
                    f"median={statistics.median(ds):>7.3f} std={std:>7.3f}"
                )
        log.info("=================================================")


def print_launch_plan(
    robots: dict[str, RobotRuntime],
    launch_plan: list[LaunchPlanItem],
):
    print("\nLoaded routes:")
    for robot in sorted(robots.values(), key=lambda r: r.name):
        estimated_total = route_duration_estimate(robot)
        first_red = first_red_reservation(robot.name)
        print(
            f"  {robot.name}: {robot.route_file} ({len(robot.commands)} commands, "
            f"est total={estimated_total:.1f}s, first-red reservation={first_red:.1f}s)"
        )

    print("\nMeasured launch plan:")
    for item in launch_plan:
        if item.wave_number == 1:
            print(f"  Wave {item.wave_number}: {', '.join(item.robots)} at t=0.0s")
        else:
            print(
                f"  Wave {item.wave_number}: {', '.join(item.robots)} "
                f"at t={item.release_at_sec:.1f}s "
                f"(wait +{item.wait_after_previous_sec:.1f}s)"
            )
        print(f"    Basis: {item.basis}")

    print("\nNote:")
    print("  This launch plan uses measured first-red timing to reserve the depot/grid-entry segment.")
    print("  Actual IDLE feedback still controls when each robot receives its next command.")
    print("  Full MAPF node/edge reservations require annotated graph actions, not only raw command files.")


def main():
    routes = prompt_routes()
    if not routes:
        print("No routes selected. Exiting.")
        sys.exit(1)

    try:
        robots = validate_routes(routes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Route error: {exc}")
        sys.exit(1)

    waves = build_launch_waves(robots)
    launch_plan = build_timed_launch_plan(waves)
    print_launch_plan(robots, launch_plan)

    if not prompt_yes_no("\nProceed to launch the fleet?", default=False):
        print("Launch canceled.")
        sys.exit(0)

    typed = input('Type "LAUNCH" to confirm robot motion: ').strip()
    if typed != "LAUNCH":
        print("Launch canceled.")
        sys.exit(0)

    csv_path = Path("fleet_command_timings.csv")

    rclpy.init()
    node = MultiAgentTrafficControl(robots, csv_path=csv_path)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    worker_threads = [
        threading.Thread(target=node.run_robot_route, args=(robot_name,), daemon=True)
        for robot_name in robots
    ]
    for thread in worker_threads:
        thread.start()

    try:
        start_time = time.monotonic()
        for item in launch_plan:
            sleep_until = start_time + item.release_at_sec
            sleep_for = sleep_until - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            node.get_logger().info(
                f"Releasing launch wave {item.wave_number} at "
                f"t={time.monotonic() - start_time:.1f}s: {', '.join(item.robots)}"
            )
            for robot_name in item.robots:
                robots[robot_name].launch_event.set()

        for thread in worker_threads:
            thread.join()
    except KeyboardInterrupt:
        node.get_logger().warning("Keyboard interrupt: sending STOP to all robots")
        for robot in robots.values():
            node.send_stop(robot)
    finally:
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()

    failed = [robot.name for robot in robots.values() if robot.failed]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
