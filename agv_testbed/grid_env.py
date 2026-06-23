"""
Warehouse Grid Environment
--------------------------
8x8 transit grid = 64 nodes (IDs 0–63), all freely navigable.
Node (r, c) has integer id  r * COLS + c.
Depot is always node 0 = (row=0, col=0) = bottom-left corner.

Spur geometry — matches the physical Alvik test grid
------------------------------------------------------
Each workstation sits at the midpoint of an EDGE between two adjacent
transit nodes (its `between_nodes` pair, e.g. (8, 9)), not anchored to a
single node. The spur runs south from that midpoint into the workstation.
Each workstation gets two virtual node IDs outside 0–63:

  Spur entry  : ID = N_NODES + i          (i = workstation index)
  Workstation : ID = N_NODES + n_ws + i   (i = workstation index)

The spur entry connects to BOTH endpoints of its between_nodes edge (a
robot can approach from either side), and to its workstation (cost
SPUR_LEN). Direction (south) only matters to the physical command bridge
(path_to_commands.py), not to the grid/solver graph itself.

This separation guarantees:
  - Transit graph (0–63) is always clean — robots route freely.
  - No ID collision regardless of workstation placement.
  - Distance to a workstation is the cheaper of its two edge endpoints.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
from collections import deque
from pathlib import Path

DEPOT_NODE = 0          # (row=0, col=0) — bottom-left
SPUR_LEN   = 5          # half a cell — fixed spur length (transit→entry = 1, entry→ws = SPUR_LEN)


# ---------------------------------------------------------------------------
# Grid topology — parameterized so grid size isn't hardcoded module state.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridTopology:
    """4-connected rows x cols transit grid. node_id = row*cols + col."""
    rows: int
    cols: int

    @property
    def n_nodes(self) -> int:
        return self.rows * self.cols

    def node_id(self, r: int, c: int) -> int:
        return r * self.cols + c

    def node_rc(self, nid: int) -> tuple[int, int]:
        return nid // self.cols, nid % self.cols

    def neighbors(self, nid: int) -> list[int]:
        """4-connected transit neighbours of a transit node."""
        r, c = self.node_rc(nid)
        result = []
        if r > 0:             result.append(self.node_id(r - 1, c))
        if r < self.rows - 1: result.append(self.node_id(r + 1, c))
        if c > 0:             result.append(self.node_id(r, c - 1))
        if c < self.cols - 1: result.append(self.node_id(r, c + 1))
        return result

    def bfs_distances(self, source: int) -> np.ndarray:
        """BFS shortest-hop distances from source to all transit nodes."""
        dist = np.full(self.n_nodes, -1, dtype=np.int32)
        dist[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def bfs_path(self, source: int, target: int) -> list[int]:
        """Shortest transit-grid path from source to target (node IDs)."""
        if source == target:
            return [source]
        parent = {source: None}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if v not in parent:
                    parent[v] = u
                    if v == target:
                        path, cur = [], v
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        return path[::-1]
                    q.append(v)
        return []

    def build_distance_matrix(self) -> np.ndarray:
        """(n_nodes, n_nodes) BFS shortest-path distance matrix for transit nodes."""
        D = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int32)
        for src in range(self.n_nodes):
            D[src] = self.bfs_distances(src)
        return D


# Default topology + thin module-level wrappers, preserving every existing
# `from grid_env import neighbors, N_NODES, ROWS, COLS` import used by
# mapf_solver.py, instance_builder.py, AGV-Line-Following-Factory/*.py.
# UI-built grids of other sizes use GridTopology(rows, cols) directly instead.
DEFAULT_TOPOLOGY = GridTopology(rows=8, cols=8)
ROWS    = DEFAULT_TOPOLOGY.rows
COLS    = DEFAULT_TOPOLOGY.cols
N_NODES = DEFAULT_TOPOLOGY.n_nodes


def node_id(r: int, c: int) -> int:
    return DEFAULT_TOPOLOGY.node_id(r, c)

def node_rc(nid: int):
    """Row, col for a transit node on the default 8x8 grid."""
    return DEFAULT_TOPOLOGY.node_rc(nid)

def neighbors(nid: int) -> list[int]:
    """4-connected transit neighbours of a transit node on the default 8x8 grid."""
    return DEFAULT_TOPOLOGY.neighbors(nid)


def bfs_distances(source: int) -> np.ndarray:
    """BFS shortest-hop distances from source, on the default 8x8 grid."""
    return DEFAULT_TOPOLOGY.bfs_distances(source)

def bfs_path(source: int, target: int) -> list[int]:
    """Shortest transit-grid path from source to target, on the default 8x8 grid."""
    return DEFAULT_TOPOLOGY.bfs_path(source, target)

def build_distance_matrix() -> np.ndarray:
    """Distance matrix for the default 8x8 grid."""
    return DEFAULT_TOPOLOGY.build_distance_matrix()


# ---------------------------------------------------------------------------
# Grid environment
# ---------------------------------------------------------------------------

class WarehouseGrid:
    """
    Warehouse with edge-midpoint spur geometry (matches the physical Alvik grid).

    Virtual node ID scheme (given n workstations):
      Transit nodes  :   0 .. N_NODES-1          (the 8×8 grid)
      Spur entries   :   N_NODES .. N_NODES+n-1
      Workstations   :   N_NODES+n .. N_NODES+2n-1

    Attributes
    ----------
    dist            : (64,64) BFS transit distances
    depot           : int — depot transit node (always 0)
    between_nodes   : list[tuple[int,int]] — the two transit nodes each
                      workstation's spur entry sits between (its edge)
    spur_entry_ids  : list[int] — virtual spur entry node IDs
    workstation_ids : list[int] — virtual workstation node IDs
    processing_times: list[float]
    """

    def __init__(
        self,
        between_nodes: list[tuple[int, int]],
        processing_times: list[float],
        dist: np.ndarray | None = None,
        topology: GridTopology | None = None,
    ):
        self.topology = topology if topology is not None else DEFAULT_TOPOLOGY

        assert len(between_nodes) == len(processing_times)
        for a, b in between_nodes:
            assert DEPOT_NODE not in (a, b), "depot cannot host a workstation"
            assert b in self.topology.neighbors(a), (
                f"({a},{b}) is not a transit-grid edge on a "
                f"{self.topology.rows}x{self.topology.cols} grid"
            )

        n = len(between_nodes)
        self.depot            = DEPOT_NODE
        self.between_nodes    = list(between_nodes)
        self.processing_times = list(processing_times)
        self.dist             = dist if dist is not None else self.topology.build_distance_matrix()

        # Virtual IDs
        n_nodes = self.topology.n_nodes
        self.spur_entry_ids  = [n_nodes + i       for i in range(n)]
        self.workstation_ids = [n_nodes + n + i   for i in range(n)]

        # Convenience: sets for fast membership tests
        self._spur_entry_set  = set(self.spur_entry_ids)
        self._workstation_set = set(self.workstation_ids)

        # Backward-compat alias so existing code using .workstations still works
        self.workstations  = self.workstation_ids
        self.spur_entries  = self.spur_entry_ids

    # ------------------------------------------------------------------
    # Spur adjacency dict for MAPF A*
    # ------------------------------------------------------------------

    def spur_adjacency(self) -> dict[int, list[int]]:
        """
        Adjacency dict for all spur edges (bidirectional):
          transit_node_a <-> spur_entry   (cost 1 hop, either edge endpoint)
          transit_node_b <-> spur_entry
          spur_entry      <-> workstation  (cost SPUR_LEN hops, modelled as
                                            1 step here; the distance matrix
                                            handles the cost for VRP)
        Returns {node_id: [neighbour_id, ...]} for virtual nodes only.
        Transit↔transit edges are handled by neighbors() separately.
        """
        adj: dict[int, list[int]] = {}
        for entry_id, ws_id, (a, b) in zip(
            self.spur_entry_ids, self.workstation_ids, self.between_nodes
        ):
            # both edge endpoints <-> spur entry
            adj.setdefault(a, []).append(entry_id)
            adj.setdefault(entry_id, []).append(a)
            adj.setdefault(b, []).append(entry_id)
            adj.setdefault(entry_id, []).append(b)
            # spur entry <-> workstation
            adj.setdefault(entry_id, []).append(ws_id)
            adj.setdefault(ws_id,    []).append(entry_id)
        return adj

    def is_workstation(self, node_id: int) -> bool:
        return node_id in self._workstation_set

    def is_spur_entry(self, node_id: int) -> bool:
        return node_id in self._spur_entry_set

    # ------------------------------------------------------------------
    # Distance matrix for the VRP-RPD solver
    # rows/cols: index 0 = depot, indices 1..n = workstations
    # ------------------------------------------------------------------

    def solver_distance_matrix(self) -> np.ndarray:
        """
        (n+1)×(n+1) distance matrix.  Index 0 = depot, 1..n = workstations.

        All transit distances are BFS hops.  Visiting a workstation adds:
          +1       : transit → spur_entry (from whichever edge endpoint is nearer)
          +SPUR_LEN: spur_entry → workstation center
        So each workstation visit adds (1 + SPUR_LEN) each way, computed
        from the cheaper of the workstation's two edge endpoints.
        """
        n = len(self.between_nodes)
        size = n + 1
        D = np.zeros((size, size), dtype=np.float64)
        spur_cost = 1 + SPUR_LEN   # transit→entry + entry→ws

        def anchor_dist(u: int, idx: int) -> float:
            """Distance from transit node u to workstation idx's nearer edge endpoint."""
            a, b = self.between_nodes[idx]
            return min(float(self.dist[u, a]), float(self.dist[u, b]))

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                if i == 0 and j == 0:
                    grid_dist = 0.0
                elif i == 0:
                    grid_dist = anchor_dist(self.depot, j - 1)
                elif j == 0:
                    grid_dist = anchor_dist(self.depot, i - 1)
                else:
                    a_i, b_i = self.between_nodes[i - 1]
                    grid_dist = min(anchor_dist(a_i, j - 1), anchor_dist(b_i, j - 1))
                cost_i = 0 if i == 0 else spur_cost
                cost_j = 0 if j == 0 else spur_cost
                D[i, j] = grid_dist + cost_i + cost_j
        return D

    def __repr__(self):
        return (
            f"WarehouseGrid(depot={self.depot}, "
            f"n_workstations={len(self.between_nodes)}, "
            f"grid={self.topology.rows}x{self.topology.cols})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Factory: build grid from the physical workstations.json layout
# ---------------------------------------------------------------------------

WORKSTATIONS_JSON = Path(__file__).resolve().parent.parent / "AGV-Line-Following-Factory" / "workstations.json"


def load_physical_grid(
    workstations_path: str | Path = WORKSTATIONS_JSON,
    proc_times: list[float] | None = None,
    rng: np.random.Generator | None = None,
    proc_time_range: tuple[float, float] = (5.0, 20.0),
) -> WarehouseGrid:
    """
    Build a WarehouseGrid from the physical AGV-Line-Following-Factory
    workstations.json layout (real, fixed positions — not randomly placed).

    workstations.json's between_nodes are 1-indexed (node 1 = grid (0,0));
    this module's node IDs are 0-indexed, so 1 is subtracted on load.

    proc_times: optional explicit list (must match the JSON's workstation
    count and order). If omitted, random processing times are drawn from
    proc_time_range using `rng` (or a fresh default RNG).
    """
    data = json.loads(Path(workstations_path).read_text())
    ws_list = data["workstations"]
    between_nodes = [
        (ws["between_nodes"][0] - 1, ws["between_nodes"][1] - 1)
        for ws in ws_list
    ]

    if proc_times is None:
        rng = rng if rng is not None else np.random.default_rng()
        lo, hi = proc_time_range
        proc_times = rng.uniform(lo, hi, size=len(ws_list)).tolist()

    assert len(proc_times) == len(between_nodes), (
        f"proc_times length ({len(proc_times)}) must match "
        f"workstation count ({len(between_nodes)})"
    )

    return WarehouseGrid(
        between_nodes=between_nodes,
        processing_times=proc_times,
        dist=build_distance_matrix(),
    )


# ---------------------------------------------------------------------------
# Factory: build a grid from UI input (arbitrary grid size + clicked
# workstation edges) — used by the webapp, not tied to workstations.json.
# ---------------------------------------------------------------------------

def build_grid_from_ui(
    rows: int,
    cols: int,
    workstation_edges: list[tuple[int, int]],
    processing_times: list[float] | None = None,
    rng: np.random.Generator | None = None,
    proc_time_range: tuple[float, float] = (5.0, 20.0),
) -> WarehouseGrid:
    """
    Build a WarehouseGrid for an arbitrary rows x cols grid with workstations
    placed on caller-supplied edges (0-indexed node id pairs, e.g. from a
    UI where the user clicks two adjacent cells to place a workstation).

    Raises ValueError (not a bare AssertionError) on invalid input, with a
    message suitable for showing directly to a non-coder user — e.g. a
    workstation edge that isn't grid-adjacent, or that includes the depot.
    """
    if rows < 2 or cols < 2:
        raise ValueError(f"Grid must be at least 2x2, got {rows}x{cols}")
    if not workstation_edges:
        raise ValueError("At least one workstation is required")

    topology = GridTopology(rows=rows, cols=cols)
    n_nodes = topology.n_nodes

    seen_edges = set()
    for a, b in workstation_edges:
        if not (0 <= a < n_nodes and 0 <= b < n_nodes):
            raise ValueError(
                f"Workstation edge ({a},{b}) has a node outside the "
                f"{rows}x{cols} grid (valid range: 0-{n_nodes - 1})"
            )
        if DEPOT_NODE in (a, b):
            raise ValueError(f"Workstation edge ({a},{b}) cannot include the depot (node {DEPOT_NODE})")
        if b not in topology.neighbors(a):
            ra, ca = topology.node_rc(a)
            rb, cb = topology.node_rc(b)
            raise ValueError(
                f"Workstation edge ({a},{b}) -- (row,col) ({ra},{ca}) and "
                f"({rb},{cb}) -- is not adjacent on a {rows}x{cols} grid"
            )
        ra, ca = topology.node_rc(a)
        rb, cb = topology.node_rc(b)
        if ra != rb:
            # The spur model (path_to_commands.py) assumes every workstation
            # edge runs east-west, with the spur always heading south from
            # the midpoint -- a north-south edge has no valid spur direction.
            raise ValueError(
                f"Workstation edge ({a},{b}) -- (row,col) ({ra},{ca}) and "
                f"({rb},{cb}) -- runs north-south, but workstations must be "
                f"on an east-west edge (same row) so the spur can run south"
            )
        edge_key = tuple(sorted((a, b)))
        if edge_key in seen_edges:
            raise ValueError(f"Workstation edge ({a},{b}) is listed more than once")
        seen_edges.add(edge_key)

    if processing_times is None:
        rng = rng if rng is not None else np.random.default_rng()
        lo, hi = proc_time_range
        processing_times = rng.uniform(lo, hi, size=len(workstation_edges)).tolist()
    elif len(processing_times) != len(workstation_edges):
        raise ValueError(
            f"processing_times length ({len(processing_times)}) must match "
            f"workstation count ({len(workstation_edges)})"
        )

    return WarehouseGrid(
        between_nodes=workstation_edges,
        processing_times=processing_times,
        dist=topology.build_distance_matrix(),
        topology=topology,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = load_physical_grid(rng=np.random.default_rng(42))
    print(grid)
    n = len(grid.between_nodes)
    print(f"Between-nodes (first 5)   : {grid.between_nodes[:5]}")
    print(f"Spur entry IDs (first 5)  : {grid.spur_entry_ids[:5]}")
    print(f"Workstation IDs (first 5) : {grid.workstation_ids[:5]}")
    print(f"Processing times (first 5): {[round(t, 1) for t in grid.processing_times[:5]]}")
    D = grid.solver_distance_matrix()
    print(f"Solver dist matrix shape  : {D.shape}")
    print(f"depot->ws[0]: {D[0,1]:.0f}")
    print(f"ws[0]->ws[1]: {D[1,2]:.0f}")
    adj = grid.spur_adjacency()
    print(f"Spur adj sample: entry {grid.spur_entry_ids[0]} -> {adj[grid.spur_entry_ids[0]]}")
