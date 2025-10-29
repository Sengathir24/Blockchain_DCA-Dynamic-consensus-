"""
metrics_calculator.py

Provides functions to compute common blockchain performance metrics:
- Latency (ms)
- Throughput (TPS)
- Scalability (nodes)
- Energy efficiency (Joules/Tx)

Includes a small CLI/demo that attempts to read
`blockchain_traffic_trafficsim.csv` from the repository root. If not
found, it runs a synthetic example.

Usage (from repository root):
    python metrics_calculator.py

The functions are pure and can be imported into other scripts/tests.
"""

import csv
import os
import statistics
from datetime import datetime
from typing import Iterable, List, Tuple, Optional, Dict


def _parse_time(value: str) -> float:
    """Parse a timestamp value into seconds since epoch (float).

    Accepts numeric strings (seconds), or ISO 8601 datetimes. Raises
    ValueError if parsing fails.
    """
    try:
        # try numeric (seconds)
        return float(value)
    except Exception:
        pass
    try:
        # try ISO format
        dt = datetime.fromisoformat(value)
        return dt.timestamp()
    except Exception as e:
        raise ValueError(f"Could not parse time value: {value}") from e


def calculate_latency_ms_from_latencies(latencies_ms: Iterable[float]) -> Dict[str, float]:
    """Calculate latency summary statistics from a list of latencies in milliseconds.

    Returns a dict with: count, mean, median, p95, p99, min, max.
    """
    lat_list = [float(x) for x in latencies_ms]
    if not lat_list:
        raise ValueError("latencies list is empty")
    lat_list.sort()
    n = len(lat_list)
    mean = statistics.mean(lat_list)
    median = statistics.median(lat_list)

    def percentile(sorted_list: List[float], perc: float) -> float:
        if not sorted_list:
            return 0.0
        k = (len(sorted_list) - 1) * (perc / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_list) - 1)
        if f == c:
            return sorted_list[int(k)]
        d0 = sorted_list[f] * (c - k)
        d1 = sorted_list[c] * (k - f)
        return d0 + d1

    p95 = percentile(lat_list, 95)
    p99 = percentile(lat_list, 99)

    return {
        "count": n,
        "mean_ms": mean,
        "median_ms": median,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": lat_list[0],
        "max_ms": lat_list[-1],
    }


def calculate_latency_ms_from_timestamps(pairs: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    """Calculate latency stats from iterable of (submit_time, confirm_time).

    Each time may be a numeric string (seconds) or ISO datetime string. The
    returned latencies are in milliseconds.
    """
    latencies_ms = []
    for submit, confirm in pairs:
        s = _parse_time(str(submit))
        c = _parse_time(str(confirm))
        if c < s:
            # skip or raise; we'll skip with a warning-like behavior by ignoring
            # negative latencies
            continue
        latencies_ms.append((c - s) * 1000.0)
    return calculate_latency_ms_from_latencies(latencies_ms)


def calculate_throughput_tps(num_transactions: int, duration_seconds: float) -> float:
    """Calculate throughput in transactions per second (TPS).

    Raises ValueError if duration_seconds <= 0.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    return float(num_transactions) / float(duration_seconds)


def calculate_scalability(nodes: int, throughput: float, baseline_nodes: int = 1, baseline_throughput: Optional[float] = None) -> Dict[str, float]:
    """Calculate simple scalability metrics.

    - tps_per_node: throughput / nodes
    - scaling_efficiency: if baseline_throughput provided, compares actual
      throughput to ideal linear scaling relative to the baseline.

    Returns a dict with nodes, throughput, tps_per_node and optional
    scaling_efficiency (0..inf, where 1.0 means perfect linear scaling).
    """
    if nodes <= 0:
        raise ValueError("nodes must be > 0")
    tps_per_node = throughput / nodes
    result = {
        "nodes": nodes,
        "throughput_tps": throughput,
        "tps_per_node": tps_per_node,
    }
    if baseline_throughput is not None:
        if baseline_nodes <= 0:
            raise ValueError("baseline_nodes must be > 0")
        ideal_throughput = baseline_throughput * (nodes / float(baseline_nodes))
        # scaling_efficiency = actual / ideal
        scaling_efficiency = throughput / ideal_throughput if ideal_throughput > 0 else 0.0
        result["scaling_efficiency"] = scaling_efficiency
    return result


def calculate_energy_efficiency_j_per_tx(total_energy_joules: float, num_transactions: int) -> float:
    """Return Joules per transaction. Raises on invalid inputs."""
    if num_transactions <= 0:
        raise ValueError("num_transactions must be > 0")
    return float(total_energy_joules) / float(num_transactions)


def read_csv_latencies(file_path: str, submit_col: str = "submit_time", confirm_col: str = "confirm_time", latency_col: Optional[str] = None) -> List[Tuple[str, str]]:
    """Read a CSV and return a list of (submit, confirm) pairs or a list of latency values.

    If `latency_col` is provided and present, this will return pairs where
    submit is an index and confirm is the latency value (seconds or ms).
    The caller should know how to interpret them. This helper is permissive
    and uses header lookup.
    """
    pairs = []
    with open(file_path, newline='') as fh:
        rdr = csv.DictReader(fh)
        headers = rdr.fieldnames or []
        for row in rdr:
            if latency_col and latency_col in headers:
                pairs.append(("0", row[latency_col]))
            elif submit_col in headers and confirm_col in headers:
                pairs.append((row[submit_col], row[confirm_col]))
            else:
                # try first two columns
                vals = list(row.values())
                if len(vals) >= 2:
                    pairs.append((vals[0], vals[1]))
    return pairs


# -----------------------
# Consensus-specific metrics
# -----------------------
CONSENSUS_PROFILES = {
    'PoW': {
        'throughput_factor': 0.6,    # relative capacity multiplier
        'base_block_time_ms': 10000, # typical block time (ms)
        'comm_factor': 0.01,         # communication overhead per extra node
        'energy_per_tx_j': 50.0,     # high energy cost per tx (J)
        'min_nodes': 1,
    },
    'PoS': {
        'throughput_factor': 0.9,
        'base_block_time_ms': 2000,
        'comm_factor': 0.005,
        'energy_per_tx_j': 5.0,
        'min_nodes': 1,
    },
    'Raft': {
        'throughput_factor': 0.8,
        'base_block_time_ms': 500,
        'comm_factor': 0.01,
        'energy_per_tx_j': 2.0,
        'min_nodes': 3,
    },
    'PBFT': {
        'throughput_factor': 0.7,
        'base_block_time_ms': 400,
        'comm_factor': 0.05,   # PBFT has higher communication complexity (O(n^2))
        'energy_per_tx_j': 3.0,
        'min_nodes': 4,
    },
    'HotStuff': {
        'throughput_factor': 0.85,
        'base_block_time_ms': 350,
        'comm_factor': 0.02,
        'energy_per_tx_j': 2.5,
        'min_nodes': 3,
    }
}


def metrics_for_consensus(mode: str, nodes: int, total_transactions: int, duration_seconds: float, base_tps_per_node: float = 5.0) -> Dict[str, float]:
    """Estimate Latency (ms), Throughput (TPS), Scalability (nodes) and Energy (J/tx)

    This is a simple model using configurable consensus profiles defined in
    CONSENSUS_PROFILES. The model is intentionally lightweight and deterministic
    so it can be used for comparative analysis across consensus algorithms.

    Inputs:
    - mode: one of the keys in CONSENSUS_PROFILES ('PoW','PoS','Raft','PBFT','HotStuff')
    - nodes: number of participating nodes
    - total_transactions: total transactions observed in the period
    - duration_seconds: measurement period in seconds
    - base_tps_per_node: baseline capacity per node (tps)

    Returns a dict with keys: latency_ms (median estimate), throughput_tps,
    scalability (tps_per_node), energy_per_tx_j, nodes
    """
    mode = str(mode)
    if mode not in CONSENSUS_PROFILES:
        raise ValueError(f"Unknown consensus mode: {mode}")
    if nodes <= 0:
        raise ValueError("nodes must be > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")

    profile = CONSENSUS_PROFILES[mode]

    # Capacity model: base capacity is base_tps_per_node * nodes scaled by throughput_factor
    capacity_tps = base_tps_per_node * nodes * profile['throughput_factor']

    # Observed offered load (tps) from input
    offered_tps = float(total_transactions) / float(duration_seconds)

    # Actual throughput is limited by capacity
    actual_tps = min(offered_tps, capacity_tps)

    # Scalability metric (simple): tps per node and scaling efficiency relative to 1 node ideal
    tps_per_node = actual_tps / nodes
    baseline_throughput = base_tps_per_node * profile['throughput_factor'] if nodes >= 1 else 0.0
    scaling_efficiency = (actual_tps / (baseline_throughput * nodes)) if baseline_throughput * nodes > 0 else 0.0

    # Latency model: base_block_time scaled by communication overhead and how loaded the system is
    load_factor = offered_tps / max(capacity_tps, 1e-9)
    # communication penalty grows with nodes and comm_factor; higher load increases latency
    comm_penalty = 1.0 + profile['comm_factor'] * max(0, nodes - 1)
    latency_ms = profile['base_block_time_ms'] * comm_penalty * (1.0 + 0.5 * max(0.0, load_factor - 1.0))

    # Energy per tx: base energy divided by efficiency of throughput (if system is saturated energy per tx rises)
    base_energy = profile['energy_per_tx_j']
    energy_per_tx = base_energy * (1.0 + 0.5 * max(0.0, load_factor - 1.0))

    return {
        'mode': mode,
        'nodes': nodes,
        'offered_tps': offered_tps,
        'capacity_tps': capacity_tps,
        'throughput_tps': actual_tps,
        'tps_per_node': tps_per_node,
        'scaling_efficiency': scaling_efficiency,
        'latency_ms': latency_ms,
        'energy_per_tx_j': energy_per_tx,
    }


def dynamic_select_consensus(nodes: int, total_transactions: int, duration_seconds: float, modes: Optional[List[str]] = None, weights: Optional[Dict[str, float]] = None) -> Dict[str, any]:
    """Select the best consensus mode from `modes` given weighted importance of metrics.

    The selector computes metrics_for_consensus for each mode and scores them.
    By default weights = {'latency':0.4, 'throughput':0.4, 'energy':0.2} where higher score is better.
    Lower latency and lower energy are better; higher throughput is better.
    """
    if modes is None:
        modes = list(CONSENSUS_PROFILES.keys())
    if weights is None:
        weights = {'latency': 0.4, 'throughput': 0.4, 'energy': 0.2}

    results = {}
    metrics_list = []
    for m in modes:
        try:
            met = metrics_for_consensus(m, nodes, total_transactions, duration_seconds)
            metrics_list.append(met)
        except Exception:
            # mode may be invalid for given nodes (e.g., PBFT needs >=4), mark as infeasible
            continue

    if not metrics_list:
        raise RuntimeError("No feasible consensus modes for given inputs")

    # Normalize metrics for scoring: build arrays
    latencies = [x['latency_ms'] for x in metrics_list]
    throughputs = [x['throughput_tps'] for x in metrics_list]
    energies = [x['energy_per_tx_j'] for x in metrics_list]

    # For latency and energy: lower is better -> invert normalization
    def normalize(values, invert=False):
        mn = min(values)
        mx = max(values)
        if mx == mn:
            return [1.0 for _ in values]
        out = []
        for v in values:
            n = (v - mn) / (mx - mn)
            if invert:
                n = 1.0 - n
            out.append(n)
        return out

    norm_lat = normalize(latencies, invert=True)
    norm_tps = normalize(throughputs, invert=False)
    norm_energy = normalize(energies, invert=True)

    best = None
    best_score = -1.0
    scored = []
    for idx, met in enumerate(metrics_list):
        score = (weights.get('latency', 0.0) * norm_lat[idx] +
                 weights.get('throughput', 0.0) * norm_tps[idx] +
                 weights.get('energy', 0.0) * norm_energy[idx])
        met_copy = met.copy()
        met_copy['score'] = score
        scored.append(met_copy)
        if score > best_score:
            best_score = score
            best = met_copy

    results['candidates'] = scored
    results['best'] = best
    results['weights'] = weights
    return results



if __name__ == "__main__":
    # Demo runner: attempt to read repository CSV; otherwise generate synthetic data.
    repo_root = os.path.dirname(__file__)
    csv_path = os.path.join(repo_root, "blockchain_traffic_trafficsim.csv")

    try:
        if os.path.exists(csv_path):
            print(f"Found CSV at {csv_path}, reading...")
            raw_pairs = read_csv_latencies(csv_path)
            # Try to interpret as timestamp pairs first
            try:
                latency_stats = calculate_latency_ms_from_timestamps(raw_pairs)
            except Exception:
                # fallback: treat second column as latency in ms
                lat_ms = [float(sec) for _, sec in raw_pairs]
                latency_stats = calculate_latency_ms_from_latencies(lat_ms)
            total_txs = latency_stats.get("count", 0)
            # approximate duration using min/max timestamps if available
            # Not all CSVs supply timestamps; if we don't have numeric times, fall back
            duration_seconds = None
            try:
                # attempt to compute duration from first/last rows
                with open(csv_path, newline='') as fh:
                    rdr = csv.reader(fh)
                    rows = list(rdr)
                if len(rows) > 1:
                    # look at first data row vs last data row
                    first = rows[1][0]
                    last = rows[-1][0]
                    duration_seconds = max(0.0, _parse_time(last) - _parse_time(first))
            except Exception:
                duration_seconds = None

            if duration_seconds and duration_seconds > 0:
                throughput = calculate_throughput_tps(total_txs, duration_seconds)
            else:
                throughput = None

            print("Latency stats:")
            for k, v in latency_stats.items():
                print(f"  {k}: {v}")
            if throughput is not None:
                print(f"Throughput (TPS): {throughput:.3f}")
            else:
                print("Throughput: could not compute (missing duration)")
            # Compute per-consensus metrics and dynamic selection
            try:
                nodes = 5
                total_txs = latency_stats.get("count", 0)
                duration = duration_seconds if duration_seconds and duration_seconds > 0 else 60.0
                print(f"\nEstimating per-consensus metrics for nodes={nodes}, duration={duration}s, total_txs={total_txs}")
                for mode in CONSENSUS_PROFILES.keys():
                    try:
                        m = metrics_for_consensus(mode, nodes, total_txs, duration)
                        print(f"\nMode: {mode}")
                        print(f"  Throughput (TPS): {m['throughput_tps']:.3f} (capacity: {m['capacity_tps']:.3f})")
                        print(f"  Latency (ms): {m['latency_ms']:.2f}")
                        print(f"  Energy per tx (J): {m['energy_per_tx_j']:.4f}")
                        print(f"  tps_per_node: {m['tps_per_node']:.4f}, scaling_efficiency: {m['scaling_efficiency']:.4f}")
                    except Exception as _:
                        print(f"  Mode {mode} not feasible for nodes={nodes}")

                selection = dynamic_select_consensus(nodes, total_txs, duration)
                best = selection.get('best')
                if best:
                    print(f"\nDynamic selection -> Best mode: {best['mode']} (score={best['score']:.4f})")
                else:
                    print("\nDynamic selection: no best mode found.")
            except Exception as e:
                print(f"Per-consensus estimation failed: {e}")
        else:
            print("CSV not found — running demo with synthetic data")
            # synthetic latencies in ms
            import random

            synth = [random.uniform(50, 500) for _ in range(1000)]
            latency_stats = calculate_latency_ms_from_latencies(synth)
            duration_seconds = 60.0  # assume 1 minute for demo
            throughput = calculate_throughput_tps(latency_stats["count"], duration_seconds)
            scalability = calculate_scalability(nodes=10, throughput=throughput, baseline_nodes=1, baseline_throughput=throughput/10.0)
            energy_j = 5000.0  # total joules for demo
            energy_per_tx = calculate_energy_efficiency_j_per_tx(energy_j, latency_stats["count"])

            print("Latency stats (synthetic):")
            for k, v in latency_stats.items():
                print(f"  {k}: {v}")
            print(f"Throughput (TPS): {throughput:.3f}")
            print(f"Scalability: nodes={scalability['nodes']}, tps_per_node={scalability['tps_per_node']:.4f}, scaling_efficiency={scalability.get('scaling_efficiency', 'N/A')}")
            print(f"Energy per tx (Joules/Tx): {energy_per_tx:.6f}")
            # Run per-consensus metrics on synthetic example and choose best
            try:
                nodes = 10
                total_txs = latency_stats.get("count", 0)
                duration = duration_seconds
                print(f"\nEstimating per-consensus metrics for nodes={nodes}, duration={duration}s, total_txs={total_txs}")
                for mode in CONSENSUS_PROFILES.keys():
                    m = metrics_for_consensus(mode, nodes, total_txs, duration)
                    print(f"\nMode: {mode}")
                    print(f"  Throughput (TPS): {m['throughput_tps']:.3f} (capacity: {m['capacity_tps']:.3f})")
                    print(f"  Latency (ms): {m['latency_ms']:.2f}")
                    print(f"  Energy per tx (J): {m['energy_per_tx_j']:.4f}")

                selection = dynamic_select_consensus(nodes, total_txs, duration)
                best = selection.get('best')
                if best:
                    print(f"\nDynamic selection -> Best mode: {best['mode']} (score={best['score']:.4f})")
                else:
                    print("\nDynamic selection: no best mode found.")
            except Exception as e:
                print(f"Per-consensus estimation failed: {e}")
    except Exception as e:
        print(f"Demo run failed: {e}")
