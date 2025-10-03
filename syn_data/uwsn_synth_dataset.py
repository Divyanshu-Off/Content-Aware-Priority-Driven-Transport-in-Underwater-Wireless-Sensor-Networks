"""
uwsn_synth_dataset.py
Synthetic dataset generator for UWSN packet-level priority data.

Outputs:
 - dataset_ml.csv      : feature rows for ML training (one row per packet)
 - traffic_for_sim.csv : simulator-friendly traffic file (includes priority)
"""

import os
import math
import uuid
import csv
import argparse
import random
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime

# ------------------------
# Parameters (tune here)
# ------------------------
DEFAULTS = {
    "num_nodes": 20,
    "duration": 3600.0,         # seconds
    "mean_interval": 10.0,      # seconds between periodic readings
    "telemetry_noise": 0.05,    # std dev of normal noise
    "high_event_rate": 1/600.0, # per node (1 per 600s)
    "critical_event_rate": 1/3600.0, # per node (1 per hour)
    "bulk_event_rate": 1/7200.0,
    "high_mag": 2.0,            # multiplier for reading spike
    "critical_mag": 6.0,
    "window_size": 5,           # for moving statistics
    "initial_battery": 1.0,
    "battery_drain_per_pkt": 0.0002, # fraction
    "seed": 42,
    "output_dir": "synth_output",
}

# ------------------------
# Utility functions
# ------------------------
def exponential_jitter(rate):
    """Return a positive jitter sample (exponential) with mean = 1/rate."""
    return np.random.exponential(scale=1.0/rate) if rate > 0 else float('inf')

# ------------------------
# Core generator
# ------------------------
def generate_traffic(params):
    np.random.seed(params["seed"])
    random.seed(params["seed"])

    num_nodes = params["num_nodes"]
    T = params["duration"]
    mean_interval = params["mean_interval"]

    rows = []
    pkt_id = 0

    # For each node maintain a small ring buffer to compute moving stats
    node_windows = {n: deque(maxlen=params["window_size"]) for n in range(num_nodes)}
    node_battery = {n: params["initial_battery"] for n in range(num_nodes)}
    node_depth = {n: random.uniform(5, 100) for n in range(num_nodes)}  # depth meters

    # Pre-generate periodic sample times per node (with jitter)
    for node in range(num_nodes):
        t = 0.0
        while t < T:
            # jittered interval
            interval = max(0.1, np.random.normal(loc=mean_interval, scale=mean_interval*0.1))
            t += interval
            # base reading (slow sinusoidal + small noise)
            base = 10.0 * math.sin(2*math.pi*(t/3600.0) + node/5.0) + 20.0
            reading = base + np.random.normal(0, params["telemetry_noise"])
            # default label/priority
            label = "normal"
            pkt_size = 50  # bytes typical telemetry

            # Add to window for stats
            node_windows[node].append(reading)
            moving_mean = np.mean(node_windows[node]) if node_windows[node] else reading
            moving_std = np.std(node_windows[node]) if node_windows[node] else 0.0
            reading_delta = reading - moving_mean

            # append packet
            rows.append({
                "packet_id": pkt_id,
                "node_id": node,
                "timestamp": round(t, 3),
                "reading": round(reading, 6),
                "reading_delta": round(reading_delta, 6),
                "moving_std": round(moving_std, 6),
                "battery": round(node_battery[node], 6),
                "depth": round(node_depth[node], 2),
                "pkt_size": pkt_size,
                "label": label,
                "priority": 2  # normal -> priority 2
            })
            pkt_id += 1
            # drain battery a bit
            node_battery[node] = max(0.0, node_battery[node] - params["battery_drain_per_pkt"])

    # Now inject events (high, critical, bulk)
    def inject_events(rate_per_node, mag, label_name, priority_int, pkt_size=120, burst=1):
        nonlocal pkt_id
        if rate_per_node <= 0:
            return
        for node in range(num_nodes):
            # Poisson process: number of events ~ Poisson(rate * T)
            lam = rate_per_node * T
            k = np.random.poisson(lam)
            # generate k event times uniformly in [0,T]
            event_times = np.random.uniform(0, T, size=k)
            for et in event_times:
                # create burst around et
                for b in range(burst):
                    t = round(et + np.random.normal(0, 0.5), 3)
                    base = 10.0 * math.sin(2*math.pi*(t/3600.0) + node/5.0) + 20.0
                    reading = base + mag * np.random.rand() + np.random.normal(0, params["telemetry_noise"])
                    # update windows to reflect event if the event occurs before or after prior packets
                    node_windows[node].append(reading)
                    moving_mean = np.mean(node_windows[node]) if node_windows[node] else reading
                    moving_std = np.std(node_windows[node]) if node_windows[node] else 0.0
                    reading_delta = reading - moving_mean
                    rows.append({
                        "packet_id": pkt_id,
                        "node_id": node,
                        "timestamp": t,
                        "reading": round(reading, 6),
                        "reading_delta": round(reading_delta, 6),
                        "moving_std": round(moving_std, 6),
                        "battery": round(node_battery[node], 6),
                        "depth": round(node_depth[node], 2),
                        "pkt_size": pkt_size,
                        "label": label_name,
                        "priority": priority_int
                    })
                    pkt_id += 1
                    node_battery[node] = max(0.0, node_battery[node] - params["battery_drain_per_pkt"]*2)

    inject_events(params["high_event_rate"], params["high_mag"], "high", 1, pkt_size=80, burst=1)
    inject_events(params["critical_event_rate"], params["critical_mag"], "critical", 0, pkt_size=80, burst=2)
    inject_events(params["bulk_event_rate"], params["high_mag"]*0.5, "bulk", 3, pkt_size=1500, burst=1)

    # sort rows by timestamp
    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
    return rows_sorted

# ------------------------
# Export helpers
# ------------------------
def export_csv(rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    ml_columns = ["packet_id","node_id","timestamp","reading","reading_delta","moving_std","battery","depth","pkt_size","label","priority"]
    ml_path = os.path.join(outdir, "dataset_ml.csv")
    sim_path = os.path.join(outdir, "traffic_for_sim.csv")
    df = pd.DataFrame(rows)
    df.to_csv(ml_path, index=False, columns=ml_columns)
    df.to_csv(sim_path, index=False, columns=ml_columns)
    print(f"Exported {len(df)} rows to:")
    print(" -", ml_path)
    print(" -", sim_path)
    return ml_path, sim_path

# ------------------------
# Command line
# ------------------------
def main(args):
    params = DEFAULTS.copy()
    params.update(vars(args))
    print("Using params:", {k: params[k] for k in ["num_nodes","duration","mean_interval","high_event_rate","critical_event_rate","seed","output_dir"]})
    rows = generate_traffic(params)
    ml_path, sim_path = export_csv(rows, params["output_dir"])
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=DEFAULTS["num_nodes"])
    parser.add_argument("--duration", type=float, default=DEFAULTS["duration"])
    parser.add_argument("--mean_interval", type=float, default=DEFAULTS["mean_interval"])
    parser.add_argument("--high_event_rate", type=float, default=DEFAULTS["high_event_rate"])
    parser.add_argument("--critical_event_rate", type=float, default=DEFAULTS["critical_event_rate"])
    parser.add_argument("--bulk_event_rate", type=float, default=DEFAULTS["bulk_event_rate"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"])
    args = parser.parse_args()
    main(args)
