# PHASE 1: Design Specification for UWSN Priority-Driven Transport

Date: January 7, 2026  
Status: **Design Frozen**

## Overview

Phase 1 establishes a concrete, stable specification for the entire project. All design decisions documented here are **binding** and will guide Phase 2 (implementation), Phase 3 (simulation), and Phase 4 (experiments). Changes to this document require explicit re-review.

---

## 1. Network Model

### 1.1 Topology

- **Type:** 3D underwater network with distributed nodes and centralized sink(s).
- **Deployment:** Random or grid-based deployment in a bounding box of size **X=1000m × Y=1000m × Z=500m** (depth).
- **Nodes:** Total **N=50** sensor nodes, with at least 80% in operation at any given time (network redundancy).
- **Sink(s):** One primary **surface sink** at position (500, 500, 0m) and optional **relay nodes** at intermediate depths.
- **Topology Type:** Static (nodes do not move during simulation runs, simplifying first iteration). Dynamic topology may be added in later phases.

### 1.2 Node Characteristics

Each node has:
- **Position:** (x, y, z) in 3D space.
- **Initial Energy:** E₀ = **10,000 Joules** per node.
- **Transmission Range:** **300 meters** in water (acoustic modem assumption).
- **Modem Specs:**
  - Bit rate: **1 kbps** (conservative UWSN standard).
  - Transmission power: **2 Watts**.
  - Reception power: **0.5 Watts**.
  - Idle power: **0.1 Watts**.
- **Queue Capacity:** **100 packets** per node (small local buffer).
- **Neighbor Table:** Periodically updated via beacon broadcasts (every 10 simulation time units).

### 1.3 Propagation Model

- **Path Loss:** Urick acoustic propagation model or simplified log-distance (20 log₁₀(distance)).
- **SNR Threshold:** Packet is successfully received if SNR ≥ **10 dB**.
- **Propagation Delay:** d / 1500 seconds, where d is distance in meters and 1500 m/s is sound speed in seawater.
- **Multipath:** Ignored in Phase 1 (single-path assumption for simplicity).
- **Doppler Effect:** Ignored (nodes are static).

---

## 2. Traffic Model and Classes

### 2.1 Traffic Classes

All packets belong to one of **three classes:**

| Class | Type | Description | Target Delay | Target PDR | Priority Weight | Example |
|-------|------|-------------|--------------|------------|-----------------|----------|
| **A** | **Emergency/Critical** | Real-time alerts, anomaly detection, safety warnings | < 5s | ≥ 95% | 3.0 | Intruder alert, temp > threshold |
| **B** | **Control/Configuration** | Routing updates, node commands, network management | 5-15s | ≥ 90% | 2.0 | Route discovery, parameter sync |
| **C** | **Routine/Sensing** | Periodic sensor data, monitoring, non-urgent telemetry | 15-60s | ≥ 80% | 1.0 | Monthly water quality reading |

### 2.2 Traffic Generation Pattern

- **Class A:** Poisson arrival with λ_A = **0.1 packet/node/sec** (bursty, event-driven). Burst size: 1–3 packets.
- **Class B:** Poisson arrival with λ_B = **0.01 packet/node/sec** (regular control).
- **Class C:** Poisson arrival with λ_C = **0.02 packet/node/sec** (periodic sensing).
- **Packet Size:**
  - Class A: 128 bytes (header + alert payload).
  - Class B: 64 bytes (control commands).
  - Class C: 256 bytes (full sensor reading).
- **Simulation Duration:** **1000 time units** (simulation time, not real seconds).

---

## 3. Packet Format and Schema

Every packet carries the following fields:

```
Packet {
  pkt_id:           uint32          # Unique packet ID
  src_node_id:      uint16          # Source node ID (0–N-1)
  dst_node_id:      uint16          # Destination node ID (typically sink)
  timestamp:        float           # Time packet was created
  class_id:         uint8           # 0=C (routine), 1=B (control), 2=A (emergency)
  priority_level:   uint8           # 0–3; mapped from class_id and context
  ttl:              uint8           # Time-to-live (max hops)
  hop_count:        uint8           # Current hop count
  payload_size:     uint16          # Bytes of payload
  payload:          bytes[...]      # Actual data (sensor reading, alert, etc.)
  local_context:    struct          # Node-local metadata
    {
      depth:        float           # Node depth in meters
      residual_e:   float           # Node's residual energy (J)
      sensed_val:   float           # Raw sensor value (e.g., temp, pressure)
      is_anomaly:   bool            # True if value deviates from baseline
    }
}
```

### 3.1 Packet Size

Total bytes per packet:
- **Overhead (header):** 50 bytes (src, dst, class, priority, ttl, hop, timestamps, etc.).
- **Payload:** 64–256 bytes (depends on class).
- **Total:** 114–306 bytes per packet.

---

## 4. Node State and Queues

Each node maintains:

```
NodeState {
  node_id:          uint16
  position:         (x, y, z)
  residual_energy:  float          # Joules remaining
  neighbor_list:    list[uint16]   # IDs of reachable neighbors
  queue_A:          PriorityQueue  # Class A packets (highest priority)
  queue_B:          PriorityQueue  # Class B packets
  queue_C:          PriorityQueue  # Class C packets (lowest priority)
  stats:            struct
    {
      tx_count:     dict[class] → int   # Packets sent by class
      rx_count:     dict[class] → int   # Packets received by class
      drop_count:   dict[class] → int   # Packets dropped by class
      energy_used:  dict[class] → float # Energy per class
    }
}
```

### 4.1 Queue Discipline

- **Per-class queues:** Each node maintains 3 separate queues (A, B, C).
- **Service order:** **Strict Priority** with starvation avoidance:
  - Serve 1 Class A packet (if available).
  - Serve 1 Class B packet (if available and A queue was empty).
  - Serve 1 Class C packet (if available and both A and B queues were empty).
- **Max queue size:** 100 packets per node total; if exceeded, drop from class C first, then B, then A (if absolutely necessary).

---

## 5. Content-to-Priority Mapping

### 5.1 Mapping Rule

Each packet's **priority_level** (0–3) is determined by:

```
priority_level = base_priority + context_boost

where:
  base_priority = {
    2  if class_id == A
    1  if class_id == B
    0  if class_id == C
  }
  
  context_boost = {
    +1 if is_anomaly == True   (sensor value unusual)
    +1 if residual_energy < 20% (node near end-of-life, boost to preserve criticality)
    0  otherwise
  }
  
  priority_level = min(base_priority + context_boost, 3)
```

### 5.2 Examples

| Class | Anomaly | Low Energy | Computed Priority | Interpretation |
|-------|---------|------------|-------------------|----------------|
| C     | No      | No         | 0                 | Lowest priority, safe to drop |
| C     | Yes     | No         | 1                 | Routine data but unusual → boost |
| B     | No      | No         | 1                 | Control message, moderate priority |
| B     | No      | Yes        | 2                 | Control + node stressed → high |
| A     | No      | No         | 2                 | Emergency is naturally high |
| A     | Yes     | Yes        | 3                 | Emergency + anomaly + stress → max |

---

## 6. Energy Model

### 6.1 Energy Consumption Formula

For each packet transmission:

```
Energy_tx = (Payload_size + Overhead) * (Tx_power / Bit_rate) + Tx_power * Propagation_delay

where:
  Payload_size = 64–256 bytes (depends on class)
  Overhead = 50 bytes
  Tx_power = 2 W
  Bit_rate = 1 kbps
  Propagation_delay = distance / 1500 (seconds)
```

For example, transmitting a 128-byte Class A packet over 200m:
```
Energy_tx = (128 + 50) * 8 bits / 1000 kbps * 2W + 2W * (200/1500)s
          ≈ 1.184 J + 0.267 J ≈ 1.45 J
```

Reception: **Energy_rx = Payload_size * 0.5W / Bit_rate** ≈ 0.064–0.256 J per packet.

Idle (per second): **0.1 W × 1s = 0.1 J**.

### 6.2 Simulation Energy Accounting

- Track energy consumed per class (A, B, C) separately for later QoS analysis.
- Node "death" occurs when residual_energy < 0; it cannot transmit or forward afterward.

---

## 7. Metrics and Key Performance Indicators (KPIs)

All metrics will be collected per-packet and aggregated by class and network-wide.

### 7.1 Per-Packet Metrics

| Metric | Definition | Unit | Importance |
|--------|-----------|------|------------|
| **End-to-End Delay** | Time from generation to delivery at sink | seconds | Critical |
| **Packet Delivery Ratio (PDR)** | (Packets delivered) / (Packets generated) | % | Critical |
| **Hop Count** | Number of hops taken | count | Medium |
| **Energy per Bit** | Total energy used / Payload bits | mJ/bit | Critical |

### 7.2 Network-Level Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| **Network Lifetime** | Time until 50% of nodes die (or first node dies) | seconds |
| **Throughput** | Bits delivered to sink / Simulation time | kbps |
| **Priority Satisfaction Rate** | (Delivered Class A) / (Generated Class A), etc. | % per class |
| **Average Delay per Class** | Mean delay for packets of each class | seconds |
| **Energy Efficiency** | Total bits delivered / Total energy consumed | bits/J |

---

## 8. Simulation Time and Conventions

- **Time Unit:** Simulation time (unitless); 1 time unit ≈ 1 real second (but can be scaled).
- **Random Seed:** Use fixed seeds for reproducibility in initial runs; later randomize for robustness.
- **Simulation Duration:** **1000 time units** per scenario.
- **Warm-up Period:** First 50 time units are discarded (transient phase); metrics collected from t=50 onward.

---

## 9. Phase 1 Checklist

Use this checklist to verify all aspects of Phase 1 are complete before moving to Phase 2.

- [ ] Network model finalized (topology, node count, dimensions).
- [ ] Traffic classes defined with QoS targets (Table in §2.1).
- [ ] Packet schema specified (§3).
- [ ] Priority mapping rule documented and tested on paper (§5).
- [ ] Energy model derived with example calculation (§6).
- [ ] KPIs and metrics listed (§7).
- [ ] Simulation time conventions set (§8).
- [ ] README updated with link to this document.
- [ ] GitHub issues created for Phase 2 tasks (data generator, core logic).

---

## 10. Design Review Notes

**Assumptions & Rationale:**

1. **Static topology:** Simplifies first iteration; mobility can be added as a refinement.
2. **50 nodes, 300m range:** Realistic for a moderate-scale UWSN (e.g., monitoring a bay or harbor).
3. **3-class model:** Balances realism (many real systems distinguish 3–5 priority levels) with simplicity.
4. **Strict priority + starvation control:** Common in real QoS systems; fair and understood.
5. **Content-aware boost via anomaly:** Encourages the core idea of the project (content matters, not just class).

**Known Limitations (for future refinement):**

- No MAC layer contention (ALOHA/CSMA); assumes nodes can transmit whenever.
- No retransmissions on failed delivery (single-shot delivery).
- Fixed link quality; no fading or shadowing.
- No multi-sink or load-balancing strategies.

These can be relaxed in Phase 3 for more realism if time permits.

---

## 11. Document Maintenance

- **Last Updated:** January 7, 2026.
- **Next Review:** After Phase 2 code is written (to ensure specification is implementable).
- **Change Log:** (None yet).
