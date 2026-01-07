# PHASE 2: Implementation – Synthetic Data & Core Protocol Logic

Date: January 7, 2026  
Status: **In Progress**

## Overview

Phase 2 focuses on implementing the two core pillars of the system:
1. **Synthetic Traffic Generator** – produces realistic UWSN packets matching Phase 1 spec
2. **Priority-Driven Transport Protocol** – core scheduling and classification logic

Both components are standalone and testable, designed to feed into Phase 3 (simulation). By end of Phase 2, you have working Python modules ready for integration into a simulator.

---

## Task 1: Synthetic Dataset Generator

**Goal:** Generate synthetic UWSN traffic packets that match the Phase 1 specification exactly.

**Location:** `syn_data/` folder

### 1.1 Data Structure

Generate a CSV file with columns:

```
packet_id, timestamp, src_node_id, dst_node_id, class_id, payload_size, is_anomaly, depth, residual_energy
```

**Example rows:**
```
1, 0.5, 3, 50, 2, 128, 1, 250.0, 9800.0
2, 1.2, 7, 50, 0, 256, 0, 180.5, 9950.0
3, 1.5, 15, 50, 1, 64, 0, 320.0, 8500.0
```

### 1.2 Implementation Steps

1. **Create `syn_data/packet_generator.py`**
   - Class `TrafficGenerator` with methods:
     - `generate_traffic(num_nodes, num_packets, output_file)` → generates CSV
     - Uses Poisson arrivals: λ_A = 0.1, λ_B = 0.01, λ_C = 0.02 packets/node/sec
     - Assigns class uniformly based on arrival rates
     - Random sensor values ~N(20, 2) for realistic data
     - Anomaly detection: if sensed_value > mean + 2*std, set is_anomaly=1
     - Random depth from 0–500m
     - Random residual energy from 5,000–10,000 J

2. **Create `syn_data/dataset.csv`** (generated output)
   - At least 5000 packets for realistic scenario
   - Covers full 1000 time units as per Phase 1
   - Mix of all three classes (roughly 10% Class A, 5% B, 20% C; rest from Poisson)

3. **Test**
   - Verify class distribution matches expectations
   - Check anomaly detection logic
   - Validate CSV format

### 1.3 Acceptance Criteria

- [ ] `packet_generator.py` can generate valid CSV with ≥5000 packets
- [ ] Class distribution: ~10% A, ~5% B, ~20% C
- [ ] Anomaly flag set correctly (>mean + 2σ)
- [ ] All fields match Phase 1 packet schema
- [ ] Output CSV can be read as pandas DataFrame

---

## Task 2: Priority-Driven Transport Protocol (Core Logic)

**Goal:** Implement the content-aware priority scheduling and classification rules from Phase 1.

**Location:** `src/` folder (create `src/protocol/` if needed)

### 2.1 Core Components

#### 2.1.1 `Packet` Class

```python
class Packet:
    def __init__(self, pkt_id, src, dst, class_id, payload_size, timestamp,
                 depth, residual_energy, sensed_value, is_anomaly):
        self.pkt_id = pkt_id
        self.src_node_id = src
        self.dst_node_id = dst
        self.class_id = class_id  # 0=C, 1=B, 2=A
        self.payload_size = payload_size
        self.timestamp = timestamp
        self.hop_count = 0
        self.ttl = 10  # max hops before dropping
        
        # Local context
        self.depth = depth
        self.residual_energy = residual_energy
        self.sensed_value = sensed_value
        self.is_anomaly = is_anomaly
        
        # Computed fields
        self.priority_level = self.compute_priority()
    
    def compute_priority(self) -> int:
        """Implement content-to-priority mapping from Phase 1 §5"""
        base = [0, 1, 2][self.class_id]  # C→0, B→1, A→2
        boost = 0
        if self.is_anomaly:
            boost += 1
        if self.residual_energy < 0.2 * 10000:  # < 20%
            boost += 1
        return min(base + boost, 3)
```

#### 2.1.2 `Node` Class (for simulation prep)

```python
class Node:
    def __init__(self, node_id, x, y, z):
        self.node_id = node_id
        self.position = (x, y, z)
        self.residual_energy = 10000.0  # Joules
        
        # Queues per Phase 1 §4
        self.queue_A = PriorityQueue()  # Class A (highest)
        self.queue_B = PriorityQueue()  # Class B
        self.queue_C = PriorityQueue()  # Class C (lowest)
        
        # Stats
        self.stats = {'tx': {0:0, 1:0, 2:0}, 'rx': {0:0, 1:0, 2:0}, 
                      'drop': {0:0, 1:0, 2:0}, 'energy': {0:0, 1:0, 2:0}}
    
    def enqueue_packet(self, pkt: Packet) -> bool:
        """Add packet to appropriate queue. Return False if dropped."""
        queue = [self.queue_C, self.queue_B, self.queue_A][pkt.class_id]
        if queue.qsize() < 100:
            queue.put((-pkt.priority_level, pkt.pkt_id, pkt))  # negative for max-heap
            return True
        else:
            # Drop policy from Phase 1 §4.1
            self.stats['drop'][pkt.class_id] += 1
            return False
    
    def dequeue_next_packet(self) -> Packet | None:
        """Strict priority: A > B > C"""
        for queue in [self.queue_A, self.queue_B, self.queue_C]:
            if not queue.empty():
                _, _, pkt = queue.get()
                return pkt
        return None
    
    def consume_energy(self, pkt: Packet, distance: float):
        """Deduct energy for transmission per Phase 1 §6"""
        overhead = 50  # bytes
        total_bytes = overhead + pkt.payload_size
        tx_power = 2.0  # Watts
        bit_rate = 1000.0  # bps
        prop_delay = distance / 1500.0  # seconds
        
        energy_tx = (total_bytes * 8 / bit_rate) * tx_power + tx_power * prop_delay
        self.residual_energy -= energy_tx
        self.stats['energy'][pkt.class_id] += energy_tx
        return energy_tx
```

#### 2.1.3 `Classifier` Class

```python
class PriorityClassifier:
    @staticmethod
    def classify_packet(pkt: Packet) -> int:
        """Re-compute or update priority level dynamically."""
        return pkt.compute_priority()
```

### 2.2 Implementation Steps

1. **Create `src/protocol/packet.py`**
   - Implement `Packet` class with priority computation
   - Unit tests to verify priority mapping (all 6 cases from Phase 1 Table §5.2)

2. **Create `src/protocol/node.py`**
   - Implement `Node` class with 3 queues (A, B, C)
   - Implement `enqueue_packet()` and `dequeue_next_packet()`
   - Test queue operations and drop policy
   - Test energy consumption formula

3. **Create `src/protocol/classifier.py`**
   - `PriorityClassifier.classify_packet()` function
   - Verify against Phase 1 rules

4. **Create `src/protocol/__init__.py`**
   - Export `Packet`, `Node`, `PriorityClassifier`

5. **Create `tests/test_protocol.py`**
   - Test priority mapping: 6 cases from Phase 1 §5.2
   - Test queue discipline: strict priority with fairness
   - Test energy accounting: verify formula matches Phase 1 §6
   - Test packet enqueue/dequeue

### 2.3 Acceptance Criteria

- [ ] `Packet` class correctly computes priority for all 6 cases (Phase 1 §5.2)
- [ ] `Node` maintains 3 separate queues per class
- [ ] Strict priority scheduling (A > B > C) verified via tests
- [ ] Energy consumption formula matches Phase 1 §6 exactly
- [ ] Drop policy: C→B→A implemented correctly
- [ ] All tests pass: `pytest tests/test_protocol.py -v`
- [ ] Code is well-documented with docstrings

---

## Task 3: Integration & Testing

### 3.1 Create Utility Module (`src/utils.py`)

```python
def load_synthetic_data(csv_path: str) -> list[Packet]:
    """Load synthetic dataset and create Packet objects."""
    df = pd.read_csv(csv_path)
    packets = []
    for _, row in df.iterrows():
        pkt = Packet(
            pkt_id=int(row['packet_id']),
            src=int(row['src_node_id']),
            dst=int(row['dst_node_id']),
            class_id=int(row['class_id']),
            payload_size=int(row['payload_size']),
            timestamp=float(row['timestamp']),
            depth=float(row['depth']),
            residual_energy=float(row['residual_energy']),
            sensed_value=0.0,  # Can add to CSV if needed
            is_anomaly=bool(row['is_anomaly'])
        )
        packets.append(pkt)
    return packets

def compute_class_distribution(packets: list[Packet]) -> dict:
    """Return {0: count_C, 1: count_B, 2: count_A}."""
    dist = {0: 0, 1: 0, 2: 0}
    for pkt in packets:
        dist[pkt.class_id] += 1
    return dist
```

### 3.2 Integration Test

```bash
# Generate synthetic data
python syn_data/packet_generator.py --nodes 50 --packets 5000 --output syn_data/dataset.csv

# Run protocol tests
pytest tests/test_protocol.py -v

# Load dataset and test on real packets
python -c "from src.utils import load_synthetic_data; packets = load_synthetic_data('syn_data/dataset.csv'); print(f'Loaded {len(packets)} packets'); print(f'Class dist: {compute_class_distribution(packets)}');"
```

---

## Task 4: Documentation & Cleanup

1. **Create `src/README.md`**
   - Quick start guide for using `Packet`, `Node`, `PriorityClassifier`
   - Example: creating a packet and checking its priority
   - Example: initializing nodes and enqueuing packets

2. **Create `syn_data/README.md`**
   - How to run the generator
   - Description of CSV columns
   - Expected class distributions

3. **Update main README**
   - Add link to Phase 2 Implementation doc
   - Note on running tests

4. **Code style**
   - Follow PEP 8
   - Use type hints
   - Minimum docstring for all public functions

---

## Timeline & Milestones

| Task | Effort | Status |
|------|--------|--------|
| Task 1: Data Generator | 2–3 days | Pending |
| Task 2: Core Protocol | 3–4 days | Pending |
| Task 3: Integration | 1–2 days | Pending |
| Task 4: Docs & Polish | 1 day | Pending |
| **Total Phase 2** | **7–10 days** | **In Progress** |

---

## Dependencies

```
Python 3.9+
pandas >= 1.3
numpy >= 1.21
pytest >= 6.0 (for testing)
```

Create `requirements.txt`:
```
pandas>=1.3
numpy>=1.21
pytest>=6.0
```

---

## Definition of Done (Phase 2)

- [ ] Both `syn_data/` and `src/` modules work independently
- [ ] All unit tests pass with 100% coverage of core logic
- [ ] Synthetic dataset generated and verified
- [ ] Packet priority computed correctly for all test cases
- [ ] Documentation complete
- [ ] README updated with Phase 2 link
- [ ] All code committed to main branch

**Once Phase 2 is done:** Ready for Phase 3 (simulation integration)
