# UWSN Priority-Driven Transport - Setup & Installation Guide

## Quick Start

This guide walks you through setting up and running the UWSN Priority-Driven Transport project on your local machine.

---

## Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Virtual Environment** (recommended: `venv` or `conda`)

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Divyanshu-Off/Content-Aware-Priority-Driven-Transport-in-Underwater-Wireless-Sensor-Networks.git
cd Content-Aware-Priority-Driven-Transport-in-Underwater-Wireless-Sensor-Networks
```

### 2. Create a Virtual Environment

**Using venv:**
```bash
python -m venv uwsn_env
source uwsn_env/bin/activate  # On Windows: uwsn_env\\Scripts\\activate
```

**Using conda:**
```bash
conda create -n uwsn python=3.9
conda activate uwsn
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas>=1.3` - Data processing
- `numpy>=1.21` - Numerical computing
- `pytest>=6.0` - Unit testing
- `matplotlib` - Visualization (optional)
- `seaborn` - Advanced plotting (optional)

---

## Project Structure

```
Content-Aware-Priority-Driven-Transport-in-Underwater-Wireless-Sensor-Networks/
├── docs/
│   ├── PHASE_1_DESIGN.md              # Network model, traffic classes, packet schema
│   ├── PHASE_2_IMPLEMENTATION.md      # Synthetic data generator & protocol logic
│   └── PHASE_3_SIMULATION.md          # Simulator setup & experiment framework
├── syn_data/
│   ├── packet_generator.py            # Synthetic traffic generator
│   ├── dataset.csv                    # Generated packet dataset
│   └── README.md                      # Data generation docs
├── src/
│   ├── protocol/
│   │   ├── packet.py                  # Packet class with priority mapping
│   │   ├── node.py                    # Node class with queuing logic
│   │   ├── classifier.py              # Priority classifier
│   │   └── __init__.py
│   ├── simulator/
│   │   ├── event_simulator.py         # Discrete-event simulator
│   │   ├── uwsn_network.py            # Network topology and operations
│   │   └── __init__.py
│   ├── utils.py                       # Utility functions
│   └── README.md                      # Source code docs
├── tests/
│   ├── test_protocol.py               # Unit tests for protocol logic
│   ├── test_simulator.py              # Simulator tests
│   └── test_data_generation.py        # Data generator tests
├── results/
│   ├── logs/                          # Simulation logs
│   └── plots/                         # Generated plots & figures
├── SETUP.md                           # This file
├── requirements.txt                   # Python dependencies
├── README.md                          # Project overview
└── LICENSE                            # MIT License
```

---

## Usage Guide

### Phase 1: Review Design Specification

Start by reading the design document to understand the network model, traffic classes, and protocol:

```bash
cat docs/PHASE_1_DESIGN.md
```

### Phase 2: Generate Synthetic Data

Generate synthetic UWSN traffic packets:

```bash
python syn_data/packet_generator.py
```

This creates `syn_data/dataset.csv` with 5000 realistic packets.

**Output:**
```
[TrafficGenerator] Generated 5000 packets
  Class A (emergency): 500 (10.0%)
  Class B (control): 250 (5.0%)
  Class C (routine): 4250 (85.0%)
  Anomalies: 147
  Saved to syn_data/dataset.csv
```

### Phase 2: Run Unit Tests

Test the protocol logic (packet priority mapping, queue discipline, energy accounting):

```bash
pytest tests/test_protocol.py -v
```

Expected output:
```
test_priority_mapping_class_a PASSED
test_priority_mapping_class_b PASSED
test_priority_mapping_class_c PASSED
test_queue_strict_priority PASSED
test_energy_consumption PASSED
```

### Phase 3: Run Simulation

Integrate protocol logic with the discrete-event simulator and run experiments:

```bash
python src/simulator/run_simulation.py \
  --nodes 50 \
  --duration 1000 \
  --output results/logs/sim_baseline.log
```

**Command-line options:**
- `--nodes` (int, default=50): Number of sensor nodes
- `--duration` (int, default=1000): Simulation time units
- `--output` (str): Log file path
- `--seed` (int): Random seed for reproducibility
- `--verbose` (flag): Print detailed event logs

**Output:**
```
Simulation Results:
  Network Lifetime: 845.23 seconds
  Total Packets Generated: 6500
  Total Packets Delivered: 6148 (94.6%)
  Average Delay (Class A): 3.2s
  Average Delay (Class B): 8.1s
  Average Delay (Class C): 32.5s
  Energy Efficiency: 102.3 bits/Joule
```

### Phase 4: Analyze Results

Generate plots and statistical analysis:

```bash
python src/analysis/plot_results.py \
  --input results/logs/sim_baseline.log \
  --output results/plots/
```

---

## Key Configuration Parameters

Edit `src/config.py` to customize simulation parameters:

```python
# Network
NUM_NODES = 50
TRANSMISSION_RANGE = 300  # meters
INITIAL_ENERGY = 10000    # Joules

# Traffic
LAMBDA_A = 0.1  # Class A arrival rate (packets/node/sec)
LAMBDA_B = 0.01 # Class B arrival rate
LAMBDA_C = 0.02 # Class C arrival rate

# Protocol
QUEUE_CAPACITY = 100
MAX_HOPS = 10
STRICT_PRIORITY = True
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pandas'`

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Permission denied when running tests

**Solution:**
```bash
chmod +x tests/*.py
pytest tests/ -v
```

### Issue: Simulation runs very slowly

**Solution:** Reduce number of nodes or simulation duration:
```bash
python src/simulator/run_simulation.py --nodes 25 --duration 500
```

---

## Running Experiments

Compare different protocol configurations:

```bash
# Baseline (strict priority)
python src/simulator/run_simulation.py --nodes 50 --output results/baseline.log

# Weighted Fair Queuing (WFQ)
python src/simulator/run_simulation.py --nodes 50 --scheduler wfq --output results/wfq.log

# No Priority (FIFO)
python src/simulator/run_simulation.py --nodes 50 --scheduler fifo --output results/fifo.log

# Analyze differences
python src/analysis/compare_protocols.py \
  --baseline results/baseline.log \
  --alternatives results/wfq.log results/fifo.log
```

---

## Development Workflow

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes** and test locally:
   ```bash
   pytest tests/ -v
   ```

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Add: description of changes"
   git push origin feature/my-feature
   ```

4. **Create a Pull Request** on GitHub

---

## Documentation

- **Phase 1 (Design):** `docs/PHASE_1_DESIGN.md` - Network model, traffic classes, packet schema, KPIs
- **Phase 2 (Implementation):** `docs/PHASE_2_IMPLEMENTATION.md` - Synthetic data generator, protocol logic, unit tests
- **Phase 3 (Simulation):** `docs/PHASE_3_SIMULATION.md` - Simulator architecture, experiment setup, result analysis
- **Code:** Inline docstrings and README files in each module

---

## Performance Tips

1. **Reproducibility:** Always use `--seed` for consistent results
2. **Batch Runs:** Use shell scripts to run multiple experiments
3. **Profiling:** Enable `--verbose` to identify bottlenecks
4. **Scaling:** Start with 25 nodes, then scale to 50 for initial testing

---

## Contributing

Issues, suggestions, and pull requests are welcome! Please follow the style guide:
- Use PEP 8 for Python code
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation

---

## License

MIT License - See LICENSE file for details.

---

## Contact & Support

For questions or issues, open an issue on GitHub or contact the project maintainer.

**Project Repository:**
https://github.com/Divyanshu-Off/Content-Aware-Priority-Driven-Transport-in-Underwater-Wireless-Sensor-Networks
