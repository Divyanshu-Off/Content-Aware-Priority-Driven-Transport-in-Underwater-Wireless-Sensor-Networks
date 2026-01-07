# Content-Aware Priority-Driven Transport in Underwater Wireless Sensor Networks (UWSNs)

## 📌 Project Overview
This project focuses on developing a **priority-driven packet transmission mechanism** for **Underwater Wireless Sensor Networks (UWSNs)**.  
Unlike terrestrial networks, UWSNs face challenges such as high propagation delay, limited bandwidth, dynamic topology, and high energy consumption.  

Our approach is **content-aware**: packets are prioritized based on their **type, importance, and network conditions**, ensuring critical data (e.g., disaster warnings, navigation signals, or marine monitoring) is transmitted reliably and with reduced latency.

---

## 📋 Project Documentation

- **[Phase 1 Design Specification](./docs/PHASE_1_DESIGN.md)** - Complete design doc with network model, traffic classes, packet schema, priority mapping rules, energy model, and KPIs. **Start here!**

## 🎯 Objectives
- Implement a **content-aware transport protocol** for UWSNs.  
- Prioritize packets based on **urgency, reliability requirements, and data sensitivity**.  
- Optimize **end-to-end delay, throughput, and energy consumption**.  
- Simulate and evaluate performance using **DESSERT (or NS-3)**.  
- Generate a **synthetic dataset** for testing when real-world data is unavailable.

---

## ⚙️ Features
- 📡 **Priority-driven packet scheduling**  
- 🔋 **Energy-efficient routing & transmission**  
- 🌊 **UWSN-specific propagation modeling**  
- 📊 **Performance metrics evaluation (Delay, PDR, Throughput, Energy)**  
- 🧪 **Simulation via DESSERT / NS-3**  
- 📑 **Synthetic dataset generation for controlled experiments**  

---

## 🛠️ Technology Stack
- **Simulation Environment**: DESSERT / NS-3  
- **Programming**: Python, C++  
- **Dataset**: Synthetic data generator (custom scripts)  
- **Visualization**: Matplotlib / Seaborn / Excel reports  

---
