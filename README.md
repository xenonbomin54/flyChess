# 🪰 FlyChess

> A computational experiment using fruit fly connectome data to play chess.

FlyChess explores whether a computational model based on the neural connections of a fruit fly can be used to make decisions in a chess environment.

The project uses publicly available fruit fly connectome data and attempts to connect neural activity with chess actions.

## 🎯 Goal

The goal of this project is not to create a conventional chess engine.

Instead, FlyChess explores a different question:

> **Can a computational model inspired by the fruit fly brain make meaningful decisions in chess?**

The project focuses on connecting:

Fruit Fly Connectome
↓
Neural Simulation
↓
Decision Making
↓
Chess Move

## 🧠 Approach

The project is divided into several components:

- **Connectome** — Loading and processing fruit fly neural connection data
- **Brain** — Simulating neural activity
- **Chess** — Representing the chess environment and converting neural output into chess actions
- **Experiments** — Running and recording different experiments
- **Web** — Visualizing the brain simulation and chess game

## 📁 Project Structure

    flyChess/
    ├── data/
    │   ├── processed/
    │   └── raw/
    │
    ├── src/
    │   ├── brain/
    │   │   ├── neuron.py
    │   │   ├── output.py
    │   │   └── simulation.py
    │   │
    │   ├── chess/
    │   │   ├── agent.py
    │   │   └── board.py
    │   │
    │   ├── connectome/
    │   │   ├── graph.py
    │   │   └── loader.py
    │   │
    │   ├── experiments/
    │   │   └── experiment_001.py
    │   │
    │   └── main.py
    │
    ├── tests/
    ├── web/
    │
    ├── .gitignore
    ├── README.md
    └── requirements.txt

## 🛠️ Tech Stack

- Python
- React
- TypeScript
- FlyWire Connectome Data
- More to be added

## 🚧 Project Status

**In Development**

Current focus:

- [ ] Explore fruit fly connectome data
- [ ] Load neural connection data
- [ ] Build a basic neural simulation
- [ ] Connect neural activity to decision making
- [ ] Integrate chess
- [ ] Build interactive visualization

## 🔬 Experiments

Experiments will be documented as the project develops.

Each experiment will investigate different ways of representing neural activity and connecting it to chess decision-making.

## ⚠️ Disclaimer

This project is an experimental computational model.

It does not claim that a real fruit fly understands or can play chess. The purpose is to explore what can be achieved by using biological neural connectivity as the basis for a computational system.

## 📌 References

- FlyWire
- More references to be added

---

Made with curiosity about brains, computation, and chess. 🪰♟️
