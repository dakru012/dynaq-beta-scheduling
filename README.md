# Dyna-QB Agent with Beta Scheduling

This project implements a **Dyna-Q** reinforcement learning agent and augments it using **beta-scheduled mellowmax based soft Q-updates** based on UQL principles
(https://arxiv.org/pdf/2110.14818 by Liang et al.).  
The agent uses an **ensemble of Q-tables** to estimate uncertainty and dynamically adjust the inverse temperature β during training.  
It supports running on **MiniGrid** environments and includes utilities for tracking results and visualizing learning.

---

## Features
- **Dyna-Q** with planning steps from a learned transition model.
- **Beta scheduling** for state-dependent soft updates.
- **Ensemble-based uncertainty estimation**.
- Works with **MiniGrid** environments via `gymnasium`.
- Notebook support for experiments and analysis.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/dakru012/dynaq-beta-scheduling.git
cd dynaq-beta-scheduling
````

### 2. Create a virtual environment (recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

Install all required packages from `requirements.txt`: <br>
(If you don't want to use the visualization notebooks, you can remove the last segment of the `requirements.txt`)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Run training

```bash
python -m scripts.training
```

## Requirements

* Python **3.11.13**
* Dependencies listed in `requirements.txt`

---

## Project Structure

```
scripts/            # Training scripts
src/                # Code for the agents and models
img/                # Training runs and visualizaton scripts
requirements.txt    # Python dependencies
```
