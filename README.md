# FReT - Forecasting using Recurrence Topology

Python implementation of the FReT (Forecasting through Recurrent Topology) algorithm — a parameter-free method for multi-step time series forecasting based on recurrent topological pattern analysis.

Based on the work by [Chomiak, T., Hu, B. Time-series forecasting through recurrent topology. Commun Eng 3, 9 (2024).](https://doi.org/10.1038/s44172-023-00142-8)

## Description

FReT (Forecasting through Recurrent Topology) is a time series forecasting algorithm that requires no free parameters, hyperparameter tuning, or critical model assumptions. It is based on identifying recurrent patterns in a signal's local topology, which are then used to model the system's expected future behaviour.
Main steps of the algorithm:

1. Distance matrix construction — a Euclidean distance matrix is computed from the input time series.
2. Local topology extraction — for each pair of points, an 8-bit binary code is computed from the local 3×3 neighbourhood, encoding the curvature of the signal's surface.
3. Topological state matrix construction — the binary codes are arranged into a 2D matrix, where each row represents the topological state of the system at a given point in time.
4. Topological archetype detection — a similarity metric Sim is computed between the current system state (last row of the matrix) and all prior states; the problem reduces to a simple maximisation task.
5. Forecast generation — the identified archetypes (most similar past states) are used to produce the forecast: the predicted trajectory is computed as the element-wise average of the signal segments that followed each archetype.

## Repository Structure

```
.
├── fret.py              # Core FReT algorithm implementation
├── test_fret.py         # Unit tests
├── pyproject.toml       # Project configuration
├── README.md            # This file
├── AGENTS.md            # Developer guidelines
├── data/                # Time-series datasets
│   ├── Lorenz.csv        # Lorenz system (3D)
│   ├── Rossler.csv       # Rössler system
│   ├── MackeyGlass.csv   # Mackey-Glass equation
│   ├── SN_m_tot_V2.0.csv # Sunspot number (monthly)
│   └── SN_ms_tot_V2.0.csv # Sunspot number (smoothed)
└── examples/             # Jupyter notebooks
    ├── 01.synthetic_data.ipynb      # Basic usage with synthetic data
    ├── 02.dynamical_systems.ipynb   # Application to chaotic systems
    └── 03.sunspot_number.ipynb      # Solar cycle forecasting
```

## Installation

Requires Python 3.11+.

```bash
# Install core dependencies
uv sync

# Install with notebook support
uv sync --extra notebooks

# Install with dev tools (pytest, ruff)
uv sync --dev

# Install everything
uv sync --extra notebooks --dev
```
## Testing

```bash
# Run all tests
uv run pytest
```

## Quick Start

```python
import numpy as np
from fret import FReT_forecast

# Generate synthetic data
t = np.arange(0, 60, 0.1)
data = np.cos(2 * np.pi * t / 3) + 0.75 * np.sin(2 * np.pi * t / 5)

# Split into train/test
x_train = data[:500].reshape(-1, 1)
x_test = data[500:600].reshape(-1, 1)

# Forecast
forecast = FReT_forecast(x_train, forecast_horizon=100)
```

## Examples

### 1. Synthetic Data (`01.synthetic_data.ipynb`)
Basic demonstration of the FReT algorithm on synthetic periodic data.

### 2. Dynamical Systems (`02.dynamical_systems.ipynb`)
Application to well-known chaotic systems:
- **Lorenz** - 3D chaotic system (forecast X, Y, Z separately)
- **Rössler** - Single variable forecasting
- **Mackey-Glass** - Delayed feedback system

### 3. Sunspot Number Forecasting (`03.sunspot_number.ipynb`)
Application of FReT methods to a series of monthly sunspot numbers (SSN). See 
[Volobuev, D. M., Rybintsev, A. S., Makarenko,  N. G. Forecasting of The Solar Cycles Using Recurrent Topology Technique. (2026)]()

## Acknowledgements

Original reference implementation: https://github.com/tgchomia/ts

## Citation

If you use this code in your research, please cite:

Volobuev, D. M., Rybintsev, A. S., Makarenko, N. G.
*Forecasting of The Solar Cycles Using Recurrent Topology Technique.* (2026)
