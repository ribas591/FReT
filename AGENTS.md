# AGENTS.md - Time-Series Forecasting (FReT)

## Project Overview

FReT (Forecasting using Recurrence Topology) algorithm in `fret.py`. Uses `uv` for Python environment management.

## Build/Test Commands

### Environment Setup
```bash
uv sync                        # Core dependencies
uv sync --extra notebooks      # With notebook support
uv sync --dev                  # With dev tools (pytest, ruff)
uv sync --extra notebooks --dev # Everything
source .venv/bin/activate      # Activate venv
```

### Running Tests
```bash
uv run pytest                  # All tests
uv run pytest test_fret.py      # Single file
uv run pytest test_fret.py::TestExtractLT::test_all_zeros_returns_255  # Specific test
uv run pytest -v               # Verbose
uv run pytest -k "extract"     # Match pattern
```

### Code Quality
```bash
uv run ruff check .            # Lint
uv run ruff format .          # Format
```

## Code Style

### Python 3.11+, Type Hints
```python
def extract_LT(x: np.ndarray) -> float: ...

def FReT_forecast(
    x_train: np.ndarray, forecast_horizon: int, num_archetypes: int = 3
) -> np.ndarray | None: ...
```

### Imports (stdlib → third-party)
```python
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist
```

### Numba
- Use `@njit` for tight loops
- Avoid lists inside `@njit`: use `np.array`, `np.zeros_like`

### Naming
- Functions/variables: `snake_case` (e.g., `extract_LT`, `dm_padded`)
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Error Handling
- Return `None` for expected failures
- Use `np.where()` over try/except for array ops

### Documentation
Concise docstrings for public APIs:
```python
def align_forecast(forecast: np.ndarray, first_value: float) -> np.ndarray:
    """align forecast with some known value"""
    return forecast - forecast[0] + first_value
```

## Testing

### Test Data (synthetic)
```python
t_step = np.arange(0.1, 70.1, 0.1)
data_tot = np.cos(2 * np.pi * t_step / 3) + 0.75 * np.sin(2 * np.pi * t_step / 5)
x_train = data_tot[:600].reshape(-1, 1)
x_test = data_tot[600:700].reshape(-1, 1)
```

### Coverage
- `extract_LT`: 3x3 windows with known outputs
- `create_layer`: Threshold boundaries
- `mae`, `align_forecast`: Deterministic tests
- `FReT_forecast`: Shape, regression, `None` case

## File Organization

- `fret.py`: Core FReT implementation
- `test_fret.py`: 16 tests
- `examples/*.ipynb`: Notebooks
- `data/`: CSV datasets

## Implementation Notes

### FReT_forecast
- Returns `None` if no archetypes found above similarity threshold
- Validates archetypes have enough points for forecast window
- For multi-variable systems (Lorenz), forecast X, Y, Z separately

### Data Loading
- Mackey-Glass: `pd.read_csv("data/MackeyGlass.csv", skiprows=1)` (NaN header)
- Notebooks: add `sys.path.insert(0, "..")` for imports

### Layer Thresholds
| Layer | Lower | Upper | Value |
|-------|-------|-------|-------|
| 1 | 0 | 42.5 | 1 |
| 2 | 42.5 | 85.5 | 2 |
| 3 | 85.5 | 127.5 | 3 |
| 4 | 127.5 | 170.5 | 4 |
| 5 | 170.5 | 212.5 | 5 |
| 6 | 212.5 | None | 6 |
