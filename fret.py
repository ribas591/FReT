import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


@njit
def extract_LT(x: np.ndarray) -> float:
    """Extract local topology value from 3x3 neighborhood.

    Bit positions correspond to neighborhood cells:
      [0, 1, 2]     [1,   2,   4]
      [3, 4, 5]  = [128,  0,   8]
      [6, 7, 8]    [64,  32,  16]

    Bit is set if cell value >= center (index 4).
    """
    pos_weights = np.array([2**0, 2**1, 2**2, 2**7, 0, 2**3, 2**6, 2**5, 2**4])
    new_values = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] >= x[4]:
            new_values[i] = 1
    return np.sum(pos_weights * new_values)


@njit
def apply_extract_LT(dm: np.ndarray, dm_padded: np.ndarray) -> np.ndarray:
    """Apply extract_LT to all 3x3 neighborhoods in a distance matrix."""
    LT_Output = np.zeros_like(dm)
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            neighborhood = dm_padded[i : i + 3, j : j + 3]
            LT_Output[i, j] = extract_LT(neighborhood.flatten())
    return LT_Output


def FReT_forecast(
    x_train: np.ndarray, forecast_horizon: int, num_archetypes: int = 3
) -> np.ndarray | None:
    """Forecast using FReT (Forecasting using Recurrence Topology).

    Args:
        x_train: Training time series, shape (n_samples, 1)
        forecast_horizon: Number of steps to forecast
        num_archetypes: Minimum number of archetype states required

    Returns:
        Forecast array of shape (forecast_horizon,) or None if no archetypes found
    """
    dm = cdist(x_train, x_train, metric="euclidean")
    dm_padded = np.pad(dm, ((1, 1), (1, 1)), "constant", constant_values=np.nan)

    LT_Output = apply_extract_LT(dm, dm_padded)

    LT_layers = LT_Output[1:-1, 1:-1]
    bins = np.array([42.5, 85.5, 127.5, 170.5, 212.5])
    flattened_layers = np.digitize(LT_layers, bins) + 1

    prior_states = flattened_layers[:-1]
    rv_xm = flattened_layers[-1]
    S_im = (prior_states == rv_xm).mean(axis=1, keepdims=True)
    sim_flat = S_im.flatten()

    testseqth = np.arange(0.61, 1.01, 0.01)
    arche_idx = -1
    for b, thresh in enumerate(testseqth):
        candidates = np.where(sim_flat > thresh)[0]
        count = np.sum(candidates < x_train.shape[0] - forecast_horizon - 2)
        if count >= num_archetypes:
            arche_idx = b

    if arche_idx == -1:
        return None

    archetypes = np.where(sim_flat > testseqth[arche_idx])[0]
    window_offset = 3
    max_start_idx = len(x_train) - forecast_horizon - window_offset
    archetypes = archetypes[archetypes <= max_start_idx]

    if len(archetypes) == 0:
        return None

    arche_data = np.array(
        [
            x_train[(a + window_offset) : (a + forecast_horizon + window_offset), 0]
            for a in archetypes
        ]
    ).T
    return arche_data.mean(axis=1)


def mae(x_test: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error between test data and forecast."""
    return np.abs(x_test.flatten() - forecast).mean()


def align_forecast(forecast: np.ndarray, first_value: float) -> np.ndarray:
    """Shift forecast so first value matches a known value."""
    return forecast - forecast[0] + first_value
