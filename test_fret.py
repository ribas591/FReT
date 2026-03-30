import numpy as np
import pytest

from fret import FReT_forecast, align_forecast, extract_LT, mae


class TestExtractLT:
    def test_all_zeros_returns_255(self):
        x = np.zeros(9)
        assert extract_LT(x) == 255.0

    def test_center_only_larger(self):
        x = np.array([0, 0, 0, 0, 10, 0, 0, 0, 0])
        assert extract_LT(x) == 0

    def test_all_smaller_than_center(self):
        x = np.array([0, 0, 0, 0, 100, 1, 2, 3, 4])
        assert extract_LT(x) == 0

    def test_all_larger_than_center(self):
        x = np.array([10, 10, 10, 10, 0, 10, 10, 10, 10])
        assert extract_LT(x) == 255.0

    def test_top_left_corner_larger(self):
        x = np.array([10, 0, 0, 0, 5, 0, 0, 0, 0])
        assert extract_LT(x) == 1

    def test_top_row_larger(self):
        x = np.array([10, 10, 10, 0, 5, 0, 0, 0, 0])
        assert extract_LT(x) == 7

    def test_bottom_right_corner_larger(self):
        x = np.array([0, 0, 0, 0, 5, 0, 0, 0, 10])
        assert extract_LT(x) == 16


class TestMAE:
    def test_identical_arrays(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mae(x, x) == 0.0

    def test_known_difference(self):
        x_test = np.array([3.0, 3.0, 3.0])
        forecast = np.array([1.0, 1.0, 1.0])
        assert mae(x_test, forecast) == 2.0

    def test_single_element(self):
        x_test = np.array([5.0])
        forecast = np.array([3.0])
        assert mae(x_test, forecast) == 2.0


class TestAlignForecast:
    def test_align_to_first_value(self):
        forecast = np.array([10.0, 15.0, 20.0, 25.0])
        result = align_forecast(forecast, 100.0)
        expected = np.array([100.0, 105.0, 110.0, 115.0])
        np.testing.assert_array_equal(result, expected)

    def test_align_to_zero(self):
        forecast = np.array([5.0, 10.0, 15.0])
        result = align_forecast(forecast, 0.0)
        expected = np.array([0.0, 5.0, 10.0])
        np.testing.assert_array_equal(result, expected)


class TestFReTForecast:
    @pytest.fixture
    def synthetic_data(self):
        t_step = np.arange(0.1, 70.1, 0.1)
        data_tot = np.cos(2 * np.pi * t_step / 3) + 0.75 * np.sin(
            2 * np.pi * t_step / 5
        )
        x_train = data_tot[:600].reshape(-1, 1)
        x_test = data_tot[600:700].reshape(-1, 1)
        return x_train, x_test

    def test_returns_correct_shape(self, synthetic_data):
        x_train, _ = synthetic_data
        forecast_horizon = 100
        result = FReT_forecast(x_train, forecast_horizon)
        assert result is not None
        assert result.shape == (forecast_horizon,)

    def test_returns_non_none(self, synthetic_data):
        x_train, _ = synthetic_data
        result = FReT_forecast(x_train, 100)
        assert result is not None

    def test_regression_baseline(self, synthetic_data):
        x_train, _ = synthetic_data
        result = FReT_forecast(x_train, 100)
        baseline = np.array(
            [
                1.07214753,
                1.10006287,
                1.08511041,
                1.03044586,
                0.94083894,
                0.82242732,
                0.68241340,
                0.52871748,
                0.36960329,
                0.21329239,
                0.06758483,
                -0.06049695,
                -0.16502541,
                -0.24143216,
                -0.28670761,
                -0.29952731,
                -0.28029951,
                -0.23113206,
                -0.15572028,
                -0.05916106,
                0.05229826,
                0.17156495,
                0.29104588,
                0.40301692,
                0.50000000,
                0.57513068,
                0.62249958,
                0.63745204,
                0.61683235,
                0.55916106,
                0.46473727,
                0.33566053,
                0.17577105,
                -0.00948968,
                -0.21329239,
                -0.42769844,
                -0.64399158,
                -0.85304851,
                -1.04573243,
                -1.21329239,
                -1.34775090,
                -1.44226294,
                -1.49143039,
                -1.49155793,
                -1.44083894,
                -1.33946286,
                -1.18963887,
                -0.99553441,
                -0.76313053,
                -0.50000000,
                -0.21501707,
                0.08198895,
                0.38062188,
                0.67033225,
                0.94083894,
                1.18254094,
                1.38690193,
                1.54679140,
                1.65676789,
                1.71329239,
                1.71486304,
                1.66206550,
                1.55753704,
                1.40584604,
                1.21329239,
                0.98763728,
                0.73777441,
                0.47335647,
                0.20439334,
                -0.05916106,
                -0.30781535,
                -0.53292358,
                -0.72702804,
                -0.88414768,
                -1.00000000,
                -1.07214753,
                -1.10006287,
                -1.08511041,
                -1.03044586,
                -0.94083894,
                -0.82242732,
                -0.68241340,
                -0.52871748,
                -0.36960329,
                -0.21329239,
                -0.06758483,
                0.06049695,
                0.16502541,
                0.24143216,
                0.28670761,
                0.29952731,
                0.28029951,
                0.23113206,
                0.15572028,
                0.05916106,
                -0.05229826,
                -0.17156495,
                -0.29104588,
                -0.40301692,
                -0.50000000,
            ]
        )
        np.testing.assert_allclose(result, baseline, rtol=1e-5)  # pyright: ignore[reportCallIssue]

    def test_returns_none_no_archetypes(self):
        x_train = np.random.randn(10, 1)
        result = FReT_forecast(x_train, forecast_horizon=5, num_archetypes=100)
        assert result is None
