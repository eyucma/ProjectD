import pytest
import numpy as np
import pandas as pd

# IMPORTANT: This assumes 'blackscholes_calculator.py' is in the same directory.
# We import the functions needed for testing directly from that file.
from app.utils.BlackScholes import BlackScholes, convert_to_numpy, phi

# --- Test Benchmarks ---

# Define a standard tolerance for floating point comparisons, approx tolerance for approximation
TOL = 1e-4
APPROX_TOL = 1e-3

# Test Case 1: Standard At-the-Money (ATM)
# S=100, K=100, T=1, r=0.05, q=0.00, vol=0.20
# Benchmark Call Price (norm.cdf): 10.45058357
# Benchmark Put Price (norm.cdf): 5.57352602
BENCHMARK_SCALAR = {
    "S": 100.0,
    "K": 100.0,
    "T": 1.0,
    "r": 0.05,
    "q": 0.00,
    "vol": 0.20,
}
atmc_exact = 10.45058357
atmp_exact = 5.573526022256971

# Test Case 2: Deep Out-of-the-Money (OTM)
# S=50, K=55, T=0.5, r=0.03, q=0.01, vol=0.35
# Benchmark Call Price (norm.cdf): 3.02458461
# Benchmark Put Price (norm.cdf): 7.44314157
BENCHMARK_OTM = {
    "S": 50.0,
    "K": 55.0,
    "T": 0.5,
    "r": 0.03,
    "q": 0.01,
    "vol": 0.35,
}
otmc_exact = 3.20173797
otmp_exact = 7.63227068688381

# --- Test Functions ---


def test_bs_scalar_exact():
    """Test Black-Scholes with scalar inputs using the accurate CDF (approx=False)."""
    # Test Call
    call_price = BlackScholes(**BENCHMARK_SCALAR, call=True, approx=False)
    assert np.isclose(call_price, atmc_exact, atol=TOL)

    # Test Put
    put_price = BlackScholes(**BENCHMARK_SCALAR, call=False, approx=False)
    assert np.isclose(put_price, atmp_exact, atol=TOL)


def test_bs_scalar_approx():
    """
    Test Black-Scholes with scalar inputs using the 'phi' approximation (approx=True).
    Tolerance is slightly higher due to approximation.
    """

    # Test Call
    call_price = BlackScholes(**BENCHMARK_SCALAR, call=True, approx=True)
    assert np.isclose(call_price, atmc_exact, atol=APPROX_TOL)

    # Test Put
    put_price = BlackScholes(**BENCHMARK_SCALAR, call=False, approx=True)
    assert np.isclose(put_price, atmp_exact, atol=APPROX_TOL)


def test_bs_vectorized_numpy():
    """Test Black-Scholes with vectorized NumPy inputs."""
    vectorized_inputs = {
        # Key is the parameter name (e.g., 'S')
        key: np.array([BENCHMARK_SCALAR[key], BENCHMARK_OTM[key]])
        # Iterate over all keys in the first dictionary
        for key in BENCHMARK_SCALAR.keys()
    }

    # Create NumPy arrays for multiple scenarios
    S_arr = vectorized_inputs["S"]
    K_arr = vectorized_inputs["K"]
    T_arr = vectorized_inputs["T"]
    r_arr = vectorized_inputs["r"]
    q_arr = vectorized_inputs["q"]
    vol_arr = vectorized_inputs["vol"]

    expected_calls = np.array([atmc_exact, otmc_exact])
    expected_puts = np.array([atmp_exact, otmp_exact])

    # Test Call
    call_prices = BlackScholes(
        S_arr, K_arr, T_arr, r_arr, q_arr, vol_arr, call=True, approx=False
    )
    assert np.allclose(call_prices, expected_calls, atol=TOL)

    # Test Puts
    puts_prices = BlackScholes(
        S_arr, K_arr, T_arr, r_arr, q_arr, vol_arr, call=False, approx=False
    )
    assert np.allclose(puts_prices, expected_puts, atol=TOL)


def test_bs_input_list():
    """Test Black-Scholes using standard Python lists as input."""
    S_list = [BENCHMARK_SCALAR["S"], BENCHMARK_OTM["S"]]
    K_list = [BENCHMARK_SCALAR["K"], BENCHMARK_OTM["K"]]
    T_list = [BENCHMARK_SCALAR["T"], BENCHMARK_OTM["T"]]
    r_list = [BENCHMARK_SCALAR["r"], BENCHMARK_OTM["r"]]
    q_list = [BENCHMARK_SCALAR["q"], BENCHMARK_OTM["q"]]
    vol_list = [BENCHMARK_SCALAR["vol"], BENCHMARK_OTM["vol"]]

    put_prices = BlackScholes(
        S_list,
        K_list,
        T_list,
        r_list,
        q_list,
        vol_list,  # Use scalars for other inputs
        call=False,
        approx=False,
    )

    # Verify the first element matches the benchmark
    assert np.isclose(put_prices[0], atmp_exact, atol=TOL)

    # Second:
    assert np.isclose(put_prices[1], otmp_exact, atol=TOL)


def test_bs_input_pandas_series():
    """Test Black-Scholes using Pandas Series as input."""
    S_series = pd.Series([BENCHMARK_SCALAR["S"], BENCHMARK_OTM["S"]])
    K_series = pd.Series([BENCHMARK_SCALAR["K"], BENCHMARK_OTM["K"]])
    T_series = pd.Series([BENCHMARK_SCALAR["T"], BENCHMARK_OTM["T"]])

    call_prices = BlackScholes(
        S_series, K_series, T_series, r=0.05, q=0.00, vol=0.20, call=True, approx=False
    )

    # Verify the output is a NumPy array (due to convert_to_numpy)
    assert isinstance(call_prices, np.ndarray)
    assert np.isclose(call_prices[0], atmc_exact, atol=TOL)


def test_convert_to_numpy_raises_error():
    """Test that convert_to_numpy raises TypeError for unsupported types."""
    with pytest.raises(TypeError):
        convert_to_numpy("not a number")

    with pytest.raises(TypeError):
        convert_to_numpy({"S": 100})


def test_phi_boundary_cases():
    """Test the 'phi' approximation at boundaries."""
    # Test N(0) = 0.5
    assert np.isclose(phi(0), 0.5, atol=1e-10)

    # Test N(-x) = 1 - N(x)
    x_val = 1.96
    phi_x = phi(x_val)
    phi_neg_x = phi(-x_val)
    assert np.isclose(phi_x + phi_neg_x, 1.0, atol=1e-5)

    # Test vector with mixed signs
    mixed_vector = np.array([-1.0, 0.0, 1.0])
    phi_mixed = phi(mixed_vector)
    # N(1) + N(-1) should be 1
    assert np.isclose(phi_mixed[0] + phi_mixed[2], 1.0, atol=1e-5)
    assert np.isclose(phi_mixed[1], 0.5, atol=1e-10)


def test_put_call_parity():
    """Test that Black-Scholes prices satisfy the Put-Call Parity condition."""
    S = BENCHMARK_OTM["S"]
    K = BENCHMARK_OTM["K"]
    T = BENCHMARK_OTM["T"]
    r = BENCHMARK_OTM["r"]
    q = BENCHMARK_OTM["q"]

    # 1. Calculate Call and Put prices using the most accurate method (approx=False)
    C = BlackScholes(**BENCHMARK_OTM, call=True, approx=False)
    P = BlackScholes(**BENCHMARK_OTM, call=False, approx=False)

    # 2. Calculate the theoretical arbitrage-free value (Forward Price)
    # PCP: C - P = S * exp(-q*T) - K * exp(-r*T)
    expected_difference = S * np.exp(-q * T) - K * np.exp(-r * T)
    actual_difference = C - P

    # 3. Assert that the actual difference equals the expected difference
    # The residual (actual - expected) should be close to zero.
    assert np.isclose(actual_difference, expected_difference, atol=TOL)
