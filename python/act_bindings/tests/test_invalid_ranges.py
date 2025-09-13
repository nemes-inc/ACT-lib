import pytest
import numpy as np
from pyact.mpbfgs import ActEngine

FS = 256.0
LENGTH = 512

def test_invalid_order():
    """Verify that a non-positive order raises an error."""
    engine = ActEngine(FS, LENGTH)
    signal = np.random.randn(LENGTH)
    with pytest.raises(ValueError, match="order must be a positive integer"):
        engine.transform(signal, order=0)
    with pytest.raises(ValueError, match="order must be a positive integer"):
        engine.transform(signal, order=-1)

def test_invalid_tc_range():
    """Verify that invalid time-center ranges raise errors."""
    with pytest.raises(ValueError, match="tc_min must be non-negative"):
        ActEngine(FS, LENGTH, ranges={"tc_min": -1})
    with pytest.raises(ValueError, match="tc_max must not exceed signal length - 1"):
        ActEngine(FS, LENGTH, ranges={"tc_max": LENGTH})
    with pytest.raises(ValueError, match="tc_min must be less than tc_max"):
        ActEngine(FS, LENGTH, ranges={"tc_min": 100, "tc_max": 50})
    with pytest.raises(ValueError, match="tc_step must be positive"):
        ActEngine(FS, LENGTH, ranges={"tc_step": 0})

def test_invalid_fc_range():
    """Verify that invalid frequency-center ranges raise errors."""
    with pytest.raises(ValueError, match="fc_min must be non-negative"):
        ActEngine(FS, LENGTH, ranges={"fc_min": -1})
    with pytest.raises(ValueError, match="fc_max must not exceed Nyquist frequency"):
        ActEngine(FS, LENGTH, ranges={"fc_max": FS})
    with pytest.raises(ValueError, match="fc_min must be less than fc_max"):
        ActEngine(FS, LENGTH, ranges={"fc_min": 50, "fc_max": 20})
    with pytest.raises(ValueError, match="fc_step must be positive"):
        ActEngine(FS, LENGTH, ranges={"fc_step": -0.5})

def test_invalid_logDt_range():
    """Verify that invalid duration ranges raise errors."""
    with pytest.raises(ValueError, match="logDt_min must be less than logDt_max"):
        ActEngine(FS, LENGTH, ranges={"logDt_min": -1.0, "logDt_max": -2.0})
    with pytest.raises(ValueError, match="logDt_step must be positive"):
        ActEngine(FS, LENGTH, ranges={"logDt_step": 0})

def test_invalid_c_range():
    """Verify that invalid chirp rate ranges raise errors."""
    with pytest.raises(ValueError, match="c_min must be less than c_max"):
        ActEngine(FS, LENGTH, ranges={"c_min": 10, "c_max": -10})
    with pytest.raises(ValueError, match="c_step must be positive"):
        ActEngine(FS, LENGTH, ranges={"c_step": 0})
