import pandas as pd
import pytest
import motorwrap as mw

def test_fit_peril_invalid_co_ownership():
    data = {
        'co_ownership': ['N', 'N', 'N'],
        'nu_cl': [1, 2, 3],
        'ex': [0.1, 0.2, 0.3]
    }
    aircraft_smaller = pd.DataFrame(data)

    with pytest.raises(Exception) as excinfo:
        mw.fit_peril(
            df=aircraft_smaller,
            target_col='nu_cl',
            weight_col='ex',
            model_dir="/mnt/extra/model_dir"
        )
    
    assert "conversion from `cat` to `f64` failed" in str(excinfo.value)