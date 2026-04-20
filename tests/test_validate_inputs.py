from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tcco2_accuracy.data import CONWAY_DATA_PATH
from tcco2_accuracy.validate_inputs import validate_conway_studies_df


def test_validate_conway_studies_ok() -> None:
    if not Path(CONWAY_DATA_PATH).exists():
        pytest.skip("Canonical Conway study table missing.")
    df = pd.read_csv(CONWAY_DATA_PATH)
    validate_conway_studies_df(df)


def test_validate_conway_studies_invalid() -> None:
    df = pd.DataFrame(
        {
            "study_id": ["bad"],
            "bias": [1.2],
            "sd": [1.5],
            "n_pairs": [-5],
            "n_participants": [10],
            "is_icu": [0],
            "is_arf": [0],
            "is_lft": [0],
        }
    )
    with pytest.raises(ValueError, match="n_pairs"):
        validate_conway_studies_df(df)

    df_missing = df.drop(columns=["sd"])
    with pytest.raises(ValueError, match="sd|s2"):
        validate_conway_studies_df(df_missing)
