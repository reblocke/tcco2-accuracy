from __future__ import annotations

import numpy as np
import pandas as pd

from tcco2_accuracy.conway_meta import conway_group_summary
from tcco2_accuracy.data import load_conway_studies, prepare_conway_meta_inputs


def test_uploaded_table_changes_meta_outputs() -> None:
    studies = load_conway_studies()
    baseline = conway_group_summary(prepare_conway_meta_inputs(studies))

    synthetic = {
        "study_id": "Synthetic Study",
        "bias": 8.0,
        "sd": 5.0,
        "s2": 25.0,
        "n_pairs": 20,
        "n_participants": 20,
        "c": 1.0,
        "is_icu": 0,
        "is_arf": 0,
        "is_lft": 0,
    }
    updated = pd.concat([studies, pd.DataFrame([synthetic])], ignore_index=True)
    mutated = conway_group_summary(prepare_conway_meta_inputs(updated))

    assert mutated.bias != baseline.bias or mutated.sd != baseline.sd
    assert np.isfinite([mutated.bias, mutated.sd, mutated.tau2, mutated.loa_l, mutated.loa_u]).all()
