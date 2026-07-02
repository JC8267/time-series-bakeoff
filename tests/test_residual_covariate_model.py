from __future__ import annotations

import pandas as pd

from run_residual_covariate_model import feature_matrix


def test_feature_matrix_marks_black_friday_band() -> None:
    features = feature_matrix(pd.Series(pd.to_datetime(["2025-11-28"])), "f1", openings=None, covariates=None)

    assert features.loc[0, "black_friday_day"] == 1.0
    assert features.loc[0, "thanksgiving_post_1_7"] == 1.0
