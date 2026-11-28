import os
import sys

import json5
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_region_config import region_config

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_counts_forecast_features_from_the_da_file(tmp_path):
    (tmp_path / "CISO").mkdir()
    (tmp_path / "CISO" / "CISO_168hr_forecasts_DA.csv").write_text(
        "datetime,w1,w2,w3,w4,w5,avg_coal_production_forecast,avg_solar_production_forecast\n")
    with open(os.path.join(REPO, "src", "secondTierConfig.json")) as f:
        base = json5.load(f)
    config = region_config(base, "CISO", data_dir=str(tmp_path))
    assert config["REGION_DIRECT"] == config["REGION_LIFECYCLE"] == ["CISO"]
    assert config["CISO"]["NUM_FORECAST_FEATURES"] == 7
    assert config["CISO"]["DIRECT_CEF_IN_FILE_NAME"] == str(tmp_path / "CISO" / "CISO_direct_emissions.csv")
    assert config["CONCORDANCE"] == base["CONCORDANCE"] and base["REGION_DIRECT"] == ["AECI"]  # base untouched


def test_rejects_a_file_without_a_leading_datetime_column(tmp_path):
    (tmp_path / "X").mkdir()
    (tmp_path / "X" / "X_168hr_forecasts_DA.csv").write_text("time,a\n")
    with pytest.raises(ValueError, match="datetime"):
        region_config({}, "X", data_dir=str(tmp_path))
