"""Write a tier 2 config that runs one region from its files under data/<REGION>/.

    python scripts/make_region_config.py src/secondTierConfig.json CISO src/secondTierConfig_CISO.json

The base config's settings (horizons, hyperparameters, CONCORDANCE) are kept. The region's entry is replaced so
that it reads data/<REGION>/<REGION>_168hr_forecasts_DA.csv and the emissions files next to it, and
NUM_FORECAST_FEATURES is set to the DA file's columns other than datetime: 5 weather columns plus one tier 1
forecast per source, which differs between regions.
"""
import argparse
import os

import json5 as json
import pandas as pd


def region_config(base, region, data_dir="data"):
    config = dict(base)
    forecasts = os.path.join(data_dir, region, f"{region}_168hr_forecasts_DA.csv")
    columns = pd.read_csv(forecasts, nrows=0).columns
    if columns[0] != "datetime":
        raise ValueError(f"{forecasts} should start with a datetime column, not {columns[0]!r}")
    config["REGION_DIRECT"] = config["REGION_LIFECYCLE"] = [region]
    config[region] = {
        "DIRECT_CEF_IN_FILE_NAME": os.path.join(data_dir, region, f"{region}_direct_emissions.csv"),
        "LIFECYCLE_CEF_IN_FILE_NAME": os.path.join(data_dir, region, f"{region}_lifecycle_emissions.csv"),
        "FORECAST_IN_FILE_NAME": forecasts,
        "DIRECT_CEF_OUT_FILE_NAME_PREFIX": f"CI_forecast_data/{region}/{region}_direct_168hr_CI_forecasts",
        "LIFECYCLE_CEF_OUT_FILE_NAME_PREFIX": f"CI_forecast_data/{region}/{region}_lifecycle_168hr_CI_forecasts",
        "NUM_FORECAST_FEATURES": len(columns) - 1,
    }
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_config")
    parser.add_argument("region")
    parser.add_argument("out_config")
    args = parser.parse_args()
    with open(args.base_config) as f:
        config = region_config(json.load(f), args.region)
    os.makedirs(f"CI_forecast_data/{args.region}", exist_ok=True)
    with open(args.out_config, "w") as f:
        json.dump(config, f, indent=4, quote_keys=True)
    print(f"Wrote {args.out_config}: {args.region}, {config[args.region]['NUM_FORECAST_FEATURES']} forecast features")


if __name__ == "__main__":
    main()
