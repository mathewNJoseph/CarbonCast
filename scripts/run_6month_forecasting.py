#!/usr/bin/env python3
"""
6-Month Chunk Forecasting Wrapper Script for CarbonCast

This script implements the 6-month chunk forecasting approach by running
the pipeline twice with different configurations and merging the results.

Requirements:
- First Run (H1 2023): Train Jan-Jun 2022, Val Jul-Dec 2022, Test Jan-Jun 2023
- Second Run (H2 2023): Train Jul-Dec 2022, Val Jan-Jun 2023, Test Jul-Dec 2023
- Output: Combined 2023 forecast covering full year

"Test Jan-Jun 2023" means one forecast starting on each day of that range;
the last one runs PREDICTION_WINDOW_HOURS past it. See CHUNKS.
"""

import sys
import os
import json5 as json
import pandas as pd
import argparse
from datetime import datetime
import shutil
import traceback

# Add src directory to path to import secondTierForecasts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import common
import secondTierForecasts

# Calendar dates of each chunk. test_start..test_end are the days the test
# forecasts START on (one forecast per day). Training runs up to the day before
# val_start, and validation up to the day before test_start.
CHUNKS = {
    "H1": {"train_start": "2022-01-01", "val_start": "2022-07-01",
           "test_start": "2023-01-01", "test_end": "2023-06-30"},
    "H2": {"train_start": "2022-07-01", "val_start": "2023-01-01",
           "test_start": "2023-07-01", "test_end": "2023-12-31"},
}

def create_chunk_config(base_config_path, chunk_type, output_config_path):
    """
    Create a modified config file for the specific 6-month chunk.
    
    Args:
        base_config_path: Path to the base configuration file
        chunk_type: "H1" or "H2" for first/second half of 2023
        output_config_path: Path where modified config should be saved
    """
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # NUM_TEST_DAYS and NUM_VAL_DAYS depend on the chunk's dates and on how far
    # the region's data reaches, so update_config_for_chunk_data sets them.

    # Each chunk trains its own model, so give each chunk its own model folder
    # (for example saved_second_tier_models/direct/H1/AECI/). With one shared
    # folder, H2 overwrites H1's model and scaling values, and -s loads the
    # wrong chunk's model.
    for key in ("DIRECT_SAVED_MODEL_LOCATION", "LIFECYCLE_SAVED_MODEL_LOCATION"):
        config[key] = os.path.join(config[key], chunk_type, "")
        for region in config["REGION_DIRECT"]:
            os.makedirs(os.path.join(config[key], region), exist_ok=True)

    # Update output file naming based on chunk
    for region in config["REGION_DIRECT"]:
        if region in config:
            # Update output file names to include H1 or H2 suffix
            direct_prefix = config[region]["DIRECT_CEF_OUT_FILE_NAME_PREFIX"]
            lifecycle_prefix = config[region]["LIFECYCLE_CEF_OUT_FILE_NAME_PREFIX"]
            
            # Replace the default suffix with chunk-specific suffix
            config[region]["DIRECT_CEF_OUT_FILE_NAME_PREFIX"] = direct_prefix.replace(
                "_168hr_CI_forecasts", f"_168hr_CI_forecasts_{chunk_type}"
            )
            config[region]["LIFECYCLE_CEF_OUT_FILE_NAME_PREFIX"] = lifecycle_prefix.replace(
                "_168hr_CI_forecasts", f"_168hr_CI_forecasts_{chunk_type}"
            )
    
    # Save the modified config
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created {chunk_type} config file: {output_config_path}")
    return config

def select_forecast_blocks(forecasts_df, start_date, end_date, block_hours, prediction_hours):
    """
    Keep the whole forecast blocks that secondTierForecasts pairs with the
    chunk's days, in file order.

    The forecasts file is one block_hours-row block per issue day
    (MAX_PREDICTION_WINDOW_HOURS), and the blocks overlap. secondTierForecasts
    splits it by row count and pairs block b with day b of the emissions chunk.
    It holds back the last BUFFER_HOURS = prediction_hours - 24 hours of data as
    actuals for the final forecasts, so the chunk needs one block per day from
    start_date through end_date minus BUFFER_HOURS // 24 days: whole, in issue
    order, none missing. Sorting or slicing by row timestamp breaks that.
    """
    if block_hours % 24 or prediction_hours % 24 or not 24 <= prediction_hours <= block_hours:
        raise ValueError(f"PREDICTION_WINDOW_HOURS ({prediction_hours}) and MAX_PREDICTION_WINDOW_HOURS "
                         f"({block_hours}) must be multiples of 24 with 24 <= prediction <= max")
    buffer_days = (prediction_hours - 24) // 24   # BUFFER_HOURS // 24 in secondTierForecasts
    last = pd.Timestamp(end_date).normalize() - pd.Timedelta(days=buffer_days)
    return common.selectDailyBlocks(forecasts_df, start_date, last, block_hours)

def last_full_day(df):
    """The last day whose 23:00 hour is present in an hourly, sorted DataFrame."""
    return (df.index[-1] + pd.Timedelta(hours=1)).normalize() - pd.Timedelta(days=1)

def modify_data_for_chunk(chunk_type, region, config, cef):
    """
    Write the chunk's emissions and forecast files, cut so that
    secondTierForecasts' row-count split lands on the dates in CHUNKS.

    secondTierForecasts takes the last NUM_TEST_DAYS + BUFFER_HOURS // 24 days
    as test data, the NUM_VAL_DAYS before that as validation, and the rest as
    training. So the chunk runs from train_start through test_end plus the
    buffer days, which hold the actuals for the last forecast's horizon.

    If the region's data stops too early for the last test forecasts (their
    actuals, or the forecast block issued on their start day), test_end is
    moved back to the last day the data supports and a warning names the
    skipped days.

    Only the emissions of cef ("direct" or "lifecycle"), the type being run, are read, so a region needs
    no file for the other type.

    Returns the emissions and forecast file paths, NUM_VAL_DAYS and NUM_TEST_DAYS.
    """

    # Read the full dataset
    emissions_path = f"data/{region}/{region}_{cef}_emissions.csv"
    forecasts_path = f"data/{region}/{region}_168hr_forecasts_DA.csv"
    
    print(f"Modifying data files for {chunk_type} chunk...")
    
    # Read the datasets
    emissions_df = pd.read_csv(emissions_path, parse_dates=['UTC time'], index_col=['UTC time'])
    forecasts_df = pd.read_csv(forecasts_path, parse_dates=['datetime'], index_col=['datetime'])
    
    # Sort by index to ensure monotonic datetime index. The forecasts file is
    # deliberately NOT sorted: see select_forecast_blocks.
    emissions_df = emissions_df.sort_index()

    block_hours = config["MAX_PREDICTION_WINDOW_HOURS"]
    prediction_hours = config["PREDICTION_WINDOW_HOURS"]
    buffer_days = pd.Timedelta(days=(prediction_hours - 24) // 24)   # BUFFER_HOURS // 24
    dates = {k: pd.Timestamp(v) for k, v in CHUNKS[chunk_type].items()}

    # A test period ending on day d needs actuals through d + buffer_days, and
    # select_forecast_blocks needs one forecast block per day through d.
    supported_end = min(last_full_day(emissions_df) - buffer_days,
                        forecasts_df.index[::block_hours].max())
    test_end = min(dates["test_end"], supported_end)
    if test_end < dates["test_start"]:
        raise ValueError(f"{region} data ends too early for any {chunk_type} test forecast: the first one, "
                         f"starting {dates['test_start'].date()}, needs actuals through "
                         f"{(dates['test_start'] + buffer_days).date()}")
    if test_end < dates["test_end"]:
        print(f"WARNING: {region} data supports {chunk_type} test forecasts starting up to {test_end.date()} "
              f"only. The {(dates['test_end'] - test_end).days} forecasts starting "
              f"{(test_end + pd.Timedelta(days=1)).date()}..{dates['test_end'].date()} are skipped. Restoring them "
              f"needs actuals through {(dates['test_end'] + buffer_days).date()} and forecast blocks issued "
              f"through {dates['test_end'].date()}.")

    start_date = dates["train_start"]
    end_date = test_end + buffer_days + pd.Timedelta(hours=23)
    num_val_days = (dates["test_start"] - dates["val_start"]).days
    num_test_days = (test_end - dates["test_start"]).days + 1

    # Filter datasets by date range
    emissions_chunk = emissions_df[start_date:end_date]
    expected_hours = len(pd.date_range(start_date, end_date, freq="h"))
    if len(emissions_chunk) != expected_hours:
        raise ValueError(f"{region} {cef} emissions have {len(emissions_chunk)} rows for {start_date} to {end_date}, "
                         f"not one per hour ({expected_hours}); the row-count split needs every hour")
    forecasts_chunk = select_forecast_blocks(forecasts_df, start_date, end_date, block_hours, prediction_hours)
    
    # Save temporary chunk files
    chunk_dir = f"data/{region}/temp_{chunk_type}"
    os.makedirs(chunk_dir, exist_ok=True)
    
    emissions_chunk_path = f"{chunk_dir}/{region}_{cef}_emissions_{chunk_type}.csv"
    forecasts_chunk_path = f"{chunk_dir}/{region}_168hr_forecasts_DA_{chunk_type}.csv"
    
    # Reset index to save UTC time as column
    emissions_chunk.reset_index().to_csv(emissions_chunk_path, index=False)
    forecasts_chunk.reset_index().to_csv(forecasts_chunk_path, index=False)
    
    print(f"Created chunk data files for {chunk_type}:")
    print(f"  {cef.capitalize()} emissions: {emissions_chunk_path}")
    print(f"  Forecasts: {forecasts_chunk_path}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Data points: {len(emissions_chunk)} hours")
    print(f"  Train {start_date.date()}..{(dates['val_start'] - pd.Timedelta(days=1)).date()}, "
          f"val {dates['val_start'].date()}..{(dates['test_start'] - pd.Timedelta(days=1)).date()}, "
          f"test forecasts starting {dates['test_start'].date()}..{test_end.date()} ({num_test_days} days)")

    return emissions_chunk_path, forecasts_chunk_path, num_val_days, num_test_days

def update_config_for_chunk_data(config_path, chunk_type, cef, emissions_path, forecasts_path, region,
                                 num_val_days, num_test_days):
    """Point the config at the chunk's data files and set the chunk's day counts."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update file paths for the region
    config[region][f"{cef.upper()}_CEF_IN_FILE_NAME"] = emissions_path
    config[region]["FORECAST_IN_FILE_NAME"] = forecasts_path
    config["NUM_VAL_DAYS"] = num_val_days
    config["NUM_TEST_DAYS"] = num_test_days

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated config to use {chunk_type} chunk data files")

def run_chunk_forecast(config_path, cef_type, load_saved_model, chunk_type):
    """Run the forecasting for a specific chunk."""
    print(f"\n{'='*60}")
    print(f"Running {chunk_type} 2023 Forecast")
    print(f"Config: {config_path}")
    print(f"CEF Type: {cef_type}")
    print(f"{'='*60}")
    
    try:
        # Run the second tier forecasting
        secondTierForecasts.runSecondTier(config_path, cef_type, load_saved_model)
        print(f"\n✅ {chunk_type} forecasting completed successfully!")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ {chunk_type} forecasting failed: {str(e)}")
        return False

def merge_forecast_results(region, forecast_type, iteration):
    """
    Merge H1 and H2 forecast results preserving 168hr overlapping window structure.
    For 168-hour forecasts, overlapping forecast windows are INTENTIONAL and must be preserved.
    Simply concatenates both files to maintain the complete overlapping forecast structure.
    
    Args:
        region: Region code (e.g., 'AECI', 'TVA')
        forecast_type: Type of forecast ('direct' or 'lifecycle')
        iteration: Iteration number
        
    Returns:
        str: Path to the merged output file
    """
    print(f"Starting merge process for {region} {forecast_type} forecasts, iteration {iteration}")
    
    # Define file paths
    h1_file = f"CI_forecast_data/{region}/{region}_{forecast_type}_168hr_CI_forecasts_H1_{iteration}.csv"
    h2_file = f"CI_forecast_data/{region}/{region}_{forecast_type}_168hr_CI_forecasts_H2_{iteration}.csv"
    merged_file = f"CI_forecast_data/{region}/{region}_{forecast_type}_168hr_CI_forecasts_merged_2023.csv"
    
    print(f"H1 file: {h1_file}")
    print(f"H2 file: {h2_file}")
    print(f"Output file: {merged_file}")
    
    # Check if input files exist
    if not os.path.exists(h1_file):
        raise FileNotFoundError(f"H1 file not found: {h1_file}")
    if not os.path.exists(h2_file):
        raise FileNotFoundError(f"H2 file not found: {h2_file}")
    
    # Read the files
    print("Reading H1 and H2 files...")
    h1_df = pd.read_csv(h1_file)
    h2_df = pd.read_csv(h2_file)
    
    print(f"H1 original shape: {h1_df.shape}")
    print(f"H2 original shape: {h2_df.shape}")
    
    # Convert datetime columns to datetime objects for proper sorting
    h1_df['datetime'] = pd.to_datetime(h1_df['datetime'])
    h2_df['datetime'] = pd.to_datetime(h2_df['datetime'])
    
    print(f"H1 date range: {h1_df['datetime'].min()} to {h1_df['datetime'].max()}")
    print(f"H2 date range: {h2_df['datetime'].min()} to {h2_df['datetime'].max()}")
    
    # For 168hr forecasts: Keep ALL forecast windows from both files
    # The overlapping timestamps are different forecast windows and MUST be preserved
    print("Performing complete concatenation preserving all 168hr forecast windows...")
    merged_df = pd.concat([h1_df, h2_df], ignore_index=True)
    
    # DO NOT SORT - preserve the original forecast window block structure
    # Sorting would destroy the 168hr window grouping which is essential
    
    print(f"Merged shape after preserving all windows: {merged_df.shape}")
    print(f"Merged date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")
    
    # Count overlapping timestamps (this is EXPECTED and CORRECT for 168hr forecasts)
    duplicate_count = merged_df.duplicated(subset=['datetime']).sum()
    print(f"Overlapping datetime entries: {duplicate_count} (EXPECTED for 168hr overlapping windows)")
    
    # Save the merged dataset
    print(f"Saving merged dataset to: {merged_file}")
    merged_df.to_csv(merged_file, index=False)
    
    print("Merge process completed successfully!")
    print(f"Total forecast points: {len(merged_df)} (preserving all overlapping windows)")
    print("✅ Complete 168hr overlapping forecast window structure preserved")
    
    return merged_file

def cleanup_temporary_files(region):
    """Clean up temporary chunk data files."""
    for chunk_type in ["H1", "H2"]:
        temp_dir = f"data/{region}/temp_{chunk_type}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        
        # Clean up temporary config files
        temp_config = f"src/secondTierConfig_{chunk_type}.json"
        if os.path.exists(temp_config):
            os.remove(temp_config)
            print(f"Cleaned up temporary config: {temp_config}")

def main():
    parser = argparse.ArgumentParser(description='Run 6-month chunk forecasting for CarbonCast')
    parser.add_argument('config_file', help='Base configuration file path')
    
    # Create mutually exclusive group for CEF type
    cef_group = parser.add_mutually_exclusive_group(required=True)
    cef_group.add_argument('-d', '--direct', action='store_const', const='-d', dest='cef_type',
                          help='Use direct CEF type')
    cef_group.add_argument('-l', '--lifecycle', action='store_const', const='-l', dest='cef_type',
                          help='Use lifecycle CEF type')
    
    parser.add_argument('-s', '--load-saved-model', action='store_true',
                       help='Load from saved model instead of training')
    parser.add_argument('--region', default=None, help='Region to forecast (auto-detected from config if not specified)')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup of temporary files')
    
    args = parser.parse_args()
    
    # Auto-detect region from config if not specified
    if args.region is None:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        
        # Get active region from config
        if args.cef_type == '-d' and 'REGION_DIRECT' in config:
            regions = config['REGION_DIRECT']
        elif args.cef_type == '-l' and 'REGION_LIFECYCLE' in config:
            regions = config['REGION_LIFECYCLE']
        else:
            raise ValueError("Could not detect active region from config file")
        
        if len(regions) != 1:
            raise ValueError(f"Config must specify exactly one region, found: {regions}")
        
        args.region = regions[0]
        print(f"Auto-detected region from config: {args.region}")
    
    print("CarbonCast 6-Month Chunk Forecasting")
    print("=" * 50)
    print(f"Base config: {args.config_file}")
    print(f"CEF type: {args.cef_type}")
    print(f"Region: {args.region}")
    print(f"Load saved model: {args.load_saved_model}")
    
    success_count = 0
    merged = False  # the run succeeded only if both chunks ran and were merged
    
    try:
        # Run both chunks
        for chunk_type in ["H1", "H2"]:
            print(f"\n🚀 Starting {chunk_type} 2023 forecasting...")
            
            # Create chunk-specific config
            chunk_config_path = f"src/secondTierConfig_{chunk_type}.json"
            chunk_config = create_chunk_config(args.config_file, chunk_type, chunk_config_path)

            # Create chunk-specific data files
            cef = "direct" if args.cef_type == "-d" else "lifecycle"
            emissions_path, forecasts_path, num_val_days, num_test_days = modify_data_for_chunk(
                chunk_type, args.region, chunk_config, cef)

            # Update config to use chunk data
            update_config_for_chunk_data(chunk_config_path, chunk_type, cef,
                                       emissions_path, forecasts_path, args.region,
                                       num_val_days, num_test_days)
            
            # Run forecasting for this chunk
            if run_chunk_forecast(chunk_config_path, args.cef_type, args.load_saved_model, chunk_type):
                success_count += 1
            else:
                print(f"❌ {chunk_type} forecasting failed. Continuing with next chunk...")
        
        # Merge results if both chunks completed successfully
        if success_count == 2:
            print(f"\n🔄 Both chunks completed successfully. Merging results...")
            forecast_type = "direct" if args.cef_type == "-d" else "lifecycle"
            merged_file = merge_forecast_results(args.region, forecast_type, 0)
            if merged_file:
                merged = True
                print(f"\n🎉 6-Month Chunk Forecasting Completed Successfully!")
                print(f"Final merged forecast: {merged_file}")
            else:
                print(f"\n⚠️  Forecasting completed but merging failed.")
        else:
            print(f"\n⚠️  Only {success_count}/2 chunks completed successfully.")
            print("Merged results not generated.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ Unexpected error: {str(e)}")
    
    finally:
        # Cleanup temporary files unless requested not to
        if not args.no_cleanup:
            print(f"\n🧹 Cleaning up temporary files...")
            cleanup_temporary_files(args.region)
        else:
            print(f"\n📁 Temporary files preserved (--no-cleanup specified)")
    
    print(f"\n✅ Script completed.")
    # Exit non-zero on failure, so a batch job that runs this script is marked failed.
    sys.exit(0 if merged else 1)

if __name__ == "__main__":
    main()