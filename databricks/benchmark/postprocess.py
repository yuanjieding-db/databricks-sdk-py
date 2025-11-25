#!/usr/bin/env python3
"""
Post-processing utilities for Databricks Files API benchmarking results.

This script provides functionality to:
1. Process raw benchmark CSV files into summarized format
2. Compare and merge multiple benchmark results
3. Generate summary reports with speedup calculations

Usage:
    python3 postprocess.py --postprocess
    python3 postprocess.py --comparison  
    python3 postprocess.py --summary input_file.csv
"""

import argparse
import csv
import datetime
import glob
import os
import re
import statistics
from collections import defaultdict


def postprocess_benchmarks():
    """
    Process raw benchmark CSV files by filtering relevant columns,
    grouping by key parameters, and computing median values.
    """
    input_files = glob.glob("benchmark_*.csv")
    if not input_files:
        print("No benchmark_*.csv files found to process.")
        return
    
    for input_file in input_files:
        output_file = input_file.replace("benchmark_", "processed_benchmark_")
        print(f"Processing {input_file} -> {output_file}")
        
        with open(input_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Only keep relevant columns
        filtered = []
        for r in rows:
            try:
                filtered.append({
                    "files_api_client": r["files_api_client"],
                    "source_type": r["source_type"],
                    "file_size": int(r["file_size"]),
                    "parallel_mode": r["parallel_mode"],
                    "upload_time_s": float(r["upload_time_s"]),
                    "download_time_s": float(r["download_time_s"])
                })
            except (KeyError, ValueError) as e:
                print(f"Skipping row due to error: {e}")
                continue
        
        # Group by key
        grouped = {}
        for r in filtered:
            key = (
                r["files_api_client"],
                r["source_type"],
                r["file_size"],
                r["parallel_mode"]
            )
            grouped.setdefault(key, []).append(r)
        
        # Compute medians and convert file_size to MB
        processed = []
        for key, group in grouped.items():
            upload_times = [g["upload_time_s"] for g in group]
            download_times = [g["download_time_s"] for g in group]
            processed.append({
                "files_api_client": key[0],
                "source_type": key[1],
                "file_size_MB": round(key[2] / (1024 * 1024), 2),
                "parallel_mode": key[3],
                "upload_time_s": round(statistics.median(upload_times), 4),
                "download_time_s": round(statistics.median(download_times), 4)
            })
        
        # Write output
        columns = ["files_api_client", "source_type", "file_size_MB", "parallel_mode", "upload_time_s", "download_time_s"]
        with open(output_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in processed:
                writer.writerow(row)
        
        print(f"Processed {input_file} -> {output_file}")


def sort_and_rearrange_merged_benchmark(input_file, output_file):
    """
    Sort and rearrange columns in merged benchmark file for better readability.
    """
    # Desired column order
    columns = [
        "cloud",
        "presigned_url_type",
        "compute",
        "core_number",
        "files_api_client",
        "source_type",
        "parallel_mode",
        "file_size_MB",
        "upload_time_s",
        "download_time_s"
    ]
    
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Sort by all columns, treating file_size_MB as float
    def sort_key(r):
        key = []
        for c in columns:
            if c == "file_size_MB":
                try:
                    key.append(float(r[c]))
                except Exception:
                    key.append(0.0)
            else:
                key.append(r[c])
        return key
    
    rows.sort(key=sort_key)
    
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    
    print(f"Sorted and rearranged columns in {output_file}")


def comparison_postprocess_benchmarks():
    """
    Merge multiple processed benchmark files from different environments
    into a single comparison file.
    """
    pattern = re.compile(r"processed_benchmark_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([0-9]+)C\.csv")
    input_files = glob.glob("processed_benchmark_*.csv")
    
    if not input_files:
        print("No processed_benchmark_*.csv files found for comparison.")
        return
    
    merged_rows = []
    for input_file in input_files:
        match = pattern.match(os.path.basename(input_file))
        if not match:
            print(f"Skipping file with unexpected format: {input_file}")
            continue
        
        cloud, client, url_type, compute, core_number = match.groups()
        
        with open(input_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["cloud"] = cloud
                row["presigned_url_type"] = url_type
                row["compute"] = compute
                row["core_number"] = core_number
                if row.get("files_api_client") == "FilesExt":
                    row["files_api_client"] = client
                merged_rows.append(row)
    
    if not merged_rows:
        print("No valid processed benchmark files found.")
        return
    
    # Find next available merged file number
    order = 1
    while os.path.exists(f"merged_benchmark_{order}.csv"):
        order += 1
    output_file = f"merged_benchmark_{order}.csv"
    
    # Use all keys from merged_rows[0] for initial write
    columns = list(merged_rows[0].keys())
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)
    
    print(f"Merged {len(input_files)} files into {output_file}")
    
    # Now sort and rearrange columns
    sorted_output_file = output_file.replace(".csv", "_sorted.csv")
    sort_and_rearrange_merged_benchmark(output_file, sorted_output_file)


def summary_comparison(input_file):
    """
    Generate summary comparison files with speedup calculations.
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return
    
    # Keys for grouping and joining
    group_keys = [
        "cloud", "presigned_url_type", "compute", "core_number",
        "files_api_client", "source_type", "parallel_mode", "file_size_MB"
    ]
    join_keys = [
        "cloud", "presigned_url_type", "compute", "core_number", "file_size_MB"
    ]
    
    # Target combinations for columns
    combos = [
        ("FilesAPI", "nonseekable_stream", "sequential"),
        ("FilesExtPrPr", "nonseekable_stream", "sequential"),
        ("FilesExtPuPr", "nonseekable_stream", "sequential"),
        ("FilesExtPuPr", "file_path", "sequential"),
        ("FilesExtPuPr", "file_path", "parallel"),
    ]
    
    # Step 1: Merge rows by group, keeping min upload/download times
    merged = defaultdict(lambda: {"upload_time_s": "", "download_time_s": ""})
    
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row[k] for k in group_keys)
            upload = float(row["upload_time_s"]) if row["upload_time_s"] else None
            download = float(row["download_time_s"]) if row["download_time_s"] else None
            
            if merged[key]["upload_time_s"] == "" or (upload is not None and upload < float(merged[key]["upload_time_s"])):
                merged[key]["upload_time_s"] = str(upload) if upload is not None else ""
            if merged[key]["download_time_s"] == "" or (download is not None and download < float(merged[key]["download_time_s"])):
                merged[key]["download_time_s"] = str(download) if download is not None else ""
    
    # Step 2: Build join table
    # Map: join_key -> {combo: {upload, download}}
    table = defaultdict(dict)
    for key, vals in merged.items():
        join_key = tuple(key[:4] + (key[7],))  # cloud, presigned_url_type, compute, core_number, file_size_MB
        combo = tuple(key[4:7])  # files_api_client, source_type, parallel_mode
        table[join_key][combo] = vals
    
    # Step 3: Write upload and download summary files
    def col_name(a, b, c, typ):
        return f"{a}_{b}_{c}_{typ}"
    
    upload_columns = [
        "cloud", "presigned_url_type", "compute", "core_number", "file_size_MB"
    ]
    download_columns = upload_columns.copy()
    
    # Add columns for each combo, baseline first, then each data column followed by its speedup
    for a, b, c in combos:
        upload_columns.append(col_name(a, b, c, "upload_time_s"))
        if (a, b, c) != combos[0]:
            upload_columns.append(col_name(a, b, c, "upload_speedup"))
    
    for a, b, c in combos:
        download_columns.append(col_name(a, b, c, "download_time_s"))
        if (a, b, c) != combos[0]:
            download_columns.append(col_name(a, b, c, "download_speedup"))
    
    upload_summary_file = input_file.replace(".csv", "_upload_summary.csv")
    download_summary_file = input_file.replace(".csv", "_download_summary.csv")
    
    with open(upload_summary_file, "w") as uf, \
         open(download_summary_file, "w") as df:
        
        uwriter = csv.DictWriter(uf, fieldnames=upload_columns)
        dwriter = csv.DictWriter(df, fieldnames=download_columns)
        uwriter.writeheader()
        dwriter.writeheader()
        
        for join_key, combos_dict in table.items():
            row_base = dict(zip(upload_columns[:5], join_key))
            
            # Baseline
            baseline = combos_dict.get(combos[0], {})
            baseline_upload = float(baseline.get("upload_time_s", "") or 0)
            baseline_download = float(baseline.get("download_time_s", "") or 0)
            
            # Upload row
            upload_row = row_base.copy()
            for i, combo in enumerate(combos):
                val = combos_dict.get(combo, {})
                t = val.get("upload_time_s", "")
                upload_row[col_name(*combo, "upload_time_s")] = t
                
                if combo != combos[0]:
                    speedup = ""
                    try:
                        tval = float(t)
                        if tval > 0 and baseline_upload > 0:
                            speedup = f"{(baseline_upload / tval - 1) * 100:.2f}%"
                    except Exception:
                        pass
                    upload_row[col_name(*combo, "upload_speedup")] = speedup
            
            uwriter.writerow(upload_row)
            
            # Download row
            download_row = row_base.copy()
            for i, combo in enumerate(combos):
                val = combos_dict.get(combo, {})
                t = val.get("download_time_s", "")
                download_row[col_name(*combo, "download_time_s")] = t
                
                if combo != combos[0]:
                    speedup = ""
                    try:
                        tval = float(t)
                        if tval > 0 and baseline_download > 0:
                            speedup = f"{(baseline_download / tval - 1) * 100:.2f}%"
                    except Exception:
                        pass
                    download_row[col_name(*combo, "download_speedup")] = speedup
            
            dwriter.writerow(download_row)
    
    print(f"Upload and download summary files generated:")
    print(f"  {upload_summary_file}")
    print(f"  {download_summary_file}")


def main():
    """
    Main function to handle command line arguments and execute post-processing operations.
    """
    parser = argparse.ArgumentParser(
        description='Post-processing utilities for Databricks Files API benchmarking results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process raw benchmark files
  python3 postprocess.py --postprocess
  
  # Merge and compare processed benchmark files  
  python3 postprocess.py --comparison
  
  # Generate summary with speedup calculations
  python3 postprocess.py --summary merged_benchmark_1_sorted.csv
  
  # Process all steps in sequence
  python3 postprocess.py --postprocess --comparison
        ''')
    
    parser.add_argument('--postprocess', action='store_true',
                       help='Process raw benchmark CSV files into summarized format')
    parser.add_argument('--comparison', action='store_true',
                       help='Merge and compare multiple processed benchmark files')
    parser.add_argument('--summary', type=str, metavar='INPUT_FILE',
                       help='Generate summary report from specified input file')
    
    args = parser.parse_args()
    
    # Check if no arguments provided
    if not any([args.postprocess, args.comparison, args.summary]):
        parser.print_help()
        return
    
    # Execute requested operations
    if args.postprocess:
        print("Running postprocessing on benchmark files...")
        postprocess_benchmarks()
    
    if args.comparison:
        print("Running comparison postprocessing...")
        comparison_postprocess_benchmarks()
    
    if args.summary:
        print(f"Generating summary from {args.summary}...")
        summary_comparison(args.summary)
    
    print("Post-processing completed!")


if __name__ == "__main__":
    main()