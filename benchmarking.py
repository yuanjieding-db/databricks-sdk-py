# This script uses Python SDK to upload and download files of different sizes to a Volume via
# (a) one-shot upload (existing functionality) and
# (b) multipart upload (experimental functionality activated using
# "DATABRICKS_ENABLE_EXPERIMENTAL_FILES_API_CLIENT" env variable).

# Setup on a clean VM (e.g. test shard pod or devbox):
# apt install python3.9 python3.9-distutils
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# /usr/bin/python3.9 get-pip.py
# /usr/bin/python3.9 -m pip install --upgrade databricks-sdk
# /usr/bin/python3.9 filesystem/loadtest/multipart-uploads/run.py

# Before running the script, you need:
# 1) initialize databricks client to connect to Databricks workspace (needs to be UC-enabled).
# You can do it e.g. via Databricks CLI ("databricks configure" command).
# Script will connect with workspace from default profile in ~/.databrickscfg.
# 2) prepare UC volume to upload files to.

# in Databricks Notebook, you may need to run:
# %pip install --upgrade databricks-sdk
# dbutils.library.restartPython()

import argparse
import csv
import datetime
import hashlib
import os
import shutil
import traceback
from databricks.sdk import WorkspaceClient
from typing import Optional, Tuple, BinaryIO
from io import RawIOBase, BytesIO, UnsupportedOperation
from requests import Session, Request, PreparedRequest
from typing import Callable
import json

import logging
import time
import glob
import statistics


TEST_CONFIGS = {
    "GOOGFOOD": "/Volumes/users/yuanjie_ding/default",
    "AZURE_DOGFOOD": "/Volumes/yuanjie_ding/default/python_sdk_test",
    "LM": "/Volumes/users/yuanjie_ding/yuanjie_test",
}

DATABRICKS_PROFILE = "AZURE_DOGFOOD"
TEST_VOLUME = TEST_CONFIGS[DATABRICKS_PROFILE]

# Global variable to control checkpoint mechanism
ENABLE_CHECKPOINT = True
CHECKPOINT_FILE = "benchmark_checkpoint.json"

class NonSeekableBuffer(RawIOBase, BinaryIO):
    """
    A non-seekable buffer that wraps a bytes object. Used for unit tests only.
    This class implements the BinaryIO interface but does not support seeking.
    It is used to simulate a non-seekable stream for testing purposes.
    """

    def __init__(self, data: Tuple[bytes, BytesIO]):
        if isinstance(data, bytes):
            self._stream = BytesIO(data)
        else:
            self._stream = data

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def readline(self, size: int = -1) -> bytes:
        return self._stream.readline(size)

    def readlines(self, size: int = -1) -> list[bytes]:
        return self._stream.readlines(size)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def seek(self, *args, **kwargs) -> int:
        raise UnsupportedOperation("seek not supported")

    def tell(self) -> int:
        raise UnsupportedOperation("tell not supported")


def setup_logging():
    logging.basicConfig(
        filename='multipart-uploads-performance-test.log',
        format='%(asctime)s %(module)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

    # disable unrelated logging in the notebook
    for module in ['pyspark', 'py4j', 'clientserver', 'base_comm']:
        logging.getLogger(module).setLevel(logging.ERROR)


def log(s: str):
    # print(s)
    logging.info(s)


# utility method to calc file size and checksum
def size_and_checksum(file_path: str) -> Tuple[int, str]:
    size = os.stat(file_path).st_size

    sha256 = hashlib.sha256()
    buffer_size = 1024 * 1024
    with open(file_path, "rb") as f:
        while True:
            buffer = f.read(buffer_size)
            if len(buffer) == 0:
                break
            sha256.update(buffer)

    return size, sha256.hexdigest()


def generate_random_file(target_path: str, size: int):
    log(f"Generating local file {target_path} of {size} bytes")
    buffer_size = 1024 * 1024
    current_size = 0
    with open(target_path, "wb") as file:
        while current_size < size:
            buffer = os.urandom(buffer_size)
            file.write(buffer)
            current_size += buffer_size
    log(f"Completed generating local file {target_path} of {size} bytes")


def instrument_session(session: Session, hook: Callable[[str, str, int], None]):
    def pre_hook(request: PreparedRequest):
        request.start_time = time.time()
        return request

    original_prepare = session.prepare_request
    session.prepare_request = lambda request: pre_hook(original_prepare(request))

    def response_hook(response, *args, **kwargs):
        request_time = response.request.start_time
        elapsed_sec = time.time() - request_time
        hook(response.request.method,response.request.url, elapsed_sec)

    session.hooks['response'] = [response_hook]
    return session

def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return None

def checkpoint_matches(config, checkpoint):
    # Compare config lists for equality
    for key in ["file_sizes", "source_types", "parallel_modes", "client_types"]:
        if config[key] != checkpoint.get(key):
            return False
    return True

def single_run(
        w: WorkspaceClient,
        csv_writer,
        volume_path: str,
        source_type: str,
        parallel_mode: str,
        source_local_path: str,
        parallelism: Optional[int] = None,
        part_size: Optional[int] = None,
        file_size: Optional[int] = None,
        cleanup_cloud_file: bool = False,
        target_file_suffix: Optional[str] = None):

    local_path_copy = f"{source_local_path}-copy"

    target_remote_path = f"{volume_path}/file-{file_size}{target_file_suffix or ''}.txt"

    upload_start_time = time.time()

    create_upload_parts_count = 0
    create_upload_parts_total_time_s = 0
    original_create_cloud_provider_session = None

    def files_api_hook(method: str, url: str, elapsed_s: int):
        nonlocal create_upload_parts_count
        nonlocal create_upload_parts_total_time_s
        if method == "POST" and "create-upload-part-urls" in url:
            create_upload_parts_count += 1
            create_upload_parts_total_time_s += elapsed_s

    instrument_session(w.files._api._api_client._session, files_api_hook)

    # let's measure how many parts we uploaded and how long did it take
    part_upload_count = 0
    part_upload_total_time_s = 0
    def part_upload_hook(method: str, _: str, elapsed_s: int):
        nonlocal part_upload_count
        nonlocal part_upload_total_time_s
        if method == "PUT":
            part_upload_count += 1
            part_upload_total_time_s += elapsed_s

    is_files_ext = files_api_class(w) == "FilesExt"
    if is_files_ext:
        original_create_cloud_provider_session = w.files._create_cloud_provider_session
        w.files._create_cloud_provider_session = lambda: instrument_session(original_create_cloud_provider_session(), part_upload_hook)
        log(f"Effective multipart upload part size: {w.files._config.multipart_upload_default_part_size} bytes")

    try:
        # upload file
        if not is_files_ext and file_size > 5 * 1024 * 1024 * 1024:
            log(f"Skipping upload with FilesAPI as file size {file_size} bytes is > 5GB which is not supported")
            return # skip upload with FilesAPI if file size is > 5GB as it is not supported
        if is_files_ext and source_type == "file_path":
            w.files.upload_from(
                target_remote_path,
                source_local_path,
                overwrite=True,
                part_size=part_size,
                use_parallel=(parallel_mode == "parallel"),
                parallelism=parallelism if parallel_mode == "parallel" else None
            )
        elif source_type == "nonseekable_stream":
            if parallel_mode == "sequential":
                with open(source_local_path, "rb") as input_stream:
                    w.files.upload(target_remote_path, NonSeekableBuffer(input_stream), overwrite=True)
            else:
                log("Skipping nonseekable stream upload in parallel mode")
                return  # nonseekable stream upload is only supported in sequential mode
        else:
            log("Skipping upload as source type is file_path but client is not FilesExt")
            # only FilesExt supports upload from file
            return

        upload_complete_time = time.time()
        log(f"Upload of file of {file_size} bytes succeeded in {int(upload_complete_time - upload_start_time)} s")

        # download file
        if is_files_ext and source_type == "file_path":
            w.files.download_to(
                target_remote_path,
                local_path_copy,
                overwrite=True,
                use_parallel=(parallel_mode == "parallel"),
                parallelism=parallelism if parallel_mode == "parallel" else None
            )
        else:
            download_response = w.files.download(target_remote_path)
            contents = getattr(download_response, "contents", None)
            if contents is None:
                raise Exception("Download response does not contain file contents.")
            with open(local_path_copy, "wb") as file:
                shutil.copyfileobj(contents, file)
        

        download_complete_time = time.time()

        size1, checksum1 = size_and_checksum(source_local_path)
        size2, checksum2 = size_and_checksum(local_path_copy)

        log(f"Download of file of {size2} bytes succeeded in {int(download_complete_time - upload_complete_time)} s, checksum: {checksum2}")

        if size1 != size2:
            raise Exception(f"File size mismatch: expected {size1}, observed {size2}")
        if checksum1 != checksum2:
            raise Exception(f"File checksum mismatch: expected {checksum1}, observed {checksum2}")

        upload_duration_s = upload_complete_time - upload_start_time
        download_duration_s = download_complete_time - upload_complete_time
        values = [
            files_api_class(w),
            source_type,
            file_size,
            parallel_mode,
            upload_duration_s,
            create_upload_parts_count,
            create_upload_parts_total_time_s,
            part_upload_count,
            part_upload_total_time_s,
            download_duration_s
        ]
        csv_writer.writerow(values)

    finally:
        if is_files_ext:
            if original_create_cloud_provider_session is not None:
                w.files._create_cloud_provider_session = original_create_cloud_provider_session
        if cleanup_cloud_file:
            try:
                w.files.delete(target_remote_path)
            except Exception:
                pass
        try:
            os.remove(local_path_copy)
        except OSError:
            pass


def run_series(
        w: WorkspaceClient,
        counter_pbar,
        pbar,
        csv_writer,
        runs_count: int,
        source_type: str,
        parallel_mode: str,
        file_size: int,
        volume_path: str,
        part_size: Optional[int] = None,
):
    source_local_path = f"/tmp/file-{file_size}-{int(time.time())}.txt"
    generate_random_file(source_local_path, file_size)
    try:
        for run_id in range(runs_count):
            log(f"#### Run {run_id} ####")
            try:
                single_run(
                    w,
                    csv_writer=csv_writer,
                    volume_path=volume_path,
                    source_type=source_type,
                    parallel_mode=parallel_mode,
                    part_size=part_size,
                    file_size=file_size,
                    source_local_path=source_local_path,
                    cleanup_cloud_file=True,
                    target_file_suffix=f"-run-{str(run_id)}"
                )
            except BaseException as e:
                print(f"Run failed: {e}")
                traceback.print_exc()
            finally:
                counter_pbar.update(1)
                pbar.update(file_size)
    finally:
        os.remove(source_local_path)


ENV_NAME = 'DATABRICKS_ENABLE_EXPERIMENTAL_FILES_API_CLIENT'


def files_api_class(w: WorkspaceClient) -> str:
    return str(w.files.__class__.__name__)

def get_parallelism_params(max_parallelism: int) -> list:
    # numbers = []
    # n = 1
    # while n <= max_parallelism:
    #     numbers.append(n)
    #     n *= 2
    numbers = [1]

    if numbers[-1] < max_parallelism:
        numbers.append(max_parallelism)
    return numbers

def get_workspace_client(enable_new_client: bool, running_in_notebook) -> WorkspaceClient:

    if enable_new_client:
        os.environ[ENV_NAME] = 'True'
        expected_files_api_class = 'FilesExt'
    else:
        if os.environ.get(ENV_NAME):
            os.environ.pop(ENV_NAME)
        expected_files_api_class = 'FilesAPI'
    if running_in_notebook:
        w = WorkspaceClient()
    else:
        w = WorkspaceClient(profile=DATABRICKS_PROFILE)
    if files_api_class(w) != expected_files_api_class:
        raise Exception(
            f"Expected required client to be {expected_files_api_class} but it was {w.files.__class__.__name__}. Make sure to update Python SDK.")
    return w

def postprocess_benchmarks():
    input_files = glob.glob("benchmark_*.csv")
    for input_file in input_files:
        output_file = input_file.replace("benchmark_", "processed_benchmark_")
        with open(input_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Only keep relevant columns
        filtered = [
            {
                "files_api_client": r["files_api_client"],
                "source_type": r["source_type"],
                "file_size": int(r["file_size"]),
                "parallel_mode": r["parallel_mode"],
                "upload_time_s": float(r["upload_time_s"]),
                "download_time_s": float(r["download_time_s"])
            }
            for r in rows
        ]
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
    import csv
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
    import re
    import glob
    import csv
    import os

    pattern = re.compile(r"processed_benchmark_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([0-9]+)C\.csv")
    input_files = glob.glob("processed_benchmark_*.csv")
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
    order = 1
    while os.path.exists(f"merged_benchmark_{order}.csv"):
        order += 1
    output_file = f"merged_benchmark_{order}.csv"
    if merged_rows:
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
    else:
        print("No valid processed benchmark files found.")

def summary_comparison(input_file):
    import csv
    from collections import defaultdict

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
        join_key = tuple(key[:4] + (key[7],)) # cloud, presigned_url_type, compute, core_number, file_size_MB
        combo = tuple(key[4:7]) # files_api_client, source_type, parallel_mode
        table[join_key][combo] = vals
    # Step 3: Write upload and download summary files
    def col_name(a,b,c,typ):
        return f"{a}_{b}_{c}_{typ}"
    upload_columns = [
        "cloud", "presigned_url_type", "compute", "core_number", "file_size_MB"
    ]
    download_columns = upload_columns.copy()
    # Add columns for each combo, baseline first, then each data column followed by its speedup
    for a,b,c in combos:
        upload_columns.append(col_name(a,b,c,"upload_time_s"))
        if (a,b,c) != combos[0]:
            upload_columns.append(col_name(a,b,c,"upload_speedup"))
    for a,b,c in combos:
        download_columns.append(col_name(a,b,c,"download_time_s"))
        if (a,b,c) != combos[0]:
            download_columns.append(col_name(a,b,c,"download_speedup"))
    with open(input_file.replace(".csv", "_upload_summary.csv"), "w") as uf, \
         open(input_file.replace(".csv", "_download_summary.csv"), "w") as df:
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
                upload_row[col_name(*combo,"upload_time_s")] = t
                if combo != combos[0]:
                    speedup = ""
                    try:
                        tval = float(t)
                        if tval > 0 and baseline_upload > 0:
                            speedup = f"{(baseline_upload / tval - 1) * 100:.2f}%"
                    except Exception:
                        pass
                    upload_row[col_name(*combo,"upload_speedup")] = speedup
            uwriter.writerow(upload_row)
            # Download row
            download_row = row_base.copy()
            for i, combo in enumerate(combos):
                val = combos_dict.get(combo, {})
                t = val.get("download_time_s", "")
                download_row[col_name(*combo,"download_time_s")] = t
                if combo != combos[0]:
                    speedup = ""
                    try:
                        tval = float(t)
                        if tval > 0 and baseline_download > 0:
                            speedup = f"{(baseline_download / tval - 1) * 100:.2f}%"
                    except Exception:
                        pass
                    download_row[col_name(*combo,"download_speedup")] = speedup
            dwriter.writerow(download_row)
    print(f"Upload and download summary files generated for {input_file}")

def main():
    setup_logging()
    DEFAULT_RUNS_COUNT = 3
    running_in_notebook = "DATABRICKS_RUNTIME_VERSION" in os.environ
    print(f"Running in notebook: {running_in_notebook}")
    output_file = f"./benchmark_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if running_in_notebook:
        runs_count = DEFAULT_RUNS_COUNT
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--runs_count', required=False)
        args = parser.parse_args()
        runs_count = int(args.runs_count) if args.runs_count else DEFAULT_RUNS_COUNT

    client_types = [
        # "FilesAPI",
        "FilesExt",
    ]
    parallel_modes = [
        "parallel",
        # "sequential",
    ]
    source_types = [
        # "nonseekable_stream",
        "file_path",
    ]
    file_sizes = [
        # 1 * 1024 * 1024, # 1 MB
        # 10 * 1024 * 1024, # 10 MB
        # 20 * 1024 * 1024, # 20 MB
        # 50 * 1024 * 1024, # 50 MB
        100 * 1024 * 1024, # 100 MB
        # 200 * 1024 * 1024, # 200 MB
        # 500 * 1024 * 1024, # 500 MB
        # 1 * 1024 * 1024 * 1024, # 1 GB
        # 2 * 1024 * 1024 * 1024, # 2 GB
        # 5 * 1024 * 1024 * 1024, # 5 GB
        # 10 * 1024 * 1024 * 1024, # 10 GB
        # 20 * 1024 * 1024 * 1024, # 20 GB
    ]
    config = {
        "file_sizes": file_sizes,
        "source_types": source_types,
        "parallel_modes": parallel_modes,
        "client_types": client_types
    }
    checkpoint = None
    start_indices = {
        "client_type": 0,
        "source_type": 0,
        "file_size": 0,
        "parallel_mode": 0,
        "run_id": 0
    }
    csv_mode = "w"
    csv_filename = output_file
    if ENABLE_CHECKPOINT:
        checkpoint = load_checkpoint()
        if checkpoint and checkpoint_matches(config, checkpoint):
            start_indices = checkpoint["indices"]
            csv_filename = checkpoint.get("csv_filename", output_file)
            csv_mode = "a"
            print(f"Resuming from checkpoint: {start_indices}, CSV: {csv_filename}")
        else:
            print("No matching checkpoint found or configuration changed. Starting fresh.")
    print(f"Will be uploading to {DATABRICKS_PROFILE}, Volume: {TEST_VOLUME}")
    print(f"Will be using client types: {client_types}")
    print(f"Will be using source types: {source_types}")
    print(f"Will be using file sizes (MB): {[s // (1024 * 1024) for s in file_sizes]}")
    print(f"Will be running {runs_count} runs for each file size")
    print(f"Will be running upload with parallel modes: {parallel_modes}")
    columns = [
        "files_api_client",
        "source_type",
        "file_size",
        "parallel_mode",
        "upload_time_s",
        "create_upload_part_urls_count",
        "create_upload_part_urls_total_time_s",
        "part_upload_count",
        "part_upload_total_time_s",
        "download_time_s"
    ]
    from tqdm import tqdm
    runs_per_file_size = len(client_types) * len(source_types) * len(parallel_modes) * runs_count
    total_size = sum(file_sizes) * runs_per_file_size
    counter_pbar = tqdm(total=len(file_sizes) * runs_per_file_size, desc="Runs progress")
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Upload Data progress", position=1, leave=False) as pbar:
        with open(csv_filename, csv_mode) as f:
            csv_writer = csv.writer(f)
            if csv_mode == "w":
                csv_writer.writerow(columns)
            for i_client_type, client_type in enumerate(client_types):
                if i_client_type < start_indices["client_type"]:
                    continue
                w = get_workspace_client(enable_new_client=(client_type == "FilesExt"), running_in_notebook=running_in_notebook)
                for i_source_type, source_type in enumerate(source_types):
                    if i_client_type == start_indices["client_type"] and i_source_type < start_indices["source_type"]:
                        continue
                    for i_file_size, file_size in enumerate(file_sizes):
                        if (i_client_type == start_indices["client_type"] and
                            i_source_type == start_indices["source_type"] and
                            i_file_size < start_indices["file_size"]):
                            continue
                        for i_parallel_mode, parallel_mode in enumerate(parallel_modes):
                            if (i_client_type == start_indices["client_type"] and
                                i_source_type == start_indices["source_type"] and
                                i_file_size == start_indices["file_size"] and
                                i_parallel_mode < start_indices["parallel_mode"]):
                                continue
                            for run_id in range(runs_count):
                                if (i_client_type == start_indices["client_type"] and
                                    i_source_type == start_indices["source_type"] and
                                    i_file_size == start_indices["file_size"] and
                                    i_parallel_mode == start_indices["parallel_mode"] and
                                    run_id < start_indices["run_id"]):
                                    continue
                                run_series(
                                    w=w,
                                    counter_pbar=counter_pbar,
                                    pbar=pbar,
                                    csv_writer=csv_writer,
                                    source_type=source_type,
                                    runs_count=1,
                                    parallel_mode=parallel_mode,
                                    file_size=file_size,
                                    volume_path=TEST_VOLUME)
                                # Save checkpoint after each run
                                if ENABLE_CHECKPOINT:
                                    checkpoint_state = {
                                        "file_sizes": file_sizes,
                                        "source_types": source_types,
                                        "parallel_modes": parallel_modes,
                                        "client_types": client_types,
                                        "indices": {
                                            "client_type": i_client_type,
                                            "source_type": i_source_type,
                                            "file_size": i_file_size,
                                            "parallel_mode": i_parallel_mode,
                                            "run_id": run_id + 1
                                        },
                                        "csv_filename": csv_filename
                                    }
                                    save_checkpoint(checkpoint_state)
                            # Reset run_id for next combination
                            start_indices["run_id"] = 0
                        start_indices["parallel_mode"] = 0
                    start_indices["file_size"] = 0
                start_indices["source_type"] = 0
            # Remove checkpoint file when done
            if ENABLE_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "postprocess":
        postprocess_benchmarks()
    elif len(sys.argv) > 1 and sys.argv[1] == "comparison":
        comparison_postprocess_benchmarks()
    elif len(sys.argv) > 2 and sys.argv[1] == "summary":
        summary_comparison(sys.argv[2])
    else:
        main()
