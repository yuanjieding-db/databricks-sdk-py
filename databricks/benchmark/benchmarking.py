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
import cProfile
from databricks.sdk import WorkspaceClient
from typing import Optional, Tuple, BinaryIO
from io import RawIOBase, BytesIO, UnsupportedOperation
from requests import Session, PreparedRequest
from typing import Callable
import json
import tempfile
import re

import logging
import time
import importlib.metadata


TEST_CONFIGS = {
    "GOOGFOOD": "/Volumes/users/yuanjie_ding/default",
    "AZURE_DOGFOOD": "/Volumes/yuanjie_ding/default/python_sdk_test",
    "LM": "/Volumes/users/yuanjie_ding/yuanjie_test",
    "DOGFOOD": "/Volumes/main/default/vol1",
}

RUNNING_IN_NOTEBOOK = "DATABRICKS_RUNTIME_VERSION" in os.environ
DATABRICKS_PROFILE = "DOGFOOD"

if RUNNING_IN_NOTEBOOK:
    TEST_VOLUME = "/dbfs/yuanjie_ding/python_sdk_test"
else:
    TEST_VOLUME = TEST_CONFIGS[DATABRICKS_PROFILE]

# Global variable to control checkpoint mechanism
ENABLE_CHECKPOINT = True
CHECKPOINT_FILE = "benchmark_checkpoint.json"
NO_TRACE_MODE = False

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


def setup_logging(running_in_notebook=False):
    mode = 'w' if running_in_notebook else 'a'
    logging.basicConfig(
        filename='multipart-uploads-performance-test.log',
        filemode=mode,
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

def apply_presigned_url_disable_patch(w: WorkspaceClient, disable_operations: list):
    """
    Monkey-patch the FilesExt client to disable presigned URLs for specified operations.
    This wraps the _api.do() method to intercept presigned URL creation requests.
    """
    from databricks.sdk.mixins.files import FallbackToUploadUsingFilesApi, FallbackToDownloadUsingFilesApi
    
    original_do = w.files._api.do
    
    def patched_do(method, path, *args, **kwargs):
        # Intercept upload presigned URL requests
        if "upload" in disable_operations:
            if "create-upload-part-urls" in path or "create-resumable-upload-url" in path:
                raise FallbackToUploadUsingFilesApi(
                    None,
                    f"Presigned URL disabled for upload via --disable-presigned-url"
                )
        
        # Intercept download presigned URL requests
        if "download" in disable_operations:
            if "create-download-url" in path:
                raise FallbackToDownloadUsingFilesApi(
                    f"Presigned URL disabled for download via --disable-presigned-url"
                )
        
        # Call original method for all other requests
        return original_do(method, path, *args, **kwargs)
    
    w.files._api.do = patched_do

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

    with tempfile.NamedTemporaryFile(delete=True) as local_copy_file:
        local_path_copy = local_copy_file.name

    target_remote_path = f"{volume_path}/file-{file_size}{target_file_suffix or ''}.txt"

    upload_start_time = time.time()

    create_upload_parts_count = 0
    create_upload_parts_total_time_s = 0
    complete_upload_total_time_s = 0
    create_download_url_count = 0
    create_download_url_total_time_s = 0.
    download_total_tims_s = 0
    original_create_cloud_provider_session = None

    def files_api_hook(method: str, url: str, elapsed_s: int):
        nonlocal create_upload_parts_count
        nonlocal create_upload_parts_total_time_s
        nonlocal complete_upload_total_time_s
        nonlocal create_download_url_count
        nonlocal create_download_url_total_time_s
        if method == "POST" and "create-upload-part-urls" in url:
            create_upload_parts_count += 1
            create_upload_parts_total_time_s += elapsed_s
        elif method == "POST" and "complete-upload" in url:
            complete_upload_total_time_s += elapsed_s
        elif method == "POST" and "create-download-url" in url:
            create_download_url_count += 1
            create_download_url_total_time_s += elapsed_s

    instrument_session(w.files._api._api_client._session, files_api_hook)

    # let's measure how many parts we uploaded and how long did it take
    part_upload_count = 0
    part_upload_total_time_s = 0
    part_download_count = 0
    part_download_total_time_s = 0
    presigned_url_type = ""
    def part_upload_hook(method: str, url: str, elapsed_s: int):
        nonlocal part_upload_count
        nonlocal part_upload_total_time_s
        nonlocal presigned_url_type
        nonlocal part_download_count
        nonlocal part_download_total_time_s
        if method == "PUT":
            part_upload_count += 1
            part_upload_total_time_s += elapsed_s
            if "storage-proxy.databricks" in url:
                presigned_url_type = "DBURL"
            elif "databricks" in url:
                presigned_url_type = "FilesAPI"
            else:
                presigned_url_type = "CSPURL"
        elif method == "GET":
            part_download_count += 1
            part_download_total_time_s += elapsed_s

    is_files_ext = files_api_class(w) == "FilesExt"
    if is_files_ext:
        original_create_cloud_provider_session = w.files._create_cloud_provider_session
        w.files._create_cloud_provider_session = lambda: instrument_session(original_create_cloud_provider_session(), part_upload_hook)

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
                parallelism=parallelism
            )
        elif source_type == "nonseekable_stream":
            if not is_files_ext:
                if parallel_mode == "parallel":
                    log("Skipping upload as source type is nonseekable_stream but client is not FilesExt and parallel_mode is parallel")
                    return
                with open(source_local_path, "rb") as input_stream:
                    w.files.upload(
                        target_remote_path,
                        NonSeekableBuffer(input_stream),
                        overwrite=True,
                    )
            else:
                with open(source_local_path, "rb") as input_stream:
                    # pr = cProfile.Profile()
                    # pr.enable()
                    w.files.upload(
                        target_remote_path,
                        NonSeekableBuffer(input_stream),
                        overwrite=True,
                        part_size=part_size,
                        use_parallel=(parallel_mode == "parallel"),
                        parallelism=parallelism
                    )
                    # pr.disable()
                    # pr.dump_stats(f"profile-{file_size}{target_file_suffix or ''}.prof")
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
                parallelism=parallelism
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
            presigned_url_type,
            source_type,
            file_size,
            parallel_mode,
            upload_duration_s,
            create_upload_parts_count,
            create_upload_parts_total_time_s,
            part_upload_count,
            part_upload_total_time_s,
            complete_upload_total_time_s,
            download_duration_s,
            create_download_url_count,
            create_download_url_total_time_s,
            part_download_count,
            part_download_total_time_s,
        ]
        if csv_writer is not None:
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
        config,
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
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        source_local_path = temp_file.name
    generate_random_file(source_local_path, file_size)
    try:
        for run_id in range(runs_count):
            log(f"#### Run {run_id} ####")
            try:
                single_run(
                    w,
                    csv_writer,
                    volume_path,
                    source_type,
                    parallel_mode,
                    source_local_path,
                    parallelism=config.get('parallelism', None),
                    part_size=part_size,
                    file_size=file_size,
                    cleanup_cloud_file=True,
                    target_file_suffix=f"-run-{str(run_id)}"
                )
            except BaseException as e:
                print(f"Run failed: {e}")
                traceback.print_exc()
                if csv_writer is None:
                    # For pilot run, propagate error
                    raise
            finally:
                if counter_pbar:
                    counter_pbar.update(1)
                if pbar:
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

    if not enable_new_client:
        raise ValueError("Only FilesExt client is supported in the benchmarking script.")
    if running_in_notebook:
        w = WorkspaceClient()
    else:
        w = WorkspaceClient(profile=DATABRICKS_PROFILE)
    return w

def parse_size_string(size_str):
    """
    Parse size string like '100M', '10G', '1.5GB' into bytes.
    Supports units: B, K/KB, M/MB, G/GB, T/TB (case insensitive).
    """
    if not size_str:
        return None
    
    # Remove whitespace and convert to uppercase
    size_str = size_str.strip().upper()
    
    # Use regex to extract number and unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([BKMGT]B?)$', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Use format like '100M', '1.5GB', '10T'")
    
    number = float(match.group(1))
    unit = match.group(2)
    
    # Define size multipliers
    multipliers = {
        'B': 1,
        'K': 1024, 'KB': 1024,
        'M': 1024**2, 'MB': 1024**2,
        'G': 1024**3, 'GB': 1024**3,
        'T': 1024**4, 'TB': 1024**4,
    }
    
    if unit not in multipliers:
        raise ValueError(f"Unsupported unit: {unit}")
    
    return int(number * multipliers[unit])


def parse_list_arg(arg_value, valid_values=None, arg_name="argument"):
    """
    Parse comma-separated list argument and validate against valid values.
    """
    if not arg_value:
        return []
    
    values = [v.strip() for v in arg_value.split(',') if v.strip()]
    
    if valid_values:
        invalid_values = [v for v in values if v not in valid_values]
        if invalid_values:
            raise ValueError(f"Invalid {arg_name} values: {invalid_values}. Valid values: {valid_values}")
    
    return values


def filter_sizes_by_range(file_sizes, min_size=None, max_size=None):
    """
    Filter file sizes based on min and max size constraints.
    """
    filtered_sizes = []
    for size in file_sizes:
        if min_size is not None and size < min_size:
            continue
        if max_size is not None and size > max_size:
            continue
        filtered_sizes.append(size)
    return filtered_sizes


def get_default_file_sizes():
    """
    Return the default file sizes for benchmarking.
    """
    return [
        1 * 1024 * 1024, # 1 MB
        10 * 1024 * 1024, # 10 MB
        20 * 1024 * 1024, # 20 MB
        50 * 1024 * 1024, # 50 MB
        100 * 1024 * 1024, # 100 MB
        200 * 1024 * 1024, # 200 MB
        500 * 1024 * 1024, # 500 MB
        1 * 1024 * 1024 * 1024, # 1 GB
        2 * 1024 * 1024 * 1024, # 2 GB
        5 * 1024 * 1024 * 1024, # 5 GB
        10 * 1024 * 1024 * 1024, # 10 GB
        20 * 1024 * 1024 * 1024, # 20 GB
    ]


def parse_cli_arguments():
    """
    Parse command line arguments for the benchmarking script.
    Returns None if running in notebook mode, otherwise returns parsed args.
    """
    if RUNNING_IN_NOTEBOOK:
        return None
    
    parser = argparse.ArgumentParser(
        description='Databricks Files API benchmarking script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with file size range 100MB to 10GB
  python3 benchmarking.py --min-size 100M --max-size 10G
  
  # Run with specific clients and parallel mode
  python3 benchmarking.py --client "FilesAPI,FilesExt" --parallel sequential
  
  # Run with specific source type and custom runs
  python3 benchmarking.py --source_type file_path --runs_count 5
  
  # Disable presigned URLs for both upload and download
  python3 benchmarking.py --disable-presigned-url "upload,download"
  
  # Disable presigned URLs for upload only
  python3 benchmarking.py --disable-presigned-url "upload"
  
  # Skip pilot run for faster execution
  python3 benchmarking.py --skip-pilot
  
  # Full example with all options
  python3 benchmarking.py --min-size 100M --max-size 10G --client "FilesExt" --parallel sequential --source_type "file_path" --runs_count 3

Post-processing:
  For post-processing benchmark results, use the separate postprocess.py script:
  python3 postprocess.py --postprocess
  python3 postprocess.py --comparison
  python3 postprocess.py --summary input_file.csv

Valid values:
  client: FilesAPI, FilesExt
  parallel: sequential, parallel
  source_type: nonseekable_stream, file_path
  disable_presigned_url: upload, download
  size units: B, K/KB, M/MB, G/GB, T/TB (case insensitive)
        ''')
    
    # File size arguments
    parser.add_argument('--min-size', type=str, help='Minimum file size (e.g., 100M, 1G)')
    parser.add_argument('--max-size', type=str, help='Maximum file size (e.g., 10G, 1T)')
    
    # Client configuration
    parser.add_argument('--client', type=str, 
                       help='Comma-separated list of client types: FilesAPI, FilesExt')
    parser.add_argument('--parallel', type=str,
                       help='Comma-separated list of parallel modes: sequential, parallel')
    parser.add_argument('--source_type', type=str,
                       help='Comma-separated list of source types: nonseekable_stream, file_path')
    
    # Run configuration
    parser.add_argument('--runs_count', type=int, help='Number of runs per test case')
    
    # Volume configuration
    parser.add_argument('--volume', type=str, help='Volume path to use for testing')
    parser.add_argument('--profile', type=str, help='Databricks profile to use')
    
    # Output configuration
    parser.add_argument('--output', type=str, help='Output CSV file path')
    
    # Checkpoint control
    parser.add_argument('--no-checkpoint', action='store_true',
                       help='Disable checkpoint mechanism')

    parser.add_argument('--enable-cprofile', action='store_true', help='Enable cProfile profiling for uploads')

    # Parallelism argument
    parser.add_argument('--parallelism', type=int, default=None, help='Parallelism for upload and download. If not set, None will be passed.')

    parser.add_argument('--disable-presigned-url', type=str,
                       help='Comma-separated list of operations to disable presigned URLs for: upload, download')

    parser.add_argument('--skip-pilot', action='store_true',
                       help='Skip the pilot run and proceed directly to benchmarking')

    return parser.parse_args()


def validate_and_apply_cli_args(args):
    """
    Validate CLI arguments and return configuration dictionaries.
    """
    config = {}
    
    # Parse size constraints
    min_size = None
    max_size = None
    if args and args.min_size:
        min_size = parse_size_string(args.min_size)
        print(f"Minimum file size: {args.min_size} ({min_size:,} bytes)")
    
    if args and args.max_size:
        max_size = parse_size_string(args.max_size)
        print(f"Maximum file size: {args.max_size} ({max_size:,} bytes)")
    
    if min_size and max_size and min_size > max_size:
        raise ValueError("Minimum size cannot be greater than maximum size")
    
    # Get and filter file sizes
    default_file_sizes = get_default_file_sizes()
    file_sizes = filter_sizes_by_range(default_file_sizes, min_size, max_size)
    
    if not file_sizes:
        print("Warning: No file sizes match the specified range. Using default sizes.")
        file_sizes = default_file_sizes
    
    config['file_sizes'] = file_sizes
    
    # Parse client types
    default_client_types = ["FilesExt"]
    if args and args.client:
        client_types = parse_list_arg(args.client, default_client_types, "client")
    else:
        client_types = default_client_types
    config['client_types'] = client_types
    
    # Parse parallel modes
    default_parallel_modes = ["sequential", "parallel"]
    if args and args.parallel:
        parallel_modes = parse_list_arg(args.parallel, default_parallel_modes, "parallel")
    else:
        parallel_modes = default_parallel_modes
    config['parallel_modes'] = parallel_modes
    
    # Parse source types
    default_source_types = ["nonseekable_stream", "file_path"]
    if args and args.source_type:
        source_types = parse_list_arg(args.source_type, default_source_types, "source_type")
    else:
        source_types = default_source_types
    config['source_types'] = source_types
    
    # Other configurations
    config['runs_count'] = args.runs_count if args and args.runs_count else 3
    config['volume'] = args.volume if args and args.volume else None
    config['profile'] = args.profile if args and args.profile else None
    config['output_file'] = args.output if args and args.output else None
    config['enable_checkpoint'] = not (args and args.no_checkpoint)
    config['enable_cprofile'] = args and args.enable_cprofile
    config['parallelism'] = args.parallelism if args and args.parallelism is not None else None
    config['skip_pilot'] = args and args.skip_pilot
    
    # Parse disable_presigned_url
    default_operations = ["upload", "download"]
    if args and args.disable_presigned_url:
        disable_operations = parse_list_arg(
            args.disable_presigned_url, 
            default_operations, 
            "disable_presigned_url"
        )
    else:
        disable_operations = None
    config['disable_presigned_url'] = disable_operations
    
    return config


def benchmark_sdk(
    min_size=None,
    max_size=None,
    client=None,
    parallel=None,
    source_type=None,
    runs_count=None,
    volume=None,
    profile=None,
    output=None,
    no_checkpoint=False,
    enable_cprofile=False,
    parallelism=None,
    disable_presigned_url=None,
    skip_pilot=False
):
    setup_logging(running_in_notebook=RUNNING_IN_NOTEBOOK)

    # Compose a namespace-like object for argument compatibility
    class Args:
        pass
    args = Args()
    args.min_size = min_size
    args.max_size = max_size
    args.client = client
    args.parallel = parallel
    args.source_type = source_type
    args.runs_count = runs_count
    args.volume = volume
    args.profile = profile
    args.output = output
    args.no_checkpoint = no_checkpoint
    args.enable_cprofile = enable_cprofile
    args.parallelism = parallelism
    args.disable_presigned_url = disable_presigned_url
    args.skip_pilot = skip_pilot

    config = validate_and_apply_cli_args(args)

    global ENABLE_CHECKPOINT, TEST_VOLUME, DATABRICKS_PROFILE
    ENABLE_CHECKPOINT = config['enable_checkpoint']
    if config['volume']:
        TEST_VOLUME = config['volume']
        print(f"Using custom volume: {TEST_VOLUME}")
    if config['profile']:
        DATABRICKS_PROFILE = config['profile']
        print(f"Using custom profile: {DATABRICKS_PROFILE}")
    
    runs_count = config['runs_count']
    client_types = config['client_types']
    parallel_modes = config['parallel_modes']
    source_types = config['source_types']
    file_sizes = config['file_sizes']
    
    print(f"Running in notebook: {RUNNING_IN_NOTEBOOK}")
    
    # Setup output file
    if config['output_file']:
        output_file = config['output_file']
    else:
        output_file = f"./benchmark_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    run_start_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    environment_str = os.environ.get("DB_INSTANCE_TYPE", "")
    client_version = importlib.metadata.version('databricks-sdk')
    
    benchmark_config = {
        "file_sizes": config['file_sizes'],
        "source_types": config['source_types'],
        "parallel_modes": config['parallel_modes'],
        "client_types": config['client_types']
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
        if checkpoint and checkpoint_matches(benchmark_config, checkpoint):
            start_indices = checkpoint["indices"]
            csv_filename = checkpoint.get("csv_filename", output_file)
            csv_mode = "a"
            print(f"Resuming from checkpoint: {start_indices}, CSV: {csv_filename}")
        else:
            print("No matching checkpoint found or configuration changed. Starting fresh.")
    
    print(f"Will be uploading to {DATABRICKS_PROFILE}, Volume: {TEST_VOLUME}")
    print(f"Will be using file sizes (MB): {[s // (1024 * 1024) for s in file_sizes]}")
    print(f"Will be using client types: {client_types}")
    print(f"Will be using source types: {source_types}")
    print(f"Will be running upload with parallel modes: {parallel_modes}")
    print(f"Will be running {runs_count} runs for each file size")
    
    columns = [
        "run_start_timestamp",
        "case_start_timestamp",
        "environment",
        "client_version",
        "files_api_client",
        "presigned_url_type",
        "source_type",
        "file_size",
        "parallel_mode",
        "upload_time_s",
        "create_upload_part_urls_count",
        "create_upload_part_urls_total_time_s",
        "part_upload_count",
        "part_upload_total_time_s",
        "complete_upload_total_time_s",
        "download_time_s",
        "create_download_url_count",
        "create_download_url_total_time_s",
        "part_download_count",
        "part_download_total_time_s",
    ]
    
    # PILOT RUN
    if not config.get('skip_pilot', False):
        print("PILOT running")
        pilot_file_size = file_sizes[0]
        pilot_failed = False
        for client_type in client_types:
            w = get_workspace_client(enable_new_client=(client_type == "FilesExt"), running_in_notebook=RUNNING_IN_NOTEBOOK)
            # Apply presigned URL disable patch if configured
            if config.get('disable_presigned_url'):
                apply_presigned_url_disable_patch(w, config['disable_presigned_url'])
            for source_type in source_types:
                for parallel_mode in parallel_modes:
                    try:
                        print(f"Running pilot for {client_type}, {source_type}, {parallel_mode}")
                        run_series(
                            config,
                            w=w,
                            counter_pbar=None,
                            pbar=None,
                            csv_writer=None,
                            source_type=source_type,
                            runs_count=1,
                            parallel_mode=parallel_mode,
                            file_size=pilot_file_size,
                            volume_path=TEST_VOLUME
                        )
                        print(f"Passed")
                    except Exception as e:
                        print(f"Pilot run failed for {client_type}, {source_type}, {parallel_mode} with size {pilot_file_size}: {e}")
                        pilot_failed = True
        
        if pilot_failed:
            import sys
            print("Pilot run failed. Exiting.")
            sys.exit(1)
    else:
        print("Skipping pilot run as requested")
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
                w = get_workspace_client(enable_new_client=(client_type == "FilesExt"), running_in_notebook=RUNNING_IN_NOTEBOOK)
                # Apply presigned URL disable patch if configured
                if config.get('disable_presigned_url'):
                    apply_presigned_url_disable_patch(w, config['disable_presigned_url'])
                
                for i_source_type, source_type in enumerate(source_types):
                    if i_client_type == start_indices["client_type"] and i_source_type < start_indices["source_type"]:
                        continue
                        
                    for i_file_size, file_size in enumerate(file_sizes):
                        if (i_client_type == start_indices["client_type"] and
                            i_source_type == start_indices["source_type"] and i_file_size < start_indices["file_size"]):
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
                                    
                                case_start_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                
                                # Wrap csv_writer to prepend extra columns
                                class ExtendedWriter:
                                    def __init__(self, writer):
                                        self.writer = writer
                                    def writerow(self, values):
                                        row = [run_start_timestamp, case_start_timestamp, environment_str, client_version] + list(values)
                                        self.writer.writerow(row)
                                
                                ext_writer = ExtendedWriter(csv_writer)
                                run_series(
                                    config,
                                    w=w,
                                    counter_pbar=counter_pbar,
                                    pbar=pbar,
                                    csv_writer=ext_writer,
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
    
    if RUNNING_IN_NOTEBOOK:
        # insert_data(csv_filename)
        os.environ["RESULT_CSV_FILENAME"] = csv_filename

def insert_data(csv_filename):
    import pandas as pd
    from pyspark.sql import SparkSession

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_filename)

    # Convert the Pandas DataFrame to a Spark DataFrame
    spark_df = SparkSession.builder.getOrCreate().createDataFrame(df)

    # Insert the data into the python_sdk_benchmark table
    spark_df.write.insertInto("main.yuanjie_ding.python_sdk_benchmark")

if __name__ == "__main__":
    args = parse_cli_arguments()
    benchmark_sdk(
        min_size=getattr(args, "min_size", None),
        max_size=getattr(args, "max_size", None),
        client=getattr(args, "client", None),
        parallel=getattr(args, "parallel", None),
        source_type=getattr(args, "source_type", None),
        runs_count=getattr(args, "runs_count", None),
        volume=getattr(args, "volume", None),
        profile=getattr(args, "profile", None),
        output=getattr(args, "output", None),
        no_checkpoint=getattr(args, "no_checkpoint", False),
        enable_cprofile=getattr(args, "enable_cprofile", False),
        parallelism=getattr(args, "parallelism", None),
        disable_presigned_url=getattr(args, "disable_presigned_url", None),
        skip_pilot=getattr(args, "skip_pilot", False)
    )
