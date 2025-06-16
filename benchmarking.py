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
from typing import Optional
from requests import Session, Request, PreparedRequest
from typing import Callable

import logging
import time


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
def size_and_checksum(file_path: str) -> [int, str]:
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

def single_run(
        w: WorkspaceClient,
        csv_writer: csv.writer,
        catalog_name: str,
        schema_name: str,
        volume_name: str,
        parallel_mode: str,
        parallelism: int,
        chunk_size: Optional[int],
        file_size: Optional[int] = None,
        source_local_path: Optional[str] = None,
        cleanup_cloud_file: bool = False,
        target_file_suffix: Optional[str] = None):
    if source_local_path is None:
        source_local_path = f"/tmp/file-{file_size}-{int(time.time())}.txt"
        generate_random_file(source_local_path, file_size)
        cleanup_local_file = True
    else:
        file_size = os.stat(source_local_path).st_size
        cleanup_local_file = False

    local_path_copy = f"{source_local_path}-copy"

    target_remote_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/file-{file_size}{target_file_suffix or ''}.txt"

    upload_start_time = time.time()

    # let's measure how many times we requested presigned URLs for parts and how long did it take
    create_upload_parts_count = 0
    create_upload_parts_total_time_s = 0

    def files_api_hook(method: str, url: str, elapsed_s: int):
        nonlocal create_upload_parts_count
        nonlocal create_upload_parts_total_time_s
        if method == "POST" and "create-upload-part-urls" in url:
            create_upload_parts_count += 1
            create_upload_parts_total_time_s += elapsed_s

    instrument_session(w.files._api._api_client._session, files_api_hook)

    # let's measure how many chunks we uploaded and how long did it take
    chunk_upload_count = 0
    chunk_upload_total_time_s = 0
    def chunk_upload_hook(method: str, _: str, elapsed_s: int):
        nonlocal chunk_upload_count
        nonlocal chunk_upload_total_time_s
        if method == "PUT":
            chunk_upload_count += 1
            chunk_upload_total_time_s += elapsed_s

    if files_api_class(w) == "FilesExt":
        original_create_cloud_provider_session = w.files._create_cloud_provider_session
        w.files._create_cloud_provider_session = lambda: instrument_session(original_create_cloud_provider_session(), chunk_upload_hook)

        original_chunk_size = w.files._config.multipart_upload_chunk_size
        if chunk_size:
            w.files._config.multipart_upload_chunk_size = chunk_size

        log(f"Effective multipart upload chunk size: {w.files._config.multipart_upload_chunk_size} bytes")

    try:
        # upload file
        if parallelism is not None and parallelism > 1:
            w.files.upload(target_remote_path, source_local_path, parallel_mode=parallel_mode, parallelism=parallelism, overwrite=True)
        else:
            with open(source_local_path, "rb") as input_stream:
                # this will upload the file in chunks
                w.files.upload(target_remote_path, input_stream, overwrite=True)

        upload_complete_time = time.time()

        log(f"Upload of file of {file_size} bytes succeeded in {int(upload_complete_time - upload_start_time)} s")

        # download file
        download_response = w.files.download(target_remote_path)
        with open(local_path_copy, "wb") as file:
            # this will download file in small chunks, rather than loading all the contents into memory
            shutil.copyfileobj(download_response.contents, file)

        download_complete_time = time.time()

        size1, checksum1 = size_and_checksum(source_local_path)
        size2, checksum2 = size_and_checksum(local_path_copy)

        log(f"Download of file of {size2} bytes succeeded in {int(download_complete_time - upload_complete_time)} s, checksum: {checksum2}")

        # verify resulting copy is identical
        if size1 != size2:
            raise Exception(f"File size mismatch: expected {size1}, observed {size2}")

        if checksum1 != checksum2:
            raise Exception(f"File checksum mismatch: expected {checksum1}, observed {checksum2}")

        upload_duration_s = upload_complete_time - upload_start_time
        download_duration_s = download_complete_time - upload_complete_time

        message = f"Final stats: file size {size1}, upload took {int(upload_duration_s)} s, download took {int(download_duration_s)} s"
        log(message)

        values = [
            files_api_class(w),
            file_size,
            w.files._config.multipart_upload_chunk_size if files_api_class(w) == "FilesExt" else "0",
            parallel_mode,
            parallelism,
            upload_duration_s,
            create_upload_parts_count,
            create_upload_parts_total_time_s,
            chunk_upload_count,
            chunk_upload_total_time_s,
            download_duration_s
        ]
        # print(",".join(map(str, values)))
        csv_writer.writerow(values)

    finally:
        if files_api_class(w) == "FilesExt":
            w.files._create_cloud_provider_session = original_create_cloud_provider_session
            w.files._config.multipart_upload_chunk_size = original_chunk_size

        if cleanup_cloud_file:
            try:
                w.files.delete(target_remote_path)
            except Exception:
                # ignore
                pass

        try:
            os.remove(local_path_copy)
        except OSError:
            pass

        if cleanup_local_file:
            os.remove(source_local_path)


def run_series(
        w: WorkspaceClient,
        counter_pbar,
        pbar,
        csv_writer: csv.writer,
        runs_count: int,
        parallel_mode: str,
        parallelism: int,
        file_size: int,
        chunk_size: Optional[int],
        catalog_name: str,
        schema_name: str,
        volume_name: str
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
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    volume_name=volume_name,
                    parallel_mode=parallel_mode,
                    parallelism=parallelism,
                    chunk_size=chunk_size,
                    file_size=None,
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

def main():
    catalog_name='main'
    schema_name='default'
    volume_name='vol1'

    setup_logging()

    DEFAULT_RUNS_COUNT = 3
    system_cpu_cnt = os.cpu_count() or 1

    running_in_notebook = "DATABRICKS_RUNTIME_VERSION" in os.environ
    print(f"Running in notebook: {running_in_notebook}")
    output_file = f"./benchmark_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    if running_in_notebook:
        enable_new_client = True
        runs_count = DEFAULT_RUNS_COUNT
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--enable_new_client', required=True)
        parser.add_argument('--runs_count', required=False)
        args = parser.parse_args()

        if args.enable_new_client == 'true':
            enable_new_client = True
        elif args.enable_new_client == 'false':
            enable_new_client = False
        else:
            raise Exception(
                f'Unexpected value "{args.enable_new_client}" of "--enable_new_client" argument, use "true" or "false"')

        runs_count = int(args.runs_count) if args.runs_count else DEFAULT_RUNS_COUNT

    if enable_new_client:
        os.environ[ENV_NAME] = 'True'
        w = WorkspaceClient()
        expected_files_api_class = 'FilesExt'
    else:
        if os.environ.get(ENV_NAME):
            os.environ.pop(ENV_NAME)
        w = WorkspaceClient()
        expected_files_api_class = 'FilesAPI'

    if files_api_class(w) != expected_files_api_class:
        raise Exception(
            f"Expected required client to be {expected_files_api_class} but it was {w.files.__class__.__name__}. Make sure to update Python SDK.")

    parallel_params = get_parallelism_params(system_cpu_cnt - 1)

    parallel_modes = [
        "multithreading",
        "multiprocessing",
    ]

    file_sizes = [
        1 * 1024 * 1024,
        # 10 * 1024 * 1024,
        # 100 * 1024 * 1024,
        # 500 * 1024 * 1024,
        # 1024 * 1024 * 1024,
        # 4 * 1024 * 1024 * 1024,
    ]

    multipart_upload_chunk_sizes = [
        None, # uses default chunk size
        100 * 1024 * 1024,
        250 * 1024 * 1024
    ] if enable_new_client else [0]

    print(f"Will be uploading to {w.config.host}, Volume: {catalog_name}/{schema_name}/{volume_name}")
    print(f"Will be running {runs_count} runs for each file size")
    print(f"Will be running with {system_cpu_cnt} system CPUs, parallelism values: {parallel_params}")
    print(f"Will be running upload with parallel modes: {parallel_modes}")

    columns = [
        "files_api_client",
        "file_size",
        "chunk_size",
        "parallel_mode",
        "parallelism",
        "upload_time_s",
        "create_upload_part_urls_count",
        "create_upload_part_urls_total_time_s",
        "chunk_upload_count",
        "chunk_upload_total_time_s",
        "download_time_s"
    ]

    # print(",".join(columns))

    from tqdm import tqdm
    runs_per_file_size = len(multipart_upload_chunk_sizes) * len(parallel_params) * len(parallel_modes) * runs_count
    total_size = sum(file_sizes) * runs_per_file_size
    counter_pbar = tqdm(total=len(file_sizes) * runs_per_file_size, desc="Runs progress")
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Upload Data progress", position=1, leave=False) as pbar:
        with open(output_file, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(columns)
            for file_size in file_sizes:
                for chunk_size in multipart_upload_chunk_sizes:
                    for parallel_mode in parallel_modes:
                        for parallelism in parallel_params:
                            run_series(
                                w=w,
                                counter_pbar=counter_pbar,
                                pbar=pbar,
                                csv_writer=csv_writer,
                                runs_count=runs_count,
                                parallel_mode=parallel_mode,
                                parallelism=parallelism,
                                file_size=file_size,
                                chunk_size=chunk_size,
                                catalog_name=catalog_name,
                                schema_name=schema_name,
                                volume_name=volume_name)
                            f.flush()


if __name__ == "__main__":
    main()
