from tempfile import mkstemp
from typing import Optional, BinaryIO

from databricks.sdk import WorkspaceClient, FilesAPI
from databricks.sdk.mixins.files import CreateDownloadUrlResponse
from io import BytesIO, RawIOBase, UnsupportedOperation
import random
import requests
import logging
import time

TEST_CONFIGS = {
    "GOOGFOOD": "/Volumes/users/yuanjie_ding/default",
    "AZURE_DOGFOOD": "/Volumes/yuanjie_ding/default/python_sdk_test",
    "DOGFOOD": "/Volumes/main/default/vol1",
}

DATABRICKS_PROFILE = "DOGFOOD"
TEST_VOLUME = TEST_CONFIGS[DATABRICKS_PROFILE]

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self  # allows use of `as`

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

class NonSeekableBuffer(RawIOBase, BinaryIO):
    def __init__(self, data: bytes):
        self._stream = BytesIO(data)

    def read(self, size=-1):
        return self._stream.read(size)

    def readable(self):
        return True

    def seekable(self):
        return False

    def seek(self, *args, **kwargs):
        raise UnsupportedOperation("seek not supported")

    def tell(self):
        raise UnsupportedOperation("tell not supported")

def setup_logging():
    logging.basicConfig(
        filename='multipart-uploads-test.log',
        format='%(asctime)s %(module)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

    # disable unrelated logging in the notebook
    for module in ['pyspark', 'py4j', 'clientserver', 'base_comm']:
        logging.getLogger(module).setLevel(logging.ERROR)


def get_ext_files_api(w: WorkspaceClient):
    w.config.multipart_upload_min_stream_size = 0
    from databricks.sdk.mixins.files import FilesExt
    return FilesExt(w.api_client, w.config)

def dumb_test(w: WorkspaceClient):
    text = "Hello world from Berlin!"
    file = BytesIO(text.encode())
    print(f"Uploading to {TEST_VOLUME}/test.txt")
    w.files.upload(f"{TEST_VOLUME}/test.txt", file, overwrite=True)
    print(list(w.files.list_directory_contents(TEST_VOLUME)))
    resp = w.files.download(f"{TEST_VOLUME}/test.txt")
    print("Downloaded content:")
    print(resp.contents.read().decode())

def get_content(size: int, version: int) -> bytes:
    rnd = random.Random(version)
    return bytes(rnd.getrandbits(8) for _ in range(size))

def multipart_upload(w: WorkspaceClient):
    file_path = f"{TEST_VOLUME}/test_single_multipart.txt"
    files_api = get_ext_files_api(w)
    content_size = 10 * 1024 * 1024  # 10 MB
    content = BytesIO(get_content(content_size, 0))
    with Timer() as t:
        files_api.upload(file_path, content, overwrite=True)
    print(f"Multipart upload took {t.interval:.2f} seconds for {content_size / (1024 * 1024):.2f} MB")
    result_content = files_api.download(file_path).contents.read()

    assert len(result_content) == content_size, f"Expected {content_size} bytes, got {len(result_content)} bytes"
    assert result_content == get_content(content_size, 0), "Content mismatch after upload"
    print(f"Successfully uploaded and verified {file_path} with size {content_size} bytes.")

def single_and_multipart_upload(w: WorkspaceClient):
    single_part_files_api = w.files
    multipart_files_api = get_ext_files_api(w)

    file_path = f"{TEST_VOLUME}/test_single_multipart.txt"
    content_size = 10 * 1024 * 1024  # 10 MB
    content_bytes = get_content(content_size, 0)
    content = BytesIO(content_bytes)

    # Single part upload
    single_part_files_api.upload(file_path, content, overwrite=True)
    downloaded_content_single = single_part_files_api.download(file_path).contents.read()
    assert downloaded_content_single == content_bytes, "Single part upload content mismatch"
    print("Single part upload test passed successfully.")

    # Multipart upload
    content = BytesIO(get_content(content_size, 1))
    multipart_files_api.upload(file_path, content, overwrite=True)
    downloaded_content_multipart = multipart_files_api.download(file_path).contents.read()
    assert downloaded_content_multipart == get_content(content_size, 1), "Multipart upload content mismatch"
    print("Multi part upload test passed successfully.")

def new_download_interface(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/test_download.txt"
    content = BytesIO(b"Test content for download interface hahaha")
    files_api.upload(file_path, content, overwrite=True)

    # Download the file using the old interface
    downloaded_content = files_api.download(file_path).contents.read()
    assert downloaded_content == b"Test content for download interface hahaha", "Downloaded content does not match uploaded content"
    print("Download test passed successfully.")

    # Download the file using the new interface 1
    local_file_path = "/tmp/test_download_new_interface.txt"
    files_api.download_to(file_path, destination=local_file_path)
    with open(local_file_path, 'rb') as f:
        downloaded_content_new = f.read()
    assert downloaded_content_new == downloaded_content, "Downloaded content does not match uploaded content"
    print("New download interface test passed successfully.")

def range_download(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/test_range_download.txt"
    content_size = 5 * 1024 * 1024  # 10 MB
    content = BytesIO(get_content(content_size, 1))
    # files_api.upload(file_path, content, overwrite=True)

    head_response = files_api._head_download(file_path)
    print(f"Head response: {head_response.as_dict()}")

    # Download a specific range of bytes
    start_byte = 1024 * 1024  # 1 MB
    end_byte = 2 * 1024 * 1024 - 1  # Up to 2 MB
    resp = files_api._open_download_stream(file_path, start_byte_offset=start_byte, end_byte_offset=end_byte, if_unmodified_since_timestamp=head_response.last_modified)
    print(resp.as_dict())
    downloaded_content_range = resp.contents.read()

    assert len(downloaded_content_range) == (end_byte - start_byte + 1), "Downloaded range size mismatch"
    assert downloaded_content_range == get_content(content_size, 1)[start_byte:end_byte + 1], "Downloaded content does not match expected range"
    print("Range download test passed successfully.")

def parallel_download(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/test_parallel_download.txt"
    content_size = 5 * 1024 * 1024
    content = get_content(content_size, 2)

    # print(f"Uploading file for parallel download test with size {content_size/1024/1024} MB")
    files_api.upload(file_path, BytesIO(content), overwrite=True)
    print(f"File uploaded to {file_path}")

    # Download the file using the new interface with parallel download
    local_file_path = "/tmp/test_parallel_download.txt"
    files_api.download_to(file_path, destination=local_file_path)
    with open(local_file_path, 'rb') as f:
        downloaded_content_parallel = f.read()

    assert downloaded_content_parallel == content, "Downloaded content does not match uploaded content"
    print("Parallel download test passed successfully.")

def download_logs(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/multipart-uploads-performance-test.log"
    local_file_path = "./multipart-uploads-performance-test-remote.log"
    files_api.download_to(file_path, destination=local_file_path, overwrite=True)
    print(f"Downloaded logs to {local_file_path}")


def parallel_upload(w: WorkspaceClient, parallel_mode: Optional[str] = None):
    print(f"Using parallel mode: {parallel_mode}")
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/test_parallel_upload.txt"
    local_file_path = "/tmp/test_parallel_upload.txt"
    content_size = 5 * 1024 * 1024
    content = get_content(content_size, 3)

    # Write the content to a local file
    with open(local_file_path, 'wb') as f:
        f.write(content)

    # Upload the file using the new interface with parallel upload
    files_api.upload(file_path, local_file_path, overwrite=True, parallel_mode=parallel_mode)

    # Verify the upload
    downloaded_content = files_api.download(file_path).contents.read()
    assert downloaded_content == content, "Uploaded content does not match expected content"
    print("Parallel upload test passed successfully.")

def download_performance_test(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/test_download_performance.txt"
    content_size = 5 * 1024 * 1024
    content = get_content(content_size, 4)

    print(f"Uploading file for download performance test with size {content_size/1024/1024} MB")
    files_api.upload(file_path, BytesIO(content), overwrite=True)
    print(f"File uploaded to {file_path}")

    # Download E2E the file using the old interface
    with Timer() as t:
        resp = files_api.download(file_path, force_old_client=True)
        if resp.contents is None:
            raise ValueError("Response contents is None")
        downloaded_content = resp.contents.read()
    assert downloaded_content == content, "Downloaded content does not match uploaded content"
    print(f"[E2E old client]Downloaded file in {t.interval:.2f} seconds")

    # Download E2E the file using the new interface
    with Timer() as t:
        resp = files_api.download(file_path)
        if resp.contents is None:
            raise ValueError("Response contents is None")
        downloaded_content = resp.contents.read()
    assert downloaded_content == content, "Downloaded content does not match uploaded content"
    print(f"[E2E new client]Downloaded file in {t.interval:.2f} seconds")

    # Get the presigned URL and download the file using requests
    with Timer() as t:
        raw_response = files_api._api.do(
            "POST",
            f"/api/2.0/fs/create-download-url",
            query={
                "path": file_path,
                "expire_time": files_api._get_download_url_expire_time(),
            },
        )
        url_and_headers = CreateDownloadUrlResponse.from_dict(raw_response)
    print(f"Got presigned URL in {t.interval:.2f} seconds")
    if url_and_headers.url is None:
        raise ValueError("Presigned URL is None")
    print(f"Presigned URL: {url_and_headers.url}")
    with Timer() as t:
        response = requests.get(url_and_headers.url, headers=url_and_headers.headers)
        response.raise_for_status()
        downloaded_content = response.content
    print(f"[Presigned URL]Downloaded file in {t.interval:.2f} seconds")
    assert downloaded_content == content, "Downloaded content does not match uploaded content"

    print("Download performance test passed successfully.")

def new_upload_interface(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    local_file_path = "/tmp/test_new_interface.txt"
    file_path = f"{TEST_VOLUME}/test_new_interface.txt"
    content_string = "This is a test content for the new upload interface."

    # Write the content to a local file
    with open(local_file_path, 'w') as f:
        f.write(content_string)

    # Upload the file using the new interface
    files_api.upload(file_path, local_file_path, overwrite=True)
    # Verify the upload
    downloaded_content = files_api.download(file_path).contents.read()
    assert downloaded_content.decode() == content_string, "Uploaded content does not match expected content"
    print("New upload interface test passed successfully.")

def download_with_presigned_url(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/test_presigned_download.txt"
    content_size = 5 * 1024 * 1024
    content = BytesIO(get_content(content_size, 1))
    files_api.upload(file_path, content, overwrite=True)

    # Get a presigned URL for downloading the file
    download_resp = files_api.download(file_path)
    print(f"content length: {download_resp.content_length} ({download_resp.content_length.__class__})")
    assert download_resp.content_length == content_size, "Downloaded content does not match expected content"
    assert download_resp.contents.read() == content, "Downloaded content does not match uploaded content"

ENV_NAME = 'DATABRICKS_ENABLE_EXPERIMENTAL_FILES_API_CLIENT'



if __name__ == "__main__":
    setup_logging()
    # Create a WorkspaceClient instance

    import os
    os.environ[ENV_NAME] = "true"

    print(f"Using profile: {DATABRICKS_PROFILE}")
    w = WorkspaceClient(profile=DATABRICKS_PROFILE)
    print(f"Using Workspace: {w.config.host}")
    # dumb_test(w)

    # new_upload_interface(w)
    # multipart_upload(w)
    # new_download_interface(w)
    # parallel_download(w)
    # range_download(w)
    # parallel_upload(w, parallel_mode="subprocess")
    # single_and_multipart_upload(w)
    # download_with_presigned_url(w)
    # download_logs(w)
    download_performance_test(w)