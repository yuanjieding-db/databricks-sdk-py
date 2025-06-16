
from databricks.sdk import WorkspaceClient, FilesAPI
from io import BytesIO
from typing import BinaryIO
import random

TEST_VOLUME = "/Volumes/yuanjie_ding/default/python_sdk_test"

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
    files_api.upload(file_path, content, overwrite=True)
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
    resp = files_api.download(file_path, destination=local_file_path)
    with open(local_file_path, 'rb') as f:
        downloaded_content_new = f.read()
    assert downloaded_content_new == downloaded_content, "Downloaded content does not match uploaded content"
    print("New download interface test passed successfully.")

    # Download the file using the new interface 2
    local_file_path = "/tmp/test_download_new_interface.txt"
    with open(local_file_path, 'wb') as f:
        resp = files_api.download(file_path, destination=f)
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
    # files_api.upload(file_path, BytesIO(content), overwrite=True)
    # print(f"File uploaded to {file_path}")

    # Download the file using the new interface with parallel download
    local_file_path = "/tmp/test_parallel_download.txt"
    resp = files_api._parallel_download(file_path, destination=local_file_path)
    print(resp)
    with open(local_file_path, 'rb') as f:
        downloaded_content_parallel = f.read()

    assert downloaded_content_parallel == content, "Downloaded content does not match uploaded content"
    print("Parallel download test passed successfully.")


def parallel_upload(w: WorkspaceClient):
    files_api = get_ext_files_api(w)
    file_path = f"{TEST_VOLUME}/test_parallel_upload.txt"
    local_file_path = "/tmp/test_parallel_upload.txt"
    content_size = 5 * 1024 * 1024
    content = get_content(content_size, 3)

    # Write the content to a local file
    with open(local_file_path, 'wb') as f:
        f.write(content)

    # Upload the file using the new interface with parallel upload
    files_api.upload(file_path, local_file_path, overwrite=True, use_parallel=True)

    # Verify the upload
    downloaded_content = files_api.download(file_path).contents.read()
    assert downloaded_content == content, "Uploaded content does not match expected content"
    print("Parallel upload test passed successfully.")

ENV_NAME = 'DATABRICKS_ENABLE_EXPERIMENTAL_FILES_API_CLIENT'

if __name__ == "__main__":
    # Create a WorkspaceClient instance

    # import os
    # os.environ[ENV_NAME] = "true"

    w = WorkspaceClient()
    print(f"Using {w.config.host}")

    # multipart_upload(w)
    # new_download_interface(w)
    # parallel_download(w)
    # range_download(w)
    # parallel_upload(w)
    single_and_multipart_upload(w)