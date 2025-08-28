

from databricks.sdk import WorkspaceClient
from io import BytesIO

TEST_VOLUME = "/Volumes/users/yuanjie_ding/default"

def write_and_read_test(w: WorkspaceClient):
    text = "Hello world!"
    file = BytesIO(text.encode())
    print(f"Uploading to {TEST_VOLUME}/test.txt")
    w.files.upload(f"{TEST_VOLUME}/test.txt", file, overwrite=True)
    print(list(w.files.list_directory_contents(TEST_VOLUME)))
    resp = w.files.download(f"{TEST_VOLUME}/test.txt")
    print("Downloaded content:")
    print(resp.contents.read().decode())

if __name__ == "__main__":
    w = WorkspaceClient()
    print(f"Using Workspace: {w.config.host}")
    write_and_read_test(w)