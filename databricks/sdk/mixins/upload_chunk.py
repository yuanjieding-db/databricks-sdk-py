# upload_chunk.py

import sys
import json
from databricks.sdk import WorkspaceClient

def main():
    if len(sys.argv) != 7:
        print("Expected 6 arguments: target_path, input_file_path, part_index, chunk_offset, chunk_size, session_token", file=sys.stderr)
        sys.exit(1)

    target_path = sys.argv[1]
    input_file_path = sys.argv[2]
    part_index = int(sys.argv[3])
    chunk_offset = int(sys.argv[4])
    chunk_size = int(sys.argv[5])
    session_token = sys.argv[6]

    w = WorkspaceClient()
    etag = w.files.do_upload_one_chunk(
        target_path=target_path,
        input_file_path=input_file_path,
        part_index=part_index,
        chunk_offset=chunk_offset,
        chunk_size=chunk_size,
        session_token=session_token
    )

    # Return ETag to stdout
    print(json.dumps({"part_index": part_index, "etag": etag}))

if __name__ == "__main__":
    main()
