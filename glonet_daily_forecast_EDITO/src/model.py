from s3_upload import get_s3_client
from pathlib import Path


def synchronize_model_locally(local_dir: str):
    sync_s3_to_local(
        "project-glonet", "glonet_1_4_model/20241112/model/", local_dir
    )


def sync_s3_to_local(bucket_name, remote_prefix, local_dir):
    s3_client = get_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Syncing {bucket_name}/{remote_prefix} in {local_dir}...")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=remote_prefix):
        if "Contents" not in page:
            print(f"No files found in s3://{bucket_name}/{remote_prefix}")
            return

        for obj in page["Contents"]:
            s3_key = obj["Key"]
            local_path = local_dir / s3_key[len(remote_prefix) :]

            local_path.parent.mkdir(parents=True, exist_ok=True)

            if (
                not local_path.exists()
                or obj["LastModified"].timestamp() > local_path.stat().st_mtime
            ):
                s3_client.download_file(bucket_name, s3_key, str(local_path))
    print(f"Files {bucket_name}/{remote_prefix} synced in {local_dir}")
