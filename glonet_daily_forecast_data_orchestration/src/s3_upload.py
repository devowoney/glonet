import boto3
import os


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
    )


def list_objects(bucket_name: str, prefix_key: str) -> list[dict[str, str]]:
    try:
        response = get_s3_client().list_objects(
            Bucket=bucket_name,
            Prefix=prefix_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(f"Successfully list objects {bucket_name}/{prefix_key}")
            return response["Contents"]
        else:
            raise Exception(response)
    except Exception as exception:
        print(f"Failed to list object {bucket_name}/{prefix_key}: {exception}")
        raise


def delete_object(bucket_name: str, object_key: str):
    try:
        response = get_s3_client().delete_object(
            Bucket=bucket_name,
            Key=object_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] in [200, 204]:
            print(f"Successfully deleted object {bucket_name}/{object_key}")
        else:
            print(
                f"Failed to delete object {bucket_name}/{object_key}: {response}"
            )
    except Exception as exception:
        print(
            f"Failed to delete object {bucket_name}/{object_key}: {exception}"
        )
        raise


def save_bytes_to_s3(bucket_name: str, object_bytes, object_key: str):
    try:
        response = get_s3_client().put_object(
            Bucket=bucket_name,
            Body=object_bytes,
            Key=object_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(
                f"Successfully uploaded bytes to S3 {bucket_name}/{object_key}"
            )
        else:
            raise Exception(response)
    except Exception as exception:
        print(
            f"Failed to upload bytes to S3 {bucket_name}/{object_key}: {exception}"
        )
        raise
