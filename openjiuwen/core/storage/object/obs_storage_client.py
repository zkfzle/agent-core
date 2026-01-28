import os
import json
from pathlib import Path

from obs import CreateBucketHeader, HeadPermission
from obs import GetObjectRequest
from obs import ObsClient
from obs import PutObjectHeader
from obs.model import Content

from openjiuwen.core.common.logging import logger
from openjiuwen.core.storage.object.base_storage_client import BaseObjectStorageClient


class OBSClient(BaseObjectStorageClient):
    """
    Huawei OBS (Object Storage Service) client implementation.

    This class provides basic bucket and object operations such as
    creating buckets, uploading/downloading files, listing objects,
    and deleting objects using Huawei Cloud OBS.

    The Implementation follows instructions in https://support.huaweicloud.com/intl/en-us/qs-obs/obs_qs_0013.html
    """

    def __init__(
        self,
        server: str = None,
        access_key_id: str = None,
        secret_access_key: str = None,
    ):
        """
        Initialize the OBS client using credentials and configuration
        from environment variables.

        If no arguments provided ,the following environment variables are expected:
        - OBS_ACCESS_KEY_ID: Huawei Cloud access key
        - OBS_SECRET_ACCESS_KEY: Huawei Cloud secret key
        - OBS_SERVER: OBS endpoint URL
        - OBS_BUCKET: Default bucket name
        """
        access_key_id = access_key_id or os.getenv("OBS_ACCESS_KEY_ID")
        secret_access_key = secret_access_key or os.getenv("OBS_SECRET_ACCESS_KEY")
        server = server or os.getenv("OBS_SERVER")
        self.obs_client = ObsClient(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            server=server,
        )

    def create_bucket(self, bucket_name: str, location: str):
        header = CreateBucketHeader(
            aclControl=HeadPermission.PRIVATE,
            storageClass="STANDARD",
            availableZone="3az",
        )
        resp = self.obs_client.createBucket(bucket_name, header, location)
        if resp.status < 300:
            logger.info(
                f'🎉 Bucket "{bucket_name}" at "{location}" location created successfully: {resp.requestId}'
            )
        else:
            logger.error(
                f'❌ Create Bucket "{bucket_name}" at "{location}" location failed: '
                f'{resp.errorCode=} {resp.errorMessage=}'
            )

    def delete_bucket(self, bucket_name: str):
        resp = self.obs_client.deleteBucket(bucket_name)
        if resp.status < 300:
            logger.info(f'🎉 Bucket "{bucket_name}" deleted successfully')
        else:
            logger.error(
                f'❌ Delete Bucket "{bucket_name}" failed: {resp.errorCode=} {resp.errorMessage=}'
            )

    def upload_file(self, bucket_name: str, object_name: str, file_path: str | Path):
        # Specify the additional headers for object upload.
        headers = PutObjectHeader()
        #  Upload the file.
        resp = self.obs_client.putFile(bucket_name, object_name, file_path, headers)
        # If status code 2xx is returned, the API call succeeds. Otherwise, the API call fails.
        if resp.status < 300:
            logger.info(
                f'🎉 Upload "{object_name}" file "{file_path}" to bucket "{bucket_name}" succeeded'
            )
        else:
            logger.error(
                f'❌ Upload "{object_name}" file "{file_path}" to bucket "{bucket_name}" failed: '
                f"{resp.errorCode=} {resp.errorMessage=}"
            )

    def download_file(self, bucket_name: str, object_name: str, file_path: str | Path):
        # Specify the additional parameters for object download.
        get_object_request = GetObjectRequest()
        # Rewrite the Content-Type header in the response.
        get_object_request.content_type = "text/plain"
        # Download the object using streaming.
        resp = self.obs_client.getObject(
            bucketName=bucket_name,
            objectKey=object_name,
            getObjectRequest=get_object_request,
            loadStreamInMemory=False,
        )
        # If status code 2xx is returned, the API call succeeds. Otherwise, the API call fails.
        if resp.status < 300:
            logger.info(
                f'🎉 Get Object "{object_name}" in bucket "{bucket_name}" succeeded'
            )
            # Read the object content.
            with open(file_path, "wb") as f:
                while True:
                    chunk = resp.body.response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)

                resp.body.response.close()

            logger.info(
                f'🎉 Get object "{object_name}" successfully saved to "{file_path}"'
            )
        else:
            logger.error(
                f'❌ Get object "{object_name}" in bucket "{bucket_name}" failed: '
                f"{resp.errorCode=}, {resp.errorMessage=}"
            )

    def delete_object(self, bucket_name: str, object_name: str):
        # Deletes objects in the bucket.
        resp = self.obs_client.deleteObject(bucket_name, object_name)

        if resp.status < 300:
            logger.info(
                f'🎉 Delete file "{object_name}" in bucket "{bucket_name}" succeeded'
            )
        else:
            logger.error(
                f'❌ Delete file "{object_name}" in bucket "{bucket_name}" failed: '
                f"{resp.errorCode=}, {resp.errorMessage=}"
            )

    def list_objects(
        self,
        bucket_name: str,
        object_prefix: str,
        max_objects: int = 100,
    ) -> list[Content] | None:
        resp = self.obs_client.listObjects(
            bucket_name, object_prefix, max_keys=max_objects, encoding_type="url"
        )

        # If status code 2xx is returned, the API call succeeds. Otherwise, the API call fails.
        if resp.status < 300:
            for content in resp.body.contents:
                logger.info(json.dumps(repr(content), indent=2))
            logger.info(
                f'🎉 Successfully listed {len(resp.body.contents)} objects in "{bucket_name}".'
            )

            return resp.body.contents

        logger.error(
            f'❌ List objects in "{bucket_name}" with prefix "{object_prefix}" failed: '
            f"{resp.errorCode=}, {resp.errorMessage=}"
        )
        return None
