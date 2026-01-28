import os.path
from unittest.mock import MagicMock, patch

import pytest
from obs.model import Content
from openjiuwen.core.storage.object.obs_storage_client import OBSClient


def mock_response(status=200, body=None, error_message="error"):
    resp = MagicMock()
    resp.status = status
    resp.requestId = "req-123"

    if status < 300:
        resp.body = body or MagicMock()
    else:
        resp.errorMessage = error_message
        resp.errorCode = "ErrorCode"

    return resp


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("OBS_ACCESS_KEY_ID", "test-ak")
    monkeypatch.setenv("OBS_SECRET_ACCESS_KEY", "test-sk")
    monkeypatch.setenv("OBS_SERVER", "https://obs.r.com")
    monkeypatch.setenv("OBS_BUCKET", "test-bucket")


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_create_bucket_success(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.createBucket.return_value = mock_response()

    client = OBSClient()
    client.create_bucket("bucket", "region")

    obs_instance.createBucket.assert_called_once()


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_create_bucket_failure(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.createBucket.return_value = mock_response(status=400)

    client = OBSClient()
    client.create_bucket("bucket", "region")

    obs_instance.createBucket.assert_called_once()


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_delete_bucket_success(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.deleteBucket.return_value = mock_response()

    client = OBSClient()
    client.delete_bucket("bucket")

    obs_instance.deleteBucket.assert_called_once_with("bucket")


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_delete_bucket_failure(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.deleteBucket.return_value = mock_response(status=500)

    client = OBSClient()
    client.delete_bucket("bucket")


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_upload_file_success(mock_obs_cls, mock_env, tmp_path):
    obs_instance = mock_obs_cls.return_value

    body = MagicMock()
    body.etag = "etag"
    body.versionId = "v1"
    body.storageClass = "STANDARD"

    obs_instance.putFile.return_value = mock_response(body=body)

    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")

    client = OBSClient()
    client.upload_file("bucket", "obj", file_path)

    os.remove(file_path)

    obs_instance.putFile.assert_called_once()


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_upload_file_failure(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.putFile.return_value = mock_response(status=400)

    client = OBSClient()
    client.upload_file("bucket", "obj", "file.txt")


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_download_file_success(mock_obs_cls, mock_env, tmp_path):
    obs_instance = mock_obs_cls.return_value

    stream = MagicMock()
    stream.read.side_effect = [b"data", b"", None]

    body = MagicMock()
    body.response = stream

    obs_instance.getObject.return_value = mock_response(body=body)

    download_path = tmp_path / "test_download.txt"

    client = OBSClient()
    client.download_file("bucket", "obj", download_path)
    assert os.path.isfile(download_path)
    assert os.path.getsize(download_path) > 0

    obs_instance.getObject.assert_called_once()
    stream.read.assert_called()

    os.remove(download_path)


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_download_file_failure(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.getObject.return_value = mock_response(status=404)

    client = OBSClient()
    client.download_file("bucket", "obj", "out.txt")


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_delete_object_success(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.deleteObject.return_value = mock_response()

    client = OBSClient()
    client.delete_object("bucket", "obj")

    obs_instance.deleteObject.assert_called_once_with("bucket", "obj")


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_delete_object_failure(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.deleteObject.return_value = mock_response(status=500)

    client = OBSClient()
    client.delete_object("bucket", "obj")


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_list_objects_success(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value

    content = MagicMock(spec=Content)
    body = MagicMock()
    body.contents = [content]
    body.name = "bucket"
    body.prefix = "pre"
    body.max_keys = 100
    body.is_truncated = False

    obs_instance.listObjects.return_value = mock_response(body=body)

    client = OBSClient()
    result = client.list_objects("bucket", "pre")

    assert result == [content]
    obs_instance.listObjects.assert_called_once()


@patch("openjiuwen.core.storage.object.obs_storage_client.ObsClient")
def test_list_objects_failure(mock_obs_cls, mock_env):
    obs_instance = mock_obs_cls.return_value
    obs_instance.listObjects.return_value = mock_response(status=403)

    client = OBSClient()
    result = client.list_objects("bucket", "pre")

    assert result is None
