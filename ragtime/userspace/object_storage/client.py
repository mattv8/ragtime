"""Async S3 access for workspace object storage.

The gateway is the sole authority for workspace credentials and bucket bindings.
This module deliberately keeps boto3's blocking work outside the event loop.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any, BinaryIO
from urllib.parse import quote

from fastapi import HTTPException
from starlette.responses import Response, StreamingResponse

from ragtime.userspace.object_storage import control

_ENDPOINT = "http://runtime-s3:9000"
_CHUNK_SIZE = 1024 * 1024
_TRANSFER_LIMIT = asyncio.Semaphore(8)


def _endpoint() -> str:
    return os.environ.get("OBJECT_STORAGE_ENDPOINT", _ENDPOINT).strip() or _ENDPOINT


def _http_error(exc: Exception) -> HTTPException:
    try:
        from botocore.exceptions import ClientError  # type: ignore[import-untyped]

        if isinstance(exc, ClientError):
            error = exc.response.get("Error", {})
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 502))
            detail = str(error.get("Code") or "Object storage request failed")
            if status == 404:
                detail = "Object not found"
            return HTTPException(status_code=status, detail=detail)
    except ImportError:
        pass
    return HTTPException(status_code=503, detail="Object storage is unavailable")


@lru_cache(maxsize=128)
def _client(endpoint: str, region: str, access_key_id: str, secret_access_key: str) -> Any:
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency is installed by the image
        raise HTTPException(status_code=503, detail="Object storage client is unavailable") from exc
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(
            s3={"addressing_style": "path"}, max_pool_connections=8, connect_timeout=3, read_timeout=30, retries={"max_attempts": 3, "mode": "standard"}
        ),
    )


async def _workspace_client(workspace_id: str) -> tuple[Any, dict[str, Any]]:
    config = await control.get_workspace(workspace_id)
    access_key_id = str(config.get("access_key_id") or "")
    secret_access_key = str(config.get("secret_access_key") or "")
    if not access_key_id or not secret_access_key:
        raise HTTPException(status_code=503, detail="Workspace object storage credentials are unavailable")
    return _client(
        _endpoint(),
        str(config.get("region") or "us-east-1"),
        access_key_id,
        secret_access_key,
    ), config


def _assert_bucket(config: dict[str, Any], bucket: str) -> None:
    if not any(str(item.get("name") or "") == bucket for item in config.get("buckets", []) if isinstance(item, dict)):
        raise HTTPException(status_code=404, detail="Object storage bucket not found")


async def list_objects(
    workspace_id: str,
    bucket: str,
    prefix: str = "",
    continuation_token: str | None = None,
    max_keys: int = 100,
) -> dict[str, Any]:
    client, config = await _workspace_client(workspace_id)
    _assert_bucket(config, bucket)
    kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/", "MaxKeys": max(1, min(int(max_keys), 1000))}
    if continuation_token:
        kwargs["ContinuationToken"] = continuation_token
    try:
        async with _TRANSFER_LIMIT:
            return await asyncio.to_thread(client.list_objects_v2, **kwargs)
    except Exception as exc:
        raise _http_error(exc) from exc


async def upload_file(
    workspace_id: str,
    bucket: str,
    key: str,
    fileobj: BinaryIO,
    content_type: str | None = None,
) -> dict[str, Any]:
    client, config = await _workspace_client(workspace_id)
    _assert_bucket(config, bucket)
    extra = {"ContentType": content_type} if content_type else {}
    try:
        from boto3.s3.transfer import TransferConfig  # type: ignore[import-untyped]

        transfer_config = TransferConfig(max_concurrency=2, use_threads=True)
        async with _TRANSFER_LIMIT:
            upload_task = asyncio.create_task(asyncio.to_thread(client.upload_fileobj, fileobj, bucket, key, ExtraArgs=extra or None, Config=transfer_config))
            try:
                await asyncio.shield(upload_task)
            except asyncio.CancelledError:
                # The caller may own a temporary staging file.  Do not let its
                # context close that file while boto3's worker is still reading it.
                await asyncio.shield(upload_task)
                raise
            head = await asyncio.to_thread(client.head_object, Bucket=bucket, Key=key)
        return {
            "workspace_id": workspace_id,
            "bucket_name": bucket,
            "key": key,
            "size_bytes": int(head.get("ContentLength", 0)),
            "content_type": head.get("ContentType") or content_type,
        }
    except Exception as exc:
        raise _http_error(exc) from exc


async def download_response(
    workspace_id: str,
    bucket: str,
    key: str,
    headers: dict[str, str] | None = None,
    filename: str | None = None,
) -> Response:
    client, config = await _workspace_client(workspace_id)
    _assert_bucket(config, bucket)
    request_headers = headers or {}
    request_map = {
        "range": "Range",
        "if-match": "IfMatch",
        "if-none-match": "IfNoneMatch",
        "if-modified-since": "IfModifiedSince",
        "if-unmodified-since": "IfUnmodifiedSince",
    }
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    for source, target in request_map.items():
        if request_headers.get(source):
            kwargs[target] = request_headers[source]
    try:
        result = await asyncio.to_thread(client.get_object, **kwargs)
    except Exception as exc:
        try:
            from botocore.exceptions import ClientError  # type: ignore[import-untyped]

            if isinstance(exc, ClientError) and exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 304:
                return Response(status_code=304, headers={"ETag": str(exc.response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("etag") or "")})
        except ImportError:
            pass
        raise _http_error(exc) from exc
    body = result["Body"]

    async def stream() -> AsyncIterator[bytes]:
        try:
            while chunk := await asyncio.to_thread(body.read, _CHUNK_SIZE):
                yield chunk
        finally:
            await asyncio.to_thread(body.close)

    response_headers = {"ETag": str(result.get("ETag") or "")}
    if result.get("ContentRange"):
        response_headers["Content-Range"] = str(result["ContentRange"])
    if result.get("ContentLength") is not None:
        response_headers["Content-Length"] = str(result["ContentLength"])
    if filename:
        safe_filename = filename.replace("\r", "").replace("\n", "").replace("/", "_").replace("\\", "_")
        ascii_filename = safe_filename.encode("ascii", "ignore").decode() or "object"
        response_headers["Content-Disposition"] = f"attachment; filename=\"{ascii_filename.replace(chr(34), '')}\"; filename*=UTF-8''{quote(safe_filename)}"
    return StreamingResponse(
        stream(),
        status_code=int(result.get("ResponseMetadata", {}).get("HTTPStatusCode", 200)),
        media_type=result.get("ContentType") or mimetypes.guess_type(key)[0] or "application/octet-stream",
        headers=response_headers,
    )


async def delete_object(workspace_id: str, bucket: str, key: str) -> None:
    client, config = await _workspace_client(workspace_id)
    _assert_bucket(config, bucket)
    try:
        await asyncio.to_thread(client.delete_object, Bucket=bucket, Key=key)
    except Exception as exc:
        raise _http_error(exc) from exc


async def rename_object(workspace_id: str, bucket: str, key: str, new_key: str) -> dict[str, Any]:
    """Use the gateway's lock-fenced control operation; SDK rename is intentionally unsupported."""
    result = await control.request(
        "POST",
        f"/v1/workspaces/{workspace_id}/buckets/{bucket}/rename",
        {"key": key, "new_key": new_key},
    )
    return {
        "workspace_id": str(result.get("workspace_id") or workspace_id),
        "bucket_name": str(result.get("bucket_name") or bucket),
        "key": str(result.get("key") or new_key),
        "size_bytes": int(result.get("size_bytes") or 0),
        "content_type": result.get("content_type"),
    }
