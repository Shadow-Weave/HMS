"""Existence checks must distinguish absence from an unavailable backend."""

from __future__ import annotations

import pytest
from obstore.exceptions import NotFoundError, PermissionDeniedError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("hms_api.engine.storage.s3", "S3FileStorage"),
        ("hms_api.engine.storage.gcs", "GCSFileStorage"),
        ("hms_api.engine.storage.azure", "AzureFileStorage"),
    ],
)
async def test_object_storage_exists_returns_false_only_for_not_found(
    monkeypatch,
    module_name: str,
    class_name: str,
) -> None:
    module = __import__(module_name, fromlist=[class_name])
    storage_class = getattr(module, class_name)
    storage = object.__new__(storage_class)
    storage._store = object()

    async def missing(*_args, **_kwargs):
        raise NotFoundError("synthetic missing object")

    monkeypatch.setattr(module.obs, "head_async", missing)
    assert await storage.exists("opaque/storage/key") is False

    async def unavailable(*_args, **_kwargs):
        raise PermissionDeniedError("synthetic unavailable backend")

    monkeypatch.setattr(module.obs, "head_async", unavailable)
    with pytest.raises(PermissionDeniedError, match="unavailable backend"):
        await storage.exists("opaque/storage/key")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("hms_api.engine.storage.s3", "S3FileStorage"),
        ("hms_api.engine.storage.gcs", "GCSFileStorage"),
        ("hms_api.engine.storage.azure", "AzureFileStorage"),
    ],
)
async def test_object_storage_exists_returns_true_after_successful_head(
    monkeypatch,
    module_name: str,
    class_name: str,
) -> None:
    module = __import__(module_name, fromlist=[class_name])
    storage_class = getattr(module, class_name)
    storage = object.__new__(storage_class)
    storage._store = object()

    async def found(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(module.obs, "head_async", found)
    assert await storage.exists("opaque/storage/key") is True
