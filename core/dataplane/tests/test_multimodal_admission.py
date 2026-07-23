"""Bounded multipart admission tests for opt-in multimodal uploads."""

import base64
import hashlib
import io
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

import hms_api.engine.memory_engine as memory_engine_module
from hms_api.api.http import create_app
from hms_api.engine.memory_engine import (
    MemoryEngine,
    _canonical_multimodal_parser_policy,
    _canonical_multimodal_validator_hints,
    _classify_multimodal_media_kind,
    _derive_anonymous_multimodal_document_id,
    _reject_multimodal_transport_payload_in_metadata,
)
from hms_api.engine.multimodal.errors import MediaValidationError
from hms_api.engine.multimodal.images import ImageNormalizationConfig, normalize_image
from hms_api.models import RequestContext


class _Registry:
    def list_parsers(self) -> list[str]:
        return ["openai_multimodal"]


class _Memory:
    audit_logger = None

    def __init__(self) -> None:
        self._parser_registry = _Registry()
        self.submissions: list[dict] = []

    async def submit_async_file_retain(self, **kwargs):
        self.submissions.append(kwargs)
        return {"operation_ids": ["00000000-0000-0000-0000-000000000001"], "files_count": 1}


class _UnreadUpload:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.content_type = "image/png"
        self.read_calls = 0

    async def read(self, _size: int = -1) -> bytes:
        self.read_calls += 1
        return b"must-not-be-read"


class _RejectUnexpectedStorage:
    async def store(self, **_kwargs) -> None:
        raise AssertionError("rejected metadata must not reach source storage")


class _SingleReadUpload(_UnreadUpload):
    def __init__(self, filename: str, data: bytes) -> None:
        super().__init__(filename)
        self._data = data

    async def read(self, _size: int = -1) -> bytes:
        self.read_calls += 1
        data, self._data = self._data, b""
        return data


class _SizedReadUpload(_UnreadUpload):
    def __init__(self, filename: str, data: bytes, content_type: str) -> None:
        super().__init__(filename)
        self._data = data
        self.content_type = content_type
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        self.read_sizes.append(size)
        if not self._data:
            return b""
        if size < 0:
            data, self._data = self._data, b""
            return data
        data, self._data = self._data[:size], self._data[size:]
        return data


class _CapturingStorage:
    def __init__(self) -> None:
        self.stores: list[dict] = []
        self.deleted: list[str] = []

    async def store(self, **kwargs) -> None:
        self.stores.append(kwargs)

    async def delete(self, key: str) -> None:
        self.deleted.append(key)


class _AsyncContext:
    def __init__(self, value=None) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _AdmissionConnection:
    backend_type = "postgresql"

    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple]] = []

    def transaction(self):
        return _AsyncContext(self)

    async def execute(self, sql, *args):
        self.executions.append((sql, args))
        return "INSERT 0 1"


class _FingerprintRegistry:
    @staticmethod
    def get_parser(_name: str, _filename: str, _content_type: str):
        return SimpleNamespace(pipeline_fingerprint=lambda: "f" * 64)


class _CapturingTaskBackend:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def submit_task(self, payload: dict) -> None:
        self.payloads.append(payload)


class _DeduplicatingLedger:
    def __init__(self) -> None:
        self.commands: dict[tuple[str, str, str], SimpleNamespace] = {}
        self.attempted_specs: list = []

    async def admit_document_command(self, _conn, spec, *, now):
        del now
        self.attempted_specs.append(spec)
        identity = (spec.bank_id, spec.document_id, spec.command_key)
        command = self.commands.get(identity)
        if command is not None:
            return SimpleNamespace(created=False, command=command)
        command = SimpleNamespace(
            sequence=len(self.commands) + 1,
            operation_id=spec.operation_id,
        )
        self.commands[identity] = command
        return SimpleNamespace(created=True, command=command)


class _RecordingMetrics:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record_multimodal_pipeline(self, **event) -> None:
        self.events.append(event)

    @contextmanager
    def record_http_request(self, *_args, **_kwargs):
        yield


def _config(base_config, **overrides):
    values = {
        "enable_file_upload_api": True,
        "file_conversion_max_batch_size": 2,
        "file_conversion_max_batch_size_mb": 1,
        "file_parser_allowlist": ["openai_multimodal"],
        "file_parser": ["openai_multimodal"],
        "multimodal_max_image_bytes": 8,
        "multimodal_max_video_bytes": 32,
    }
    values.update(overrides)
    return replace(base_config, **values)


def _multimodal_admission_engine(monkeypatch, *, storage, tasks, ledger) -> MemoryEngine:
    from hms_api.engine.multimodal.ledger import MultimodalLedger

    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = storage
    engine._parser_registry = _FingerprintRegistry()
    engine._task_backend = tasks

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    async def get_backend():
        return SimpleNamespace()

    engine._authenticate_tenant = authenticate
    engine._get_backend = get_backend
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)
    monkeypatch.setattr(
        memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(_AdmissionConnection())
    )
    monkeypatch.setattr(MultimodalLedger, "for_connection", lambda *_args, **_kwargs: ledger)
    return engine


def _file_item(
    data: bytes,
    *,
    document_id: str | None = None,
    filename: str = "screen.png",
    content_type: str = "image/png",
    parser: str | list[str] = "openai_multimodal",
) -> dict:
    upload = _SingleReadUpload(filename, data)
    upload.content_type = content_type
    return {
        "file": upload,
        "document_id": document_id,
        "context": "safe context",
        "metadata": {},
        "tags": ["safe-tag"],
        "timestamp": None,
        "parser": [parser] if isinstance(parser, str) else parser,
        "strategy": None,
    }


def _small_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


def test_anonymous_multimodal_document_id_is_stable_and_scope_isolated() -> None:
    asset_sha256 = hashlib.sha256(b"same anonymous visual asset").hexdigest()
    common = {
        "tenant_scope": "tenant-a",
        "bank_id": "bank-a",
        "asset_sha256": asset_sha256,
    }

    first = _derive_anonymous_multimodal_document_id(**common)
    repeat = _derive_anonymous_multimodal_document_id(**common)
    other_bank = _derive_anonymous_multimodal_document_id(**{**common, "bank_id": "bank-b"})
    other_tenant = _derive_anonymous_multimodal_document_id(**{**common, "tenant_scope": "tenant-b"})

    assert first == repeat
    assert first.startswith("file_mm_")
    assert asset_sha256 not in first
    assert first != other_bank
    assert first != other_tenant


def test_validator_hints_canonicalize_mime_and_only_relevant_extension_family() -> None:
    jpeg = _canonical_multimodal_validator_hints(
        filename="Customer Secret.JPEG",
        content_type=" Image/JPEG ; charset=binary",
    )
    jpeg_alias = _canonical_multimodal_validator_hints(
        filename="renamed.jpg",
        content_type="image/jpeg",
    )
    unknown = _canonical_multimodal_validator_hints(
        filename="first.private-extension",
        content_type=None,
    )
    octet_stream = _canonical_multimodal_validator_hints(
        filename="second.bin",
        content_type="application/octet-stream; charset=binary",
    )
    video_aliases = {
        _canonical_multimodal_validator_hints(filename=name, content_type="video/mp4")["extension_family"]
        for name in ("clip.mp4", "clip.mov", "clip.m4v")
    }

    assert (
        jpeg
        == jpeg_alias
        == {
            "declared_mime": "image/jpeg",
            "extension_family": "image:image/jpeg",
        }
    )
    assert (
        unknown
        == octet_stream
        == {
            "declared_mime": "unconstrained",
            "extension_family": "unconstrained",
        }
    )
    assert video_aliases == {"video:isobmff"}
    assert _canonical_multimodal_validator_hints(
        filename="screen.png",
        content_type="image/jpeg",
    ) != _canonical_multimodal_validator_hints(
        filename="screen.jpg",
        content_type="image/jpeg",
    )


def test_parser_policy_preserves_chain_order_and_versions_fallback_semantics() -> None:
    multimodal_first = _canonical_multimodal_parser_policy(["openai_multimodal", "markitdown"])
    legacy_first = _canonical_multimodal_parser_policy(["markitdown", "openai_multimodal"])

    assert multimodal_first == {
        "chain": ["openai_multimodal", "markitdown"],
        "fallback_policy": "typed-not-applicable-only-v1",
    }
    assert legacy_first["chain"] == ["markitdown", "openai_multimodal"]
    assert multimodal_first != legacy_first


@pytest.mark.asyncio
async def test_anonymous_multimodal_retry_dedupes_only_inside_tenant_and_bank(monkeypatch) -> None:
    storage = _CapturingStorage()
    tasks = _CapturingTaskBackend()
    ledger = _DeduplicatingLedger()
    metrics = _RecordingMetrics()
    engine = _multimodal_admission_engine(monkeypatch, storage=storage, tasks=tasks, ledger=ledger)
    monkeypatch.setattr(memory_engine_module, "get_metrics_collector", lambda: metrics)
    media = _small_png()

    first = await MemoryEngine.submit_async_file_retain(
        engine,
        bank_id="bank-a",
        file_items=[_file_item(media)],
        document_tags=None,
        request_context=RequestContext(internal=True, tenant_id="tenant-a"),
    )
    repeat = await MemoryEngine.submit_async_file_retain(
        engine,
        bank_id="bank-a",
        file_items=[_file_item(media)],
        document_tags=None,
        request_context=RequestContext(internal=True, tenant_id="tenant-a"),
    )
    other_bank = await MemoryEngine.submit_async_file_retain(
        engine,
        bank_id="bank-b",
        file_items=[_file_item(media)],
        document_tags=None,
        request_context=RequestContext(internal=True, tenant_id="tenant-a"),
    )
    other_tenant = await MemoryEngine.submit_async_file_retain(
        engine,
        bank_id="bank-a",
        file_items=[_file_item(media)],
        document_tags=None,
        request_context=RequestContext(internal=True, tenant_id="tenant-b"),
    )

    assert first["operation_ids"] == repeat["operation_ids"]
    assert other_bank["operation_ids"] != first["operation_ids"]
    assert other_tenant["operation_ids"] != first["operation_ids"]
    first_spec, repeat_spec, other_bank_spec, other_tenant_spec = ledger.attempted_specs
    assert first_spec.document_id == repeat_spec.document_id
    assert first_spec.command_key == repeat_spec.command_key
    assert first_spec.document_id != other_bank_spec.document_id
    assert first_spec.document_id != other_tenant_spec.document_id
    assert first_spec.command_key != other_bank_spec.command_key
    assert first_spec.command_key != other_tenant_spec.command_key
    assert len(tasks.payloads) == 3
    assert len(storage.deleted) == 1


@pytest.mark.asyncio
async def test_parser_chain_order_forms_a_distinct_document_command(monkeypatch) -> None:
    storage = _CapturingStorage()
    tasks = _CapturingTaskBackend()
    ledger = _DeduplicatingLedger()
    engine = _multimodal_admission_engine(monkeypatch, storage=storage, tasks=tasks, ledger=ledger)
    media = _small_png()
    context = RequestContext(internal=True, tenant_id="tenant-a")

    for parser_chain in (
        ["openai_multimodal", "markitdown"],
        ["markitdown", "openai_multimodal"],
        ["openai_multimodal"],
    ):
        await MemoryEngine.submit_async_file_retain(
            engine,
            bank_id="bank-a",
            file_items=[
                _file_item(
                    media,
                    document_id="logical-document",
                    parser=parser_chain,
                )
            ],
            document_tags=None,
            request_context=context,
        )

    assert len(ledger.attempted_specs) == 3
    assert len({spec.descriptor_key for spec in ledger.attempted_specs}) == 3
    assert len({spec.retain_input_fingerprint for spec in ledger.attempted_specs}) == 3
    assert len({spec.command_key for spec in ledger.attempted_specs}) == 3
    assert len(tasks.payloads) == 3


@pytest.mark.asyncio
async def test_legacy_anonymous_file_retain_keeps_random_uuid_identity(monkeypatch) -> None:
    storage = _CapturingStorage()
    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = storage
    submitted: list[dict] = []

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    async def submit_operation(**kwargs):
        submitted.append(kwargs)
        return {"operation_id": f"operation-{len(submitted)}"}

    engine._authenticate_tenant = authenticate
    engine._submit_async_operation = submit_operation
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)
    data = b"ordinary legacy parser bytes"

    for _ in range(2):
        await MemoryEngine.submit_async_file_retain(
            engine,
            bank_id="legacy-bank",
            file_items=[
                _file_item(
                    data,
                    document_id=None,
                    filename="notes.txt",
                    content_type="text/plain",
                    parser="markitdown",
                )
            ],
            document_tags=None,
            request_context=RequestContext(internal=True, tenant_id="tenant-a"),
        )

    document_ids = [entry["task_payload"]["document_id"] for entry in submitted]
    assert all(document_id.startswith("file_") for document_id in document_ids)
    assert document_ids[0] != document_ids[1]
    assert all(not document_id.startswith("file_mm_") for document_id in document_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wrong_filename", "wrong_mime", "corrected_filename", "corrected_mime", "expected_failure"),
    [
        ("screen.png", "image/jpeg", "screen.png", "image/png", "media.mime_mismatch"),
        (
            "screen.jpg",
            "application/octet-stream",
            "screen.png",
            "application/octet-stream",
            "media.extension_mismatch",
        ),
    ],
    ids=["declared-mime", "extension-family"],
)
async def test_corrected_validator_hint_forms_new_command_and_revalidates(
    monkeypatch,
    wrong_filename: str,
    wrong_mime: str,
    corrected_filename: str,
    corrected_mime: str,
    expected_failure: str,
) -> None:
    storage = _CapturingStorage()
    ledger = _DeduplicatingLedger()
    outcomes: list[str] = []

    class _ValidatingTasks:
        async def submit_task(self, payload: dict) -> None:
            stored = next(entry for entry in reversed(storage.stores) if entry["key"] == payload["storage_key"])
            try:
                normalize_image(
                    file_data=stored["file_data"],
                    filename=payload["original_filename"],
                    declared_mime=payload["content_type"],
                    config=ImageNormalizationConfig(),
                )
            except MediaValidationError as exc:
                outcomes.append(exc.code)
            else:
                outcomes.append("accepted")

    engine = _multimodal_admission_engine(
        monkeypatch,
        storage=storage,
        tasks=_ValidatingTasks(),
        ledger=ledger,
    )
    media = _small_png()
    context = RequestContext(internal=True, tenant_id="tenant-a")

    await MemoryEngine.submit_async_file_retain(
        engine,
        bank_id="bank-a",
        file_items=[
            _file_item(
                media,
                document_id="logical-document",
                filename=wrong_filename,
                content_type=wrong_mime,
            )
        ],
        document_tags=None,
        request_context=context,
    )
    await MemoryEngine.submit_async_file_retain(
        engine,
        bank_id="bank-a",
        file_items=[
            _file_item(
                media,
                document_id="logical-document",
                filename=corrected_filename,
                content_type=corrected_mime,
            )
        ],
        document_tags=None,
        request_context=context,
    )

    assert outcomes == [expected_failure, "accepted"]
    wrong_spec, corrected_spec = ledger.attempted_specs
    assert wrong_spec.document_id == corrected_spec.document_id == "logical-document"
    assert wrong_spec.asset_sha256 == corrected_spec.asset_sha256
    assert wrong_spec.descriptor_key != corrected_spec.descriptor_key
    assert wrong_spec.retain_input_fingerprint != corrected_spec.retain_input_fingerprint
    assert wrong_spec.command_key != corrected_spec.command_key


@pytest.mark.asyncio
async def test_declared_image_budget_stops_before_engine_submission(monkeypatch) -> None:
    import hms_api.config

    memory = _Memory()
    metrics = _RecordingMetrics()
    config = _config(hms_api.config._get_raw_config())
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    monkeypatch.setattr("hms_api.api.http.get_metrics_collector", lambda: metrics)
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/default/banks/media-bank/files/retain",
            files={"files": ("screen.png", b"012345678", "image/png")},
            data={"request": json.dumps({"parser": "openai_multimodal"})},
        )

    assert response.status_code == 400
    assert "multimodal file exceeds" in response.json()["detail"]
    assert memory.submissions == []
    assert metrics.events == [
        {
            "media_kind": "image",
            "stage": "admission",
            "duration": 0.0,
            "success": False,
            "reason": "media.upload_bytes_exceeded",
            "asset_outcome": "rejected",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/octet-stream", "video/mp4"])
async def test_image_magic_uses_strict_budget_before_engine_submission(monkeypatch, content_type: str) -> None:
    import hms_api.config

    memory = _Memory()
    metrics = _RecordingMetrics()
    config = _config(hms_api.config._get_raw_config(), multimodal_max_image_bytes=16)
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    monkeypatch.setattr("hms_api.api.http.get_metrics_collector", lambda: metrics)
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/default/banks/media-bank/files/retain",
            files={"files": ("opaque.bin", _small_png(), content_type)},
            data={"request": json.dumps({"parser": "openai_multimodal"})},
        )

    assert response.status_code == 400
    assert "multimodal file exceeds" in response.json()["detail"]
    assert memory.submissions == []
    assert metrics.events[0]["media_kind"] == "image"
    assert metrics.events[0]["reason"] == "media.upload_bytes_exceeded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media", "expected_reads"),
    [
        (_small_png(), [12, 5]),
        (b"\x00\x00\x00\x18ftypisom" + b"V" * 40, [12, 21]),
    ],
    ids=["image", "video"],
)
async def test_internal_octet_stream_magic_stops_at_media_specific_limit(
    monkeypatch,
    media: bytes,
    expected_reads: list[int],
) -> None:
    import hms_api.config

    config = _config(
        hms_api.config._get_raw_config(),
        multimodal_max_image_bytes=16,
        multimodal_max_video_bytes=32,
    )
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    upload = _SizedReadUpload("opaque.bin", media, "application/octet-stream")
    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = _RejectUnexpectedStorage()

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    engine._authenticate_tenant = authenticate
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)
    item = {
        "file": upload,
        "document_id": "doc-budget",
        "context": None,
        "metadata": {},
        "tags": [],
        "timestamp": None,
        "parser": ["openai_multimodal"],
        "strategy": None,
    }

    with pytest.raises(ValueError, match="configured byte budget"):
        await MemoryEngine.submit_async_file_retain(
            engine,
            bank_id="bank-budget",
            file_items=[item],
            document_tags=None,
            request_context=RequestContext(internal=True),
        )

    assert upload.read_sizes == expected_reads


@pytest.mark.asyncio
async def test_magic_probe_overrides_spoofed_video_hint_for_image_budget(monkeypatch) -> None:
    import hms_api.config

    config = _config(
        hms_api.config._get_raw_config(),
        multimodal_max_image_bytes=16,
        multimodal_max_video_bytes=32,
    )
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    upload = _SizedReadUpload("opaque.bin", _small_png(), "video/mp4")
    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = _RejectUnexpectedStorage()

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    engine._authenticate_tenant = authenticate
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)
    item = _file_item(
        _small_png(),
        document_id="doc-spoofed-hint",
        filename="opaque.bin",
        content_type="video/mp4",
    )
    # Replace the generated upload so read-size observations are available.
    item["file"] = upload

    with pytest.raises(ValueError, match="configured byte budget"):
        await MemoryEngine.submit_async_file_retain(
            engine,
            bank_id="bank-budget",
            file_items=[item],
            document_tags=None,
            request_context=RequestContext(internal=True),
        )

    assert upload.read_sizes == [12, 5]


@pytest.mark.asyncio
async def test_multimodal_reader_without_sized_read_is_rejected(monkeypatch) -> None:
    import hms_api.config

    config = _config(hms_api.config._get_raw_config(), multimodal_max_image_bytes=16)
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = _RejectUnexpectedStorage()

    class _UnboundedReader:
        filename = "opaque.bin"
        content_type = "application/octet-stream"

        async def read(self):
            return _small_png()

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    engine._authenticate_tenant = authenticate
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)
    with pytest.raises(ValueError, match="must support bounded reads"):
        await MemoryEngine.submit_async_file_retain(
            engine,
            bank_id="bank-budget",
            file_items=[
                {
                    "file": _UnboundedReader(),
                    "document_id": "doc-unbounded",
                    "context": None,
                    "metadata": {},
                    "tags": [],
                    "timestamp": None,
                    "parser": ["openai_multimodal"],
                    "strategy": None,
                }
            ],
            document_tags=None,
            request_context=RequestContext(internal=True),
        )


@pytest.mark.asyncio
async def test_short_unknown_eof_cannot_bypass_hinted_limit(monkeypatch) -> None:
    import hms_api.config

    config = _config(
        hms_api.config._get_raw_config(),
        multimodal_max_image_bytes=4,
        multimodal_max_video_bytes=8,
    )
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    upload = _SizedReadUpload("opaque.bin", b"123456789", "video/mp4")
    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = _RejectUnexpectedStorage()

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    engine._authenticate_tenant = authenticate
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)
    with pytest.raises(ValueError, match="configured byte budget"):
        await MemoryEngine.submit_async_file_retain(
            engine,
            bank_id="bank-budget",
            file_items=[
                {
                    "file": upload,
                    "document_id": "doc-short-unknown",
                    "context": None,
                    "metadata": {},
                    "tags": [],
                    "timestamp": None,
                    "parser": ["openai_multimodal"],
                    "strategy": None,
                }
            ],
            document_tags=None,
            request_context=RequestContext(internal=True),
        )

    assert upload.read_sizes == [12, 3]


@pytest.mark.asyncio
async def test_bounded_multimodal_upload_preserves_existing_file_contract(monkeypatch) -> None:
    import hms_api.config

    memory = _Memory()
    config = _config(hms_api.config._get_raw_config(), multimodal_max_image_bytes=64)
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/default/banks/media-bank/files/retain",
            files={"files": ("screen.png", b"small-image", "image/png")},
            data={"request": json.dumps({"parser": ["openai_multimodal"]})},
        )

    assert response.status_code == 200
    assert response.json() == {"operation_ids": ["00000000-0000-0000-0000-000000000001"]}
    assert len(memory.submissions) == 1
    item = memory.submissions[0]["file_items"][0]
    assert item["parser"] == ["openai_multimodal"]
    assert item["document_id"] is None
    assert await item["file"].read() == b"small-image"


def _field_input(field: str, payload: str) -> tuple[_UnreadUpload, dict, list[str] | None]:
    upload = _UnreadUpload(f"{payload}.png" if field == "filename" else "safe.png")
    item = {
        "file": upload,
        "document_id": payload if field == "document_id" else "doc-admission",
        "context": payload if field == "context" else "safe context",
        "metadata": {"nested": {"payload": payload}} if field == "metadata" else {"safe": "value"},
        "tags": [payload] if field == "tags" else ["safe-tag"],
        "timestamp": payload if field == "timestamp" else None,
        "parser": ["openai_multimodal", payload] if field == "parser" else ["openai_multimodal"],
        "strategy": payload if field == "strategy" else None,
    }
    document_tags = [payload] if field == "document_tags" else ["safe-document-tag"]
    return upload, item, document_tags


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "filename",
        "document_id",
        "context",
        "metadata",
        "tags",
        "timestamp",
        "parser",
        "strategy",
        "document_tags",
    ],
)
@pytest.mark.parametrize("payload_kind", ["data_url", "long_base64"])
async def test_encoded_payload_user_fields_are_rejected_before_file_read_or_storage(
    monkeypatch,
    field: str,
    payload_kind: str,
) -> None:
    raw = b"multimodal-admission-sentinel" * 12
    encoded = base64.b64encode(raw).decode("ascii")
    payload = f"data:image/png;base64,{encoded}" if payload_kind == "data_url" else encoded
    upload, item, document_tags = _field_input(field, payload)
    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = _RejectUnexpectedStorage()

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    engine._authenticate_tenant = authenticate
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)

    with pytest.raises(ValueError, match="cannot contain an encoded media payload") as exc_info:
        await MemoryEngine.submit_async_file_retain(
            engine,
            bank_id="bank-admission",
            file_items=[item],
            document_tags=document_tags,
            request_context=RequestContext(internal=True),
        )

    assert payload not in str(exc_info.value)
    assert upload.read_calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        "data:;base64,AAAA",
        "\n".join(
            base64.b64encode(b"wrapped-multimodal-payload" * 16).decode("ascii")[offset : offset + 76]
            for offset in range(0, 512, 76)
        ),
    ],
    ids=["empty-media-type-data-url", "mime-wrapped-base64"],
)
def test_noncanonical_encoded_payload_variants_are_rejected(payload: str) -> None:
    with pytest.raises(ValueError, match="cannot contain an encoded media payload"):
        _reject_multimodal_transport_payload_in_metadata(
            {"context": payload, "metadata": {}, "tags": []},
            "safe.png",
            [],
        )


def test_recursive_admission_guard_allows_ordinary_long_text_and_separate_short_tokens() -> None:
    ordinary_prose = (
        "This is ordinary long project context, with spaces, punctuation, and human-readable sentences. " * 20
    )
    _reject_multimodal_transport_payload_in_metadata(
        {
            "context": ordinary_prose,
            "metadata": {"nested": {"notes": ordinary_prose, "tokens": ["A" * 80] * 8}},
            "tags": ["ordinary-tag"],
        },
        "ordinary-long-filename-with-readable-words.png",
        ["ordinary-document-tag"],
    )


def test_magic_classifies_octet_stream_mp4_as_video_for_early_status() -> None:
    mp4_header = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"

    assert (
        _classify_multimodal_media_kind(
            mp4_header,
            filename="opaque-upload.bin",
            content_type="application/octet-stream",
        )
        == "video"
    )


@pytest.mark.asyncio
async def test_multimodal_storage_key_omits_raw_hash_filename_and_log_identifiers(monkeypatch, caplog) -> None:
    from hms_api.engine.multimodal.ledger import MultimodalLedger

    bank_id = "private-bank-storage-key"
    document_id = "private-document-storage-key"
    filename = "private-customer-filename.png"
    file_data = b"private-asset-bytes-for-storage-key"
    asset_sha256 = hashlib.sha256(file_data).hexdigest()
    upload = _SingleReadUpload(filename, file_data)
    storage = _CapturingStorage()
    tasks = _CapturingTaskBackend()
    connection = _AdmissionConnection()
    engine = object.__new__(MemoryEngine)
    engine._backend = SimpleNamespace()
    engine._file_storage = storage
    engine._parser_registry = _FingerprintRegistry()
    engine._task_backend = tasks

    async def authenticate(_request_context) -> None:
        return None

    async def existing_bank(_backend, _bank_id):
        return SimpleNamespace(), False

    async def get_backend():
        return SimpleNamespace()

    class _Ledger:
        async def admit_document_command(self, _conn, spec, *, now):
            del now
            return SimpleNamespace(
                created=True,
                command=SimpleNamespace(sequence=1, operation_id=spec.operation_id),
            )

    engine._authenticate_tenant = authenticate
    engine._get_backend = get_backend
    monkeypatch.setattr(memory_engine_module.bank_utils, "get_or_create_bank_profile", existing_bank)
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))
    monkeypatch.setattr(MultimodalLedger, "for_connection", lambda *_args, **_kwargs: _Ledger())

    result = await MemoryEngine.submit_async_file_retain(
        engine,
        bank_id=bank_id,
        file_items=[
            {
                "file": upload,
                "document_id": document_id,
                "context": "safe context",
                "metadata": {},
                "tags": ["safe-tag"],
                "timestamp": None,
                "parser": ["openai_multimodal"],
                "strategy": None,
            }
        ],
        document_tags=["safe-document-tag"],
        request_context=RequestContext(internal=True, tenant_id="private-tenant-storage-key"),
    )

    assert len(result["operation_ids"]) == 1
    [stored] = storage.stores
    storage_key = stored["key"]
    assert storage_key.startswith("media/")
    for forbidden in (asset_sha256, filename, bank_id, document_id):
        assert forbidden not in storage_key
    [task] = tasks.payloads
    assert task["storage_key"] == storage_key
    assert task["media_kind"] == "image"

    # Admission itself must not log any sensitive locator inputs.  Worker-side
    # multimodal log assertions live in the security E2E where that path runs.
    admission_logs = "\n".join(record.getMessage() for record in caplog.records)
    for forbidden in (asset_sha256, filename, bank_id, document_id):
        assert forbidden not in admission_logs
