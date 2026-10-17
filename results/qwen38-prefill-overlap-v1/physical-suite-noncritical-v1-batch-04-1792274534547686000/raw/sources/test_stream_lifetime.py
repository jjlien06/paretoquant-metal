"""CPU-only ownership regressions against the real stdlib-safe stream module.

Tracked bytearrays stand in for file.read bytes; only Python payload ownership is
measured, not allocator/RSS/MLX residency. All file/socket boundaries are fakes.
"""

import hashlib
import importlib.util
import json
import struct
import weakref
from pathlib import Path

import pytest

import stream_weights as stream


ORIGINAL_SHA256 = "b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9"


def original_module():
    path = Path(__file__).parents[1] / "024-qwen38-hidden-prefill" / "stream_weights.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA256
    spec = importlib.util.spec_from_file_location("frozen_stream_lifetime_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Payload(bytearray):
    """Weak-referenceable buffer with identical hash/buffer-protocol contents."""


class Tracker:
    def __init__(self):
        self.refs = []
        self.peak = 0
        self.before_read = []

    @property
    def live(self):
        return sum(size for ref, size in self.refs if ref() is not None)

    def allocate(self, data):
        payload = Payload(data)
        self.refs.append((weakref.ref(payload), len(payload)))
        self.peak = max(self.peak, self.live)
        return payload


def fixture_data():
    tensors, provenance, payloads = {}, {}, {}
    offset = 0
    for index, size in enumerate((1024, 2048, 4096)):
        name = "tensor_" + str(index)
        tensors[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        provenance[name] = {"path": name, "data_start": 17}
        payloads[name] = bytes([index + 1]) * size
        offset += size
    # Reverse insertion order: production must still emit sorted tensor names.
    return dict(reversed(list(tensors.items()))), provenance, payloads


class FakeFile:
    def __init__(self, name, tensors, payloads, tracker, trace, truncate):
        self.name, self.tensors, self.payloads = name, tensors, payloads
        self.tracker, self.trace, self.truncate = tracker, trace, truncate
        self.closed = False

    def __enter__(self):
        self.trace.append(("open", self.name))
        return self

    def seek(self, offset):
        assert offset == 17 + self.tensors[self.name]["data_offsets"][0]
        self.trace.append(("seek", self.name, offset))

    def read(self, count):
        assert count == len(self.payloads[self.name])
        self.tracker.before_read.append(self.tracker.live)
        self.trace.append(("read", self.name, count))
        data = self.payloads[self.name]
        if self.name == self.truncate:
            data = data[:-1]
        return self.tracker.allocate(data)

    def __exit__(self, *args):
        self.closed = True
        self.trace.append(("close", self.name))


def disk_fixture(module, monkeypatch, truncate=None):
    tensors, provenance, payloads = fixture_data()
    tracker, trace, handles = Tracker(), [], []

    class FakePath:
        def __init__(self, name):
            self.name = name

        def open(self, mode):
            assert mode == "rb"
            handle = FakeFile(self.name, tensors, payloads, tracker, trace, truncate)
            handles.append(handle)
            return handle

    monkeypatch.setattr(module, "Path", FakePath)
    return tensors, provenance, payloads, tracker, trace, handles


def observe_disk(module, monkeypatch, truncate=None):
    tensors, origins, payloads, tracker, trace, handles = disk_fixture(module, monkeypatch, truncate)
    records = module.tensor_records(tensors, origins)
    observed, error = [], None
    try:
        for record in records:
            name = record["name"]
            assert handles[-1].closed, "File must already be closed when payload is yielded"
            assert record["descriptor"] is tensors[name]
            assert bytes(record["payload"]) == payloads[name]
            assert record["sha256"] == hashlib.sha256(payloads[name]).hexdigest()
            observed.append({**record, "payload": bytes(record["payload"])})
            del record
    except ValueError as exc:
        error = (type(exc).__name__, str(exc))
    finally:
        records.close()
    assert all(handle.closed for handle in handles)
    assert tracker.live == 0
    return {"records": observed, "error": error, "trace": trace}, tracker


def test_tensor_records_releases_payload_before_next_read(monkeypatch):
    frozen = original_module()
    happy_tracker = None
    # Byte/descriptor/digest/order/file-close/truncation behavior stays exact.
    for truncate in (None, "tensor_0", "tensor_1", "tensor_2"):
        expected, old_tracker = observe_disk(frozen, monkeypatch, truncate)
        actual, tracker = observe_disk(stream, monkeypatch, truncate)
        assert actual == expected
        if truncate is not None:
            assert actual["error"] == ("ValueError", "Source payload truncated")
        else:
            assert old_tracker.before_read == [0, 1024, 2048]
            assert old_tracker.peak == 6144
            happy_tracker = tracker
    assert happy_tracker is not None
    assert happy_tracker.before_read == [0, 0, 0], "Previous payload survives into next file.read"
    assert happy_tracker.peak == 4096


class FakeSocket:
    def __init__(self, tracker=None, fail_at=None, wire=b""):
        self.tracker, self.fail_at = tracker, fail_at
        self.calls, self.wire, self.position = [], bytearray(wire), 0

    def sendall(self, data):
        if isinstance(data, Payload):
            assert self.tracker is not None
            assert self.tracker.live >= len(data), "Payload released before full send"
        if len(self.calls) == self.fail_at:
            raise OSError("fixture full-send failure")
        self.calls.append(bytes(data))
        self.wire.extend(data)

    def recv_into(self, view):
        count = min(len(view), 7, len(self.wire) - self.position)
        view[:count] = self.wire[self.position:self.position + count]
        self.position += count
        return count


class FreshRecords:
    """No producer reference survives __next__; isolates sender ownership."""
    def __init__(self, tensors, payloads, tracker):
        self.tensors, self.payloads, self.tracker = tensors, payloads, tracker
        self.names = iter(sorted(tensors))
        self.before_next = []

    def __iter__(self):
        return self

    def __next__(self):
        self.before_next.append(self.tracker.live)
        name = next(self.names)
        return {"name": name, "descriptor": self.tensors[name],
                "payload": self.tracker.allocate(self.payloads[name]),
                "sha256": hashlib.sha256(self.payloads[name]).hexdigest()}


def expected_wire(tensors, payloads):
    chunks = []
    for name in sorted(tensors):
        metadata = {"name": name, "descriptor": tensors[name],
                    "sha256": hashlib.sha256(payloads[name]).hexdigest()}
        raw = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
        chunks.extend((struct.pack("!I", len(raw)), raw, payloads[name]))
    raw = b'{"end":true}'
    chunks.extend((struct.pack("!I", len(raw)), raw))
    return chunks


def observe_sender(module):
    tensors, _, payloads = fixture_data()
    tracker = Tracker()
    records = FreshRecords(tensors, payloads, tracker)
    sock = FakeSocket(tracker)
    module.send_records(sock, records)
    assert sock.calls == expected_wire(tensors, payloads)
    assert tracker.live == 0
    return bytes(sock.wire), records, tracker


def test_send_records_releases_record_before_generator_next():
    old_wire, old_records, old_tracker = observe_sender(original_module())
    wire, records, tracker = observe_sender(stream)
    assert wire == old_wire
    assert old_records.before_next == [0, 1024, 2048, 4096]
    assert old_tracker.peak == 6144
    assert records.before_next == [0, 0, 0, 0], "Sender retains record during next(records)"
    assert tracker.peak == 4096


def observe_disk_sender(module, monkeypatch, truncate=None):
    tensors, origins, payloads, tracker, trace, handles = disk_fixture(module, monkeypatch, truncate)
    records = module.tensor_records(tensors, origins)
    sock, error = FakeSocket(tracker), None
    try:
        module.send_records(sock, records)
    except ValueError as exc:
        error = (type(exc).__name__, str(exc))
    finally:
        records.close()
    assert all(handle.closed for handle in handles)
    assert tracker.live == 0
    if truncate is None:
        assert sock.calls == expected_wire(tensors, payloads)
    return {"wire": bytes(sock.wire), "error": error, "trace": trace}, tracker


@pytest.mark.parametrize("truncate", [None, "tensor_0", "tensor_1", "tensor_2"])
def test_real_disk_sender_exact_wire_truncation_and_bounded_peak(monkeypatch, truncate):
    expected, old_tracker = observe_disk_sender(original_module(), monkeypatch, truncate)
    actual, tracker = observe_disk_sender(stream, monkeypatch, truncate)
    assert actual == expected
    assert tracker.before_read == [0] * len(tracker.before_read)
    if truncate is None:
        assert (old_tracker.peak, tracker.peak) == (6144, 4096)
        tensors, _, payloads = fixture_data()
        records = list(stream.receive_records(FakeSocket(wire=actual["wire"]), tensors))
        assert [record["name"] for record in records] == sorted(tensors)
        for record in records:
            assert record["payload"] == payloads[record["name"]]
            assert record["sha256"] == hashlib.sha256(record["payload"]).hexdigest()
    else:
        assert actual["error"] == ("ValueError", "Source payload truncated")
        assert not actual["wire"].endswith(b'{"end":true}')


@pytest.mark.parametrize("fail_at", range(11))
def test_sender_fullsend_errors_and_wire_prefix_unchanged(fail_at):
    outcomes = []
    for module in (original_module(), stream):
        tensors, _, payloads = fixture_data()
        tracker = Tracker()
        records = FreshRecords(tensors, payloads, tracker)
        sock = FakeSocket(tracker, fail_at=fail_at)
        with pytest.raises(OSError, match="^fixture full-send failure$"):
            module.send_records(sock, records)
        assert sock.calls == expected_wire(tensors, payloads)[:fail_at]
        assert tracker.live == 0
        outcomes.append((sock.calls, len(records.before_next)))
    assert outcomes[0] == outcomes[1]


def decode_outcome(module, wire, tensors):
    records, error = [], None
    try:
        records = list(module.receive_records(FakeSocket(wire=wire), tensors))
    except (ValueError, KeyError) as exc:
        error = (type(exc).__name__, str(exc))
    return records, error


@pytest.mark.parametrize("cut", [0, 1, 3, 4, 5, 180, 1200, 4000, -1])
def test_receiver_truncation_errors_unchanged(cut):
    tensors, _, payloads = fixture_data()
    wire = b"".join(expected_wire(tensors, payloads))
    assert cut < len(wire)
    expected = decode_outcome(original_module(), wire[:cut], tensors)
    assert decode_outcome(stream, wire[:cut], tensors) == expected
    assert expected[1] == ("ValueError", "Stream truncated")


@pytest.mark.parametrize("raw,message", [
    (struct.pack("!I", 0), "Invalid metadata frame length"),
    (struct.pack("!I", 2 * 1024**2 + 1), "Invalid metadata frame length"),
    (struct.pack("!I", 2) + b"[]", "Metadata frame must be an object"),
    (struct.pack("!I", 2) + b"{}", "Unexpected network tensor metadata"),
])
def test_receiver_metadata_errors_unchanged(raw, message):
    tensors, _, _ = fixture_data()
    expected = decode_outcome(original_module(), raw, tensors)
    assert decode_outcome(stream, raw, tensors) == expected
    assert expected[1] == ("ValueError", message)


def test_empty_sender_preserves_terminator():
    for module in (original_module(), stream):
        sock = FakeSocket()
        module.send_records(sock, iter(()))
        assert sock.calls == [struct.pack("!I", 12), b'{"end":true}']
        assert list(module.receive_records(FakeSocket(wire=bytes(sock.wire)), {})) == []


def test_tensor_payload_stays_valid_for_retaining_consumer_and_close(monkeypatch):
    tensors, origins, payloads, tracker, _, _ = disk_fixture(stream, monkeypatch)
    records = stream.tensor_records(tensors, origins)
    first = next(records)
    second = next(records)
    records.close()
    assert bytes(first["payload"]) == payloads[first["name"]]
    assert bytes(second["payload"]) == payloads[second["name"]]
    del first, second
    assert tracker.live == 0
