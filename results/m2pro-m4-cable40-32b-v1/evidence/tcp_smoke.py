"""Peer-bound Thunderbolt TCP smoke; fixed buffers, no payload files."""
import hashlib
import json
import socket
import statistics
import sys
import time
from pathlib import Path

BLOCK = bytes(range(256)) * 4096
BLOCK_BYTES = len(BLOCK)
VERIFY_BYTES = 16 * 1024 * 1024
WARM_BYTES = 256 * 1024 * 1024
TRIAL_BYTES = 512 * 1024 * 1024


def setup(connection):
    connection.settimeout(15)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)


def send_json(connection, value):
    connection.sendall(json.dumps(value).encode() + b'\n')


def read_json(connection):
    data = bytearray()
    while not data.endswith(b'\n'):
        item = connection.recv(1)
        if not item:
            raise EOFError('Truncated control record')
        data.extend(item)
        if len(data) > 4096:
            raise ValueError('Oversized control record')
    return json.loads(data)


def transfer(connection, direction, size, verify):
    digest = hashlib.sha256() if verify else None
    completed = 0
    storage = bytearray(BLOCK_BYTES) if direction == 'receive' else None
    while completed < size:
        remaining = min(BLOCK_BYTES, size - completed)
        if direction == 'send':
            connection.sendall(memoryview(BLOCK)[:remaining])
            if digest is not None:
                digest.update(memoryview(BLOCK)[:remaining])
            count = remaining
        else:
            count = connection.recv_into(storage, remaining)
            if not count:
                raise EOFError('Incomplete payload')
            if digest is not None:
                digest.update(memoryview(storage)[:count])
        completed += count
    return {'bytes': completed, 'sha256': digest.hexdigest() if digest is not None else None}


def expected_digest(size):
    result = hashlib.sha256()
    for start in range(0, size, BLOCK_BYTES):
        result.update(memoryview(BLOCK)[:min(BLOCK_BYTES, size - start)])
    return result.hexdigest()


def serve(bind, peer, output):
    rows = []
    with socket.socket() as listener:
        listener.bind((bind, 0))
        listener.listen(1)
        listener.settimeout(25)
        print(json.dumps({'ready': True, 'address': list(listener.getsockname()), 'peer': peer}), flush=True)
        for _ in range(10):
            connection, address = listener.accept()
            with connection:
                assert address[0] == peer
                setup(connection)
                request = read_json(connection)
                assert set(request) == {'direction', 'bytes', 'verify'}
                assert request['direction'] in ['upload', 'download']
                assert type(request['verify']) is bool
                assert type(request['bytes']) is int and request['bytes'] in [VERIFY_BYTES, WARM_BYTES, TRIAL_BYTES]
                assert request['verify'] == (request['bytes'] == VERIFY_BYTES)
                send_json(connection, {'ready': True})
                assert connection.recv(1) == b'G'
                direction = 'receive' if request['direction'] == 'upload' else 'send'
                receipt = transfer(connection, direction, request['bytes'], request['verify'])
                send_json(connection, receipt)
                rows.append({'request': request, 'receipt': receipt, 'peer': address[0]})
    report = {'completed': True, 'request_count': len(rows), 'rows': rows,
              'bind': bind, 'peer': peer, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with Path(output).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'completed': True, 'request_count': len(rows)}), flush=True)


def client(host, port, bind, output):
    rows = []
    for direction in ['upload', 'download']:
        for label, size, verify in [('integrity', VERIFY_BYTES, True), ('warmup', WARM_BYTES, False),
                ('trial-0', TRIAL_BYTES, False), ('trial-1', TRIAL_BYTES, False), ('trial-2', TRIAL_BYTES, False)]:
            with socket.socket() as connection:
                setup(connection)
                connection.bind((bind, 0))
                connection.connect((host, port))
                send_json(connection, {'direction': direction, 'bytes': size, 'verify': verify})
                assert read_json(connection) == {'ready': True}
                began = time.perf_counter()
                connection.sendall(b'G')
                receipt = transfer(connection, 'send' if direction == 'upload' else 'receive', size, verify)
                acknowledgement = read_json(connection)
                elapsed = time.perf_counter() - began
                assert receipt['bytes'] == acknowledgement['bytes'] == size
                if verify:
                    assert receipt['sha256'] == acknowledgement['sha256'] == expected_digest(size)
                row = {'direction': direction, 'label': label, 'bytes': size,
                       'digest_verified': verify, 'sha256': receipt['sha256'],
                       'seconds': None if verify else elapsed,
                       'Gbps': None if verify else size * 8 / elapsed / 1e9,
                       'socket_local': list(connection.getsockname()), 'socket_peer': list(connection.getpeername())}
                rows.append(row)
    summary = {d: {'trials': 3, 'median_Gbps': statistics.median(r['Gbps'] for r in rows
                if r['direction'] == d and r['label'].startswith('trial-'))} for d in ['upload', 'download']}
    report = {'completed': True, 'rows': rows, 'summary': summary,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Short Python single-stream TCP smoke, not maximum cable bandwidth or inference speedup',
        'upload': 'M2 Pro to M4', 'download': 'M4 to M2 Pro',
        'buffer_bytes': BLOCK_BYTES, 'timed_bytes_per_trial': TRIAL_BYTES,
        'digest_check_separate': True, 'transport': 'Peer-bound trusted direct bridge IPv4, no encryption/authentication',
        'timer': 'Client payload plus final acknowledgement; connect/control setup excluded'}
    with Path(output).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'serve':
        serve(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        client(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
