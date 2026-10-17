#!/usr/bin/env python3
"""Bounded wired-policy CONTROL_NOT_MODEL_GATE. Import is CPU-only/inert."""
import math

MIB = 1024**2
RETAINED_BYTES = 64 * MIB
ACTIVE_CAP_BYTES = 192 * MIB
MAX_CHUNK_BYTES = MIB
MAX_ITERATIONS = 128
CLASSIFICATION = 'CONTROL_NOT_MODEL_GATE'


def validate_config(mode, *, total_bytes=RETAINED_BYTES, chunk_bytes=MAX_CHUNK_BYTES, deadline=30) -> dict:
    if type(mode) is not str or mode not in ('recommended', 'unwired'):
        raise ValueError('mode must be recommended or unwired')
    if (type(total_bytes) is not int or not 0 < total_bytes <= RETAINED_BYTES
            or type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_CHUNK_BYTES
            or chunk_bytes % 4 or total_bytes % chunk_bytes
            or not 1 <= total_bytes // chunk_bytes <= MAX_ITERATIONS):
        raise ValueError('invalid bounded U32 payload')
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or not 0 < deadline <= 30:
        raise ValueError('deadline must be finite and <=30 seconds')
    return dict(mode=mode, total_bytes=total_bytes, chunk_bytes=chunk_bytes,
                iterations=total_bytes // chunk_bytes, deadline=deadline,
                active_cap_bytes=ACTIVE_CAP_BYTES,
                # Conservative logical host payload bound, including possible
                # retained CPU input/copy lifetimes. Not an RSS/OS/runtime bound.
                host_payload_cap_bytes=2 * total_bytes + 4 * chunk_bytes + 4 * MIB)


def parse_sample(raw):
    """Accept bounded verbatim OS sampler output, never infer absent counters."""
    import re
    if type(raw) is not dict or set(raw) != {'vm_stat', 'pressure', 'mem_total', 'rss_kib'}:
        raise ValueError('missing exact sampler fields')
    if type(raw['pressure']) is not str or raw['pressure'] != '1\n':
        raise ValueError('sampled pressure must be exactly 1')
    numbers = {}
    for key in ('mem_total', 'rss_kib'):
        value = raw[key]
        if type(value) is not str or len(value) > 32 or not re.fullmatch(r'[0-9]+\n', value):
            raise ValueError('invalid OS integer')
        numbers[key] = int(value)
        if not 0 <= numbers[key] < 2**63:
            raise ValueError('OS integer overflow')
    text = raw['vm_stat']
    if type(text) is not str or len(text) > 65536:
        raise ValueError('invalid vm_stat size/type')
    lines = text.splitlines()
    header = re.fullmatch(r'Mach Virtual Memory Statistics: \(page size of (4096|16384) bytes\)',
                          lines[0] if lines else '')
    if not header:
        raise ValueError('invalid vm_stat header')
    fields = {}
    for line in lines[1:]:
        match = re.fullmatch(r'("[A-Za-z][A-Za-z -]*"|[A-Za-z][A-Za-z -]*):\s+([0-9]+)\.', line)
        if not match:
            raise ValueError('malformed vm_stat counter')
        key = match[1].strip('"')
        count = int(match[2])
        if key in fields or len(fields) >= 64 or len(key) > 64 or not 0 <= count < 2**64:
            raise ValueError('duplicate/overflow/oversize vm_stat counter')
        fields[key] = count
    required = {'Pages free', 'Pages active', 'Pages inactive', 'Pages wired down',
                'Pages occupied by compressor', 'Swapins', 'Swapouts'}
    if not required <= set(fields) or numbers['mem_total'] <= 0:
        raise ValueError('missing required vm_stat/memory information')
    return dict(pressure=1, page_size=int(header[1]), vm=fields,
                mem_total_bytes=numbers['mem_total'], rss_bytes=numbers['rss_kib'] * 1024)


def run_control(mode, *, backend, sampler, clock=None, **options):
    """Pure injected runner. Blocking backend calls need the parent's watchdog."""
    import gc
    import hashlib
    import time
    cfg = validate_config(mode, **options)
    if not callable(sampler):
        raise ValueError('sampler callback is required')
    clock = time.monotonic if clock is None else clock
    began = clock()
    report: dict = dict(classification=CLASSIFICATION, config=cfg, ok=False, policy={},
                  arrays=[], samples=[], memory=[], errors=[], cleanup=[], retained_bytes=0)
    held = []
    a = raw = returned = None
    previous = {}

    def check_time():
        if clock() - began >= cfg['deadline']:
            raise TimeoutError('control deadline')

    def sample_at(stage):
        snapshot = parse_sample(sampler())
        report['samples'].append(dict(stage=stage, elapsed_seconds=clock()-began, **snapshot))

    def memory_at(stage):
        values = backend.memory()
        if (type(values) is not dict or set(values) != {'active', 'cache', 'peak'}
                or any(type(v) is not int or v < 0 for v in values.values())
                or max(values.values()) > cfg['active_cap_bytes']):
            raise ValueError('MLX memory cap or malformed counter')
        report['memory'].append(dict(stage=stage, **values))
        if stage == 'after_cleanup' and (values['active'] != 0 or values['cache'] != 0):
            raise ValueError('allocator not empty after cleanup')
        return values

    try:
        check_time()
        sample_at('before')
        report['backend'] = backend.describe()
        recommended = report['backend']['device']['max_recommended_working_set_size']
        if type(recommended) is not int or recommended <= 0:
            raise ValueError('invalid recommended working set')
        if memory_at('before')['active'] != 0:
            raise ValueError('require fresh exclusive child allocator')
        previous['memory'] = backend.set_memory_limit(cfg['active_cap_bytes'])
        previous['cache'] = backend.set_cache_limit(0)
        requested = recommended if mode == 'recommended' else 0
        report['policy']['requested_bytes'] = requested
        previous['wired'] = backend.set_wired_limit(requested)
        report['policy']['previous_bytes'] = previous['wired']
        report['policy']['readback_bytes'] = backend.set_wired_limit(requested)
        if (type(report['policy']['readback_bytes']) is not int
                or report['policy']['readback_bytes'] != requested):
            raise ValueError('wired readback mismatch')
        for index in range(cfg['iterations']):
            check_time()
            raw = hashlib.shake_256(b'pq-wired-control-v1:' + index.to_bytes(4, 'little')).digest(cfg['chunk_bytes'])
            expected = dict(sha256=hashlib.sha256(raw).hexdigest(),
                            first=int.from_bytes(raw[:4], 'little'), last=int.from_bytes(raw[-4:], 'little'))
            a = backend.from_u32_bytes(raw)
            held.append(a)
            backend.evaluate(a)
            returned = backend.to_u32_bytes(a)
            if type(returned) is not bytes or len(returned) != cfg['chunk_bytes']:
                raise ValueError('invalid U32 roundtrip length/type')
            observed = dict(sha256=hashlib.sha256(returned).hexdigest(),
                            first=int.from_bytes(returned[:4], 'little'), last=int.from_bytes(returned[-4:], 'little'))
            report['arrays'].append(dict(index=index, bytes=len(raw), expected=expected, observed=observed))
            if observed != expected or returned != raw:
                raise ValueError('U32 payload changed')
            report['retained_bytes'] += len(raw)
            a = raw = returned = None
            memory_at('during')
            sample_at('during')
            check_time()
        for index, a in enumerate(held):
            check_time()
            backend.evaluate(a)
            returned = backend.to_u32_bytes(a)
            expected = report['arrays'][index]['expected']
            if (type(returned) is not bytes or len(returned) != cfg['chunk_bytes']
                    or hashlib.sha256(returned).hexdigest() != expected['sha256']
                    or int.from_bytes(returned[:4], 'little') != expected['first']
                    or int.from_bytes(returned[-4:], 'little') != expected['last']):
                raise ValueError('retained U32 payload changed')
            report['arrays'][index]['retained_verified'] = True
            returned = None
        a = None
        check_time()
        report['ok'] = True
    except Exception as exc:
        report['errors'].append(dict(stage='run', type=type(exc).__name__))
    finally:
        # Each cleanup step is independent: failed sync must not skip restoration.
        def cleanup(name, action):
            try:
                action()
                report['cleanup'].append(dict(step=name, ok=True))
            except Exception as exc:
                report['errors'].append(dict(stage=name, type=type(exc).__name__))
                report['cleanup'].append(dict(step=name, ok=False))
        cleanup('synchronize_before_free', backend.synchronize)
        a = raw = returned = None
        held.clear()
        gc.collect()
        cleanup('clear_cache', backend.clear_cache)
        cleanup('synchronize_after_free', backend.synchronize)
        for key in ('wired', 'cache', 'memory'):
            if key in previous:
                def restore(key=key):
                    setter = getattr(backend, 'set_' + key + '_limit')
                    setter(previous[key])
                    readback = setter(previous[key])
                    if key == 'wired':
                        report['policy']['restored_readback_bytes'] = readback
                    if type(readback) is not int or readback != previous[key]:
                        raise ValueError('restoration readback mismatch')
                cleanup('restore_' + key, restore)
        cleanup('after_memory', lambda: memory_at('after_cleanup'))
        cleanup('after_sample', lambda: sample_at('after'))
        report['elapsed_seconds'] = clock() - began
        report['ok'] = report['ok'] and not report['errors']
    return report


def parse_args(argv=None):
    import argparse
    import re
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('recommended', 'unwired'))
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--host-role', required=True, choices=('local', 'remote'))
    args = parser.parse_args(argv)
    if re.fullmatch('[a-f0-9]{32}', args.nonce) is None:
        raise ValueError('nonce must be a literal lowercase UUID hex')
    return args


def make_backend(mx, np, versions):
    """No imports or execution here: the physical entry supplies MLX/NumPy."""
    class Backend:
        def describe(self):
            return dict(device=mx.device_info(), versions=dict(versions))
        def set_memory_limit(self, value):
            # Fail closed if the installed API lacks strict allocator support.
            return mx.set_memory_limit(value, relaxed=False)
        def set_cache_limit(self, value): return mx.set_cache_limit(value)
        def set_wired_limit(self, value): return mx.set_wired_limit(value)
        def memory(self):
            return dict(active=mx.get_active_memory(), cache=mx.get_cache_memory(), peak=mx.get_peak_memory())
        def from_u32_bytes(self, data):
            cpu = np.frombuffer(data, dtype=np.dtype('<u4'))
            return mx.array(cpu, dtype=mx.uint32)
        def evaluate(self, array): mx.eval(array)
        def to_u32_bytes(self, array):
            cpu = np.asarray(array)
            if cpu.dtype != np.dtype('uint32') or cpu.ndim != 1:
                raise ValueError('expected one-dimensional uint32 MLX roundtrip')
            return cpu.astype(np.dtype('<u4'), copy=False).tobytes(order='C')
        def synchronize(self): mx.synchronize()
        def clear_cache(self): mx.clear_cache()
    return Backend()


def physical_runtime():
    """PHYSICAL ONLY: called by main, never by CPU tests or module import."""
    import importlib.metadata
    import os
    import platform
    import re
    import subprocess
    import sys
    import mlx.core as mx
    import numpy as np

    def read(argv, cap=65536):
        result = subprocess.run(argv, capture_output=True, timeout=0.25, check=True)
        if result.stderr or len(result.stdout) > cap:
            raise ValueError('OS sampler output is missing/malformed/oversize')
        return result.stdout.decode('ascii', errors='strict')

    def sampler():
        pressure = read(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], 64)
        vm = read(['/usr/bin/vm_stat'])
        total = read(['/usr/sbin/sysctl', '-n', 'hw.memsize'], 64)
        rss = read(['/bin/ps', '-o', 'rss=', '-p', str(os.getpid())], 64)
        if re.fullmatch(r'[ \t]*[0-9]+\n', rss) is None:
            raise ValueError('invalid ps RSS measurement')
        return dict(pressure=pressure, vm_stat=vm, mem_total=total, rss_kib=rss.lstrip(' \t'))

    versions = dict(mlx=importlib.metadata.version('mlx'), numpy=np.__version__,
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    hostname=platform.node(), mlx_module_path=getattr(mx, '__file__', None))
    return make_backend(mx, np, versions), sampler


def main(argv=None, *, runtime_factory=None, runner=run_control):
    import json
    import os
    import time
    args = parse_args(argv)
    began = time.monotonic()
    backend, sampler = (physical_runtime if runtime_factory is None else runtime_factory)()
    remaining = 30 - (time.monotonic() - began)
    if remaining <= 0:
        raise TimeoutError('physical import deadline')
    report = runner(args.mode, backend=backend, sampler=sampler,
                    total_bytes=RETAINED_BYTES, chunk_bytes=MAX_CHUNK_BYTES, deadline=remaining)
    report['binding'] = dict(nonce=args.nonce, host_role=args.host_role, child_pid=os.getpid(),
                             actor_role='rank0' if args.host_role == 'local' else 'rank1', mode=args.mode)
    report['physical_elapsed_seconds'] = time.monotonic() - began
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
