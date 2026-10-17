"""Parent seal + actual reviewed raw025 audit; no GPU work or model imports."""
import argparse
import errno
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/Users/jeremylien/Projects/paretoquant-metal')
OUT = ROOT / 'results/qwen38-prefill-overlap-v1'
sys.path.insert(0, str(ROOT))
from benchmarks.qwen38_overlap import integration025 as integration
from benchmarks.qwen38_overlap import supervisor025 as supervisor

REVIEW = 'fbdfe4c1a74775a7e6e98623feddb52baf6d37644e644bc7805f710eaa39e246'
REFERENCE = '0196fc158afbe460a70eddced189a1659493d53ec05ba00e5364fcb606645ec6'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--phase', default='pilot', choices=supervisor.PHASES)
    args = parser.parse_args()
    trial = Path(args.trial).resolve(strict=True)
    assert trial.parent == OUT.resolve()
    raw = trial / 'raw'
    final = read(raw / 'result.json')
    if final.get('status') != 'success':
        publish(OUT / (trial.name + '-audit-rejected.json'), dict(status='controller_not_success',
            error_type=final.get('error_type'), error_stage=final.get('error_stage'),
            owned_cleanup=final.get('owned_actor_cleanup_verified'),
            services_closed=final.get('services_closed_verified'), physical_accepted=False))
        print(json.dumps({'status': 'controller_not_success', 'trial': str(trial)}))
        return 1
    freeze = read(OUT / 'parent-noncritical-freeze.json')
    manifest = supervisor.read_freeze(freeze['freeze_path'], freeze['freeze_sha256'])
    assert supervisor.expected_pressure_policy(manifest) == 'noncritical'
    assert digest(OUT / 'noncritical-combined-independent-review.json') == REVIEW
    assert read(OUT / 'noncritical-combined-independent-review.json')['passed'] is True
    reference_path = OUT / 'parent-noncritical-native-reference.json'
    assert digest(reference_path) == REFERENCE
    reference = read(reference_path)
    assert reference['manifest_sha256'] == freeze['freeze_sha256']
    for relative, expected in reference['reference_artifact_sha256'].items():
        assert digest(ROOT / relative) == expected
    authorization = read(OUT / (trial.name + '-parent-admission.json'))
    assert authorization['approved'] is True
    assert authorization['manifest_sha256'] == freeze['freeze_sha256']
    assert authorization['phase'] == args.phase
    assert authorization['review_receipt_sha256'] == REVIEW
    assert authorization['pressure_policy'] == 'noncritical'
    deployment = read(OUT / 'parent-noncritical-deployment.json')
    profile = read(deployment['private_profile_path'])
    hostfile = read(raw / 'hostfile.json')
    endpoints = []
    for rid, expected_address in enumerate([profile['local_address'], profile['remote_address']]):
        address, port = hostfile[rid][0].rsplit(':', 1)
        assert address == expected_address
        endpoints.append([address, int(port)])
    server_ready = read(raw / 'server.ready.json')['ready']['address']
    assert server_ready[0] == profile['remote_address']
    endpoints.append([profile['remote_address'], server_ready[1]])
    assert len({tuple(v) for v in endpoints}) == 3
    closures = []
    for address, port in endpoints:
        with socket.socket() as connection:
            connection.settimeout(2)
            try:
                connection.connect((address, port))
                closures.append(False)
            except OSError as exc:
                closures.append(exc.errno == errno.ECONNREFUSED)
    assert closures == [True, True, True]
    actors = read(raw / 'actors.json')
    contexts = {}
    for name in ['rank0', 'rank1', 'server']:
        job = read(raw / (name + '.job.json'))
        actor = read(raw / (name + '.receipt.json'))
        start, pids, result = actor['startup'], actor['pids'], actor['result']
        assert actors[name]['nonce'] == job['nonce'] == actor['nonce']
        host = 'local' if name == 'rank0' else 'remote'
        assert job['host_role'] == host and job['actor_role'] == name
        assert job['pressure_policy'] == 'noncritical'
        assert start['actor_pid'] == pids['actor_pid'] == result['actor_pid']
        assert pids['child_pid'] == result['child_pid']
        contexts[name] = dict(nonce=job['nonce'], actor_pid=start['actor_pid'],
            child_pid=pids['child_pid'], host_role=host, actor_role=name,
            actor_started_monotonic=start['actor_started_monotonic'],
            execution=result['execution'], elapsed=result['elapsed'], pressure_policy='noncritical')
    refrows = reference['phases'][args.phase]
    trust = dict(schema='qwen38-overlap-raw025-trusted-v2', artifact_kind=integration.PHYSICAL_KIND,
        phase=args.phase, manifest_sha256=freeze['freeze_sha256'], pressure_policy='noncritical',
        raw_artifacts={name: digest(raw / name) for name in integration.RAW_REQUIRED},
        prompt_token_ids=[row['prompt_token_ids'] for row in refrows],
        native_outputs=[row['native_output'] for row in refrows],
        tensor_sha256_by_rank=reference['tensor_sha256_by_rank'], actor_contexts=contexts,
        listener_endpoints=endpoints,
        parent_review=dict(approved=True, phase=args.phase, manifest_sha256=freeze['freeze_sha256'],
                           review_receipt_sha256=REVIEW))
    trusted_path = OUT / (trial.name + '-external-trust.json')
    publish(trusted_path, trust)
    sealed = read(trusted_path)
    assert sealed == trust
    audit = integration.audit_raw_trial(raw, sealed, manifest)
    report = integration.collect_raw_trials([(raw, sealed)], manifest)
    assert audit['physical_accepted'] is True
    if args.phase == 'pilot':
        assert report['physical_evidence_accepted'] is True
    # A single batch has valid physical phase evidence but is not the full84 population.
    assert len(audit['rows']) == manifest['phases'][args.phase]['requests']
    assert len(report['rows']) == manifest['phases'][args.phase]['requests']
    audit_path = OUT / (trial.name + '-raw-audit.json')
    report_path = OUT / (trial.name + '-report.json')
    publish(audit_path, audit)
    publish(report_path, report)
    summary = dict(schema='parent-noncritical-trial-audit-summary-v1', timestamp=time.time(),
        trial_path=str(trial), trusted_path=str(trusted_path), trusted_sha256=digest(trusted_path),
        audit_path=str(audit_path), audit_sha256=digest(audit_path),
        report_path=str(report_path), report_sha256=digest(report_path),
        rows_verified=len(report['rows']), phase_physical_accepted=audit['physical_accepted'], physical_evidence_accepted=report['physical_evidence_accepted'],
        complete_suite=report['complete_suite'], coverage_complete=report.get('coverage_complete'),
        statistical_targets_met=report.get('statistical_targets_met'),
        pilot_budget=audit.get('pilot_budget'), pressure_observations=audit.get('pressure_observations'),
        explicit_listener_refusals=closures)
    publish(OUT / (trial.name + '-audit-summary.json'), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
