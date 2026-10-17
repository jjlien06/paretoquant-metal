"""Parent-owned sequential six-batch benchmark, audited before each next batch."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/Users/jeremylien/Projects/paretoquant-metal')
OUT = ROOT / 'results/qwen38-prefill-overlap-v1'
sys.path.insert(0, str(ROOT))
from benchmarks.qwen38_overlap import integration025 as integration
from benchmarks.qwen38_overlap import supervisor025 as supervisor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    os.environ['TMPDIR'] = '/Users/jeremylien/.hermes/cache/scratch'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PARETOQUANT_COMBINED_REVIEW_SHA256'] = 'fbdfe4c1a74775a7e6e98623feddb52baf6d37644e644bc7805f710eaa39e246'
    freeze = read(OUT / 'parent-noncritical-freeze.json')
    manifest = supervisor.read_freeze(freeze['freeze_path'], freeze['freeze_sha256'])
    pilot = OUT / 'physical-pilot-noncritical-v3-1792272408272316000-audit-summary.json'
    admitted = read(pilot)
    assert admitted['physical_evidence_accepted'] is True
    assert admitted['pilot_budget']['passed'] is True
    assert sha(admitted['trusted_path']) == admitted['trusted_sha256']
    folder = OUT / ('suite-noncritical-v1-' + str(time.time_ns()))
    folder.mkdir(mode=0o700)
    phases = [f'batch-{n:02d}' for n in range(6)]
    python = str(ROOT / '.venv/bin/python')
    runner = OUT / 'parent-run-batch-noncritical-v1.py'
    auditor = OUT / 'parent-audit-noncritical-trial-v2.py'
    publish(folder / 'plan.json', dict(schema='parent-full-suite-execution-plan-v1',
        manifest_sha256=freeze['freeze_sha256'], pressure_policy='noncritical',
        memory_acceptance_sha256=manifest['memory_acceptance']['sha256'], phases=phases,
        planned_requests=sum(manifest['phases'][p]['requests'] for p in phases),
        pilot_excluded=True, runner_sha256=sha(runner), auditor_sha256=sha(auditor),
        controller_sha256=sha(__file__), pilot_audit_summary_sha256=sha(pilot),
        sequential_hardware_owner=True, performance_accepted=False))
    completed = []
    for phase in phases:
        print(json.dumps({'phase': phase, 'status': 'starting', 'suite_path': str(folder)}), flush=True)
        try:
            proc = subprocess.run([python, '-B', str(runner), '--phase', phase], cwd=ROOT,
                capture_output=True, timeout=1000)
            with (folder / (phase + '-execution.stdout')).open('xb') as stream:
                stream.write(proc.stdout)
            with (folder / (phase + '-execution.stderr')).open('xb') as stream:
                stream.write(proc.stderr)
            assert proc.returncode == 0, 'bounded phase execution failed'
            execution = json.loads(proc.stdout)
            trial = Path(execution['trial_path'])
            assert trial.parent == OUT and execution['phase'] == phase
            assert execution['controller_result']['status'] == 'success'
            audit = subprocess.run([python, '-B', str(auditor), '--trial', str(trial), '--phase', phase],
                cwd=ROOT, capture_output=True, timeout=120)
            with (folder / (phase + '-audit.stdout')).open('xb') as stream:
                stream.write(audit.stdout)
            with (folder / (phase + '-audit.stderr')).open('xb') as stream:
                stream.write(audit.stderr)
            assert audit.returncode == 0, 'physical phase receipt audit failed'
            summary = json.loads(audit.stdout)
            assert summary['phase_physical_accepted'] is True and summary['rows_verified'] == 14
            assert summary['explicit_listener_refusals'] == [True, True, True]
            assert sha(summary['trusted_path']) == summary['trusted_sha256']
            completed.append(dict(phase=phase, trial_path=str(trial), audit_summary=summary))
            publish(folder / (phase + '-complete.json'), completed[-1])
            print(json.dumps({'phase': phase, 'status': 'physical_phase_audited', 'completed_requests': sum(v['audit_summary']['rows_verified'] for v in completed)}), flush=True)
        except Exception as exc:
            publish(folder / 'failed.json', dict(schema='parent-full-suite-stopped-v1',
                phase=phase, error_type=type(exc).__name__, completed=completed,
                complete_suite=False, performance_accepted=False,
                scope='Preserve exact execution/audit stdout and raw trials; no automatic retry.'))
            print(json.dumps({'status': 'suite_stopped', 'phase': phase, 'error_type': type(exc).__name__, 'suite_path': str(folder)}), flush=True)
            return 1
    assert len(completed) == len(phases)
    trials = []
    for item in completed:
        summary = item['audit_summary']
        assert sha(summary['trusted_path']) == summary['trusted_sha256']
        trials.append((Path(item['trial_path']) / 'raw', read(summary['trusted_path'])))
    report = integration.collect_raw_trials(trials, manifest)
    assert len(report['rows']) == 84 and report['coverage_complete'] is True
    assert report['physical_evidence_accepted'] is True and report['complete_suite'] is True
    assert len(report['cases']) == 18
    publish(folder / 'report.json', report)
    fields = ['case_id', 'formatted_prompt_tokens', 'target_long', 'ttft_reduction_percent',
              'request_reduction_percent', 'thresholds_met', 'serial_median_ttft_seconds',
              'overlap_median_ttft_seconds', 'serial_median_wall_seconds', 'overlap_median_wall_seconds']
    case_table = []
    for case in report['cases']:
        rows = [v for v in report['rows'] if v['case_id'] == case['case_id'] and not v['warmup']]
        cell = {k: case[k] for k in ['case_id', 'formatted_prompt_tokens', 'target_long', 'thresholds_met']}
        cell.update(ttft_reduction_percent=100*case['first_token_seconds']['reduction'],
                    request_reduction_percent=100*case['wall_seconds']['reduction'])
        for mode in ['serial','overlap']:
            selected = [v for v in rows if v['prefill_mode'] == mode]
            assert len(selected) == 2
            cell[mode + '_median_ttft_seconds'] = statistics.median(v['first_token_seconds'] for v in selected)
            cell[mode + '_median_wall_seconds'] = statistics.median(v['wall_seconds'] for v in selected)
        case_table.append(cell)
    with (folder / 'cases.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(case_table)
    publish(folder / 'case-table.json', case_table)
    summary = dict(schema='parent-full-suite-summary-v1', timestamp=time.time(),
        status='complete_targets_met' if report['accepted'] else 'complete_targets_not_met',
        suite_path=str(folder), manifest_sha256=freeze['freeze_sha256'],
        report_sha256=sha(folder/'report.json'), cases_csv_sha256=sha(folder/'cases.csv'),
        completed_requests=len(report['rows']), completed_cases=len(report['cases']),
        physical_evidence_accepted=report['physical_evidence_accepted'],
        statistical_targets_met=report['statistical_targets_met'], accepted=report['accepted'],
        cases=case_table, regressions=report['regressions'],
        pressure_observations=report['pressure_observations'],
        scope='Whole full-suite measurements; original native IDs/stops, small/full numerical/cache and ownership checks audited. Warning2 allowed by explicit user amendment; not normal-pressure qualification. Final independent result review still separate.')
    publish(folder / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if report['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
