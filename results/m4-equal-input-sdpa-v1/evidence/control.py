"""Reuse the verified owned diagnostic controller in a distinct deployment."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent
path = ROOT.parent / 'control_trace.py'
spec = importlib.util.spec_from_file_location('owned_trace_controller', path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.LOCAL = ROOT
if __name__ == '__main__':
    {'deploy':driver.deploy, 'run':driver.run}[sys.argv[1]]()
