"""Guard: the files that train and evaluate a model equal the original repository.

The probe reads models that `cesar-claros/csf_dynamics` trained. `ORIGIN.json` holds
the commit of that repository and the SHA-256 of each pipeline file at that commit.
If a change to one of these files is intended, record the new hash in `ORIGIN.json`,
and say in the protocol why the saved models stay valid.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = json.loads((ROOT / "ORIGIN.json").read_text())
PIPELINE = ORIGIN["pipeline_files"]


@pytest.mark.parametrize("name", sorted(PIPELINE))
def test_a_pipeline_file_equals_the_original(name: str) -> None:
    digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert digest == PIPELINE[name], (
        f"{name} differs from csf_dynamics {ORIGIN['source_commit'][:7]}")


def test_the_record_covers_every_module_that_a_run_loads() -> None:
    code = ("import sys, dynamics.run_stage; print('\\n'.join(sorted("
            "m for m in sys.modules if m.startswith('dynamics.'))))")
    found = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.split()
    loaded = {module.replace(".", "/") + ".py" for module in found}
    assert loaded == set(PIPELINE)
