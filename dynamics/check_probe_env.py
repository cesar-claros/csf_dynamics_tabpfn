"""Environment gate of the tabular probe: packages, GPU, licence, and weights.

    python -m dynamics.check_probe_env            # packages and GPU
    python -m dynamics.check_probe_env --load     # also load each model version

Protocol: ../documentation/tabular_prior_probe_protocol_2026-11-17.md
"""
from __future__ import annotations

import argparse
import importlib
import os

#: Import name to pip name, for `tabpfn` and the packages that it needs.
PACKAGES = {
    "tabpfn": "tabpfn", "einops": "einops", "safetensors": "safetensors",
    "huggingface_hub": "huggingface-hub", "pydantic": "pydantic",
    "pydantic_core": "pydantic-core", "annotated_types": "annotated-types",
    "typing_inspection": "typing-inspection", "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv", "lightgbm": "lightgbm", "skrub": "skrub",
    "filelock": "filelock", "joblib": "joblib", "pandas": "pandas", "scipy": "scipy",
    "sklearn": "scikit-learn", "matplotlib": "matplotlib"}


def missing_packages() -> list[str]:
    """Pip names of the packages that cannot be imported."""
    missing = []
    for module, name in PACKAGES.items():
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 - any import failure counts
            missing.append(name)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--load", action="store_true",
                        help="load each model version and fit a small table")
    args = parser.parse_args()
    import torch

    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}")
    missing = missing_packages()
    if missing:
        print("missing packages: " + " ".join(missing))
        print("install them with:\n  python -m pip install --no-deps --target "
              "$DYN_ENV " + " ".join(missing))
        raise SystemExit(1)
    import tabpfn

    print(f"tabpfn {tabpfn.__version__}; TABPFN_TOKEN set: "
          f"{bool(os.environ.get('TABPFN_TOKEN'))}")
    if not args.load:
        print("ok")
        return
    import numpy as np

    from dynamics.tabular_probe import VERSIONS, learner

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16))
    y = (x[:, 0] > 0).astype(int)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failed = []
    for name in VERSIONS:
        try:
            model = learner(name, device)()
            model.fit(x[:150], y[:150])
            accuracy = float((model.predict_proba(x[150:]).argmax(axis=1)
                              == y[150:]).mean())
            print(f"{name}: loaded, accuracy {accuracy:.2f} on a toy table")
        except Exception as error:  # noqa: BLE001 - report every version
            failed.append(name)
            print(f"{name}: FAILED: {type(error).__name__}: {error}")
    if failed:
        raise SystemExit(1)
    print("ok")


if __name__ == "__main__":
    main()
