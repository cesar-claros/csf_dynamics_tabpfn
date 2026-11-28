"""Fit a learner on exported tables and write its predictions (HPC).

The tables come from `python -m dynamics.analyze_tabrule export` of the main
repository (`csf_dynamics`). Each fit has the inputs and the targets of its rows,
a weight for each row, and the inputs of its test rows. This script knows only the
tables. It fits one regressor for each target column and writes the predictions of
the test rows. The main repository turns them into choices.

    python -m dynamics.fit_tables --tables rule_tables.npz --out rule_predictions.npz

Protocol: ../documentation/tabpfn_rule_check_2026-11-27.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SEED = 0
#: The penalty of the ridge rule of the main repository.
PENALTY = 10.0
#: TabPFN learners: the model version, and if the fit gets the row weights.
TABPFN = {"tabpfn_v3.5": ("V3_5", False), "tabpfn_v2.5": ("V2_5", False),
          "tabpfn_v3.5_weighted": ("V3_5", True)}
LEARNERS = (*TABPFN, "forest", "ridge")


def ridge(x_fit: np.ndarray, y_fit: np.ndarray, weights: np.ndarray,
          x_test: np.ndarray, penalty: float = PENALTY) -> np.ndarray:
    """The weighted ridge regression of the rule, for every target at once."""
    w = weights * len(weights) / weights.sum()
    offset = (w[:, None] * y_fit).sum(axis=0) / w.sum()
    beta = np.linalg.solve(
        x_fit.T @ (w[:, None] * x_fit) + penalty * np.eye(x_fit.shape[1]),
        x_fit.T @ (w[:, None] * (y_fit - offset)))
    return x_test @ beta + offset


def regressor(name: str, device: str) -> Any:
    """An unfitted regressor for one target column."""
    if name in TABPFN:
        # tabpfn builds the network again at each fit unless this cache is on.
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "3")
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        return TabPFNRegressor.create_default_for_version(
            getattr(ModelVersion, TABPFN[name][0]), device=device, random_state=SEED)
    if name == "forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=500, random_state=SEED)
    raise ValueError(f"unknown learner: {name}")


def predictions(name: str, x_fit: np.ndarray, y_fit: np.ndarray, weights: np.ndarray,
                x_test: np.ndarray, device: str) -> np.ndarray:
    """The predictions of one learner for the test rows, one column for each target.

    All test rows of a fit go into one call, so that a learner whose output depends
    on the other rows of the call sees the same rows for every target.
    """
    if name == "ridge":
        return ridge(x_fit, y_fit, weights, x_test)
    out = np.empty((len(x_test), y_fit.shape[1]))
    for column in range(y_fit.shape[1]):
        model = regressor(name, device)
        if name in TABPFN and TABPFN[name][1]:
            model.fit(x_fit, y_fit[:, column], sample_weight=weights)
        else:
            model.fit(x_fit, y_fit[:, column])
        out[:, column] = np.asarray(model.predict(x_test), dtype=np.float64)
    return out


def folds_of(tables: Any) -> list[tuple[str, int]]:
    """Every fit of the tables file, as its bed and its number."""
    manifest = json.loads(str(tables["manifest"]))
    return [(bed, number) for bed, content in manifest.items()
            for number in range(len(content["folds"]))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--learners", nargs="+", default=list(LEARNERS),
                        choices=LEARNERS)
    parser.add_argument("--beds", nargs="+", help="default: every bed of the tables")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()
    tables = np.load(args.tables, allow_pickle=False)
    content = str(tables["content_sha256"])
    out = Path(args.out)
    done: dict[str, np.ndarray] = {}
    if out.exists():
        saved = np.load(out, allow_pickle=False)
        if str(saved["content_sha256"]) != content:
            raise SystemExit(f"{out.name} holds predictions for other tables")
        done = {key: saved[key] for key in saved.files
                if key not in ("content_sha256", "learners")}
    wanted = [(bed, number) for bed, number in folds_of(tables)
              if not args.beds or bed in args.beds]
    learners = sorted({key.split(".")[0] for key in done} | set(args.learners),
                      key=LEARNERS.index)
    began = time.time()
    for name in args.learners:
        started, fits = time.time(), 0
        for bed, number in wanted:
            key = f"{name}.{bed}.{number:03d}"
            if key in done:
                continue
            part = {item: tables[f"{bed}.{number:03d}.{item}"]
                    for item in ("x_fit", "y_fit", "weights", "x_test")}
            done[key] = predictions(name, part["x_fit"], part["y_fit"],
                                    part["weights"], part["x_test"], args.device)
            fits += 1
            np.savez_compressed(out, content_sha256=np.array(content),  # type: ignore[arg-type]
                                learners=np.array(learners), **done)
            print(f"[fit] {key}: {len(part['x_fit'])} rows, "
                  f"{part['y_fit'].shape[1]} targets, {len(part['x_test'])} test "
                  f"rows, {time.time() - started:.0f}s for this learner",
                  flush=True)
        print(f"[fit] {name}: {fits} new fits of {len(wanted)} in "
              f"{time.time() - started:.0f}s", flush=True)
    np.savez_compressed(out, content_sha256=np.array(content),  # type: ignore[arg-type]
                        learners=np.array(learners), **done)
    print(f"[fit] done in {time.time() - began:.0f}s: {out}", flush=True)


if __name__ == "__main__":
    main()
