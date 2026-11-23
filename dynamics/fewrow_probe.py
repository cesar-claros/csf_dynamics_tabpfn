"""Few-row probe: OOD scores that are fitted on a few ID rows of each class (HPC).

For each run, at its last checkpoint, and for each row count n and each draw: draw
n training rows of each class, fit the feature scores of the roster on these rows,
and compute the one-class score D1 with each learner on the same rows. The data-free
scores of the roster need no rows, so they are the same in every arm. One more arm
fits the roster and the controls on all training rows.

    python -m dynamics.fewrow_probe --only A_seed5 --out-root $DYN_ROOT ...

Protocol: ../documentation/few_row_probe_protocol_2026-11-23.md
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from dynamics.config import DEFAULT_PRECISION, PENULTIMATE, DataPaths, RunSpec, run_dir
from dynamics.scores import all_scores
from dynamics.tabular_probe import (
    CONTROLS,
    SEED,
    VERSIONS,
    Prepared,
    add_common_arguments,
    class_scores,
    learner,
    metric_rows,
    prepare,
    selected,
    tied_pairs,
)

ROW_COUNTS = (10, 30, 100)
PRIMARY_ROWS = 10
DRAWS = 3
#: The learner of the verdict. It has no probability floor, so it has no ties.
PRIMARY = "tabpfn_v2.5"
#: Row count of the arm that uses every training row.
ALL_ROWS = 0


def few_rows(labels: np.ndarray, count: int, n_classes: int, draw: int,
             seed: int = SEED) -> np.ndarray:
    """`count` rows of each class, fixed by the count and the draw. Sorted."""
    rng = np.random.default_rng([seed, 20_000, count, draw])
    return np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(labels == c))[:count]
        for c in range(n_classes)]))


def with_ties(rows: list[dict], scores: dict[str, np.ndarray],
              layout: dict[str, slice]) -> list[dict]:
    """Add to each metric row the share of tied pairs of ID and OOD rows."""
    for row in rows:
        values = scores[row["score"]]
        row["tied"] = tied_pairs(values[layout["val"]], values[layout[row["set"]]])
    return rows


def arm(ready: Prepared, chosen: np.ndarray | None, learners: list[str],
        device: torch.device, seed: int) -> dict:
    """The roster and D1 of each learner, fitted on the rows in `chosen`.

    With `chosen` None, the fit uses every training row.
    """
    train = ready.train
    pick = slice(None) if chosen is None else torch.as_tensor(chosen)
    features, labels = train[PENULTIMATE][pick], train["labels"][pick]
    roster, errors = all_scores(features, labels, train["logits"][pick], ready.w,
                                ready.b, ready.probe, ready.logits, ready.n_classes,
                                device, DEFAULT_PRECISION)
    scores: dict[str, np.ndarray] = dict(roster)
    x_fit, y_fit = features.double().numpy(), labels.numpy()
    x_probe = ready.probe.double().numpy()
    predicted = ready.logits.argmax(dim=1).numpy()
    timing: dict[str, float] = {}
    for name in learners:
        began = time.time()
        try:
            table = class_scores(learner(name, str(device)), x_fit, y_fit, x_probe,
                                 ready.n_classes, seed=seed)
            scores[f"D1:{name}"] = table[np.arange(len(table)), predicted]
            scores[f"D1max:{name}"] = table.max(axis=1)
        except Exception as error:  # noqa: BLE001 - keep the other scores
            errors[f"D1:{name}"] = f"{type(error).__name__}: {error}"
            print(traceback.format_exc(), flush=True)
        timing[f"D1:{name}"] = round(time.time() - began, 1)
    finite = {name: values for name, values in scores.items()
              if np.isfinite(values).all()}
    for name in scores.keys() - finite.keys():
        errors.setdefault(name, "a value is not finite")
    return {"rows_per_class": ALL_ROWS if chosen is None else len(chosen)
            // ready.n_classes,
            "metrics": with_ties(metric_rows(finite, ready.layout, ready.correct),
                                 finite, ready.layout),
            "errors": errors, "timing": timing}


def fewrow_run(spec: RunSpec, paths: DataPaths, out_root: Path, device: torch.device,
               learners: list[str], full_learners: list[str],
               row_counts: list[int], draws: int, workers: int,
               image_size: int, limit: int | None) -> dict:
    """Every arm of one run, as a JSON-ready dict."""
    ready = prepare(spec, paths, out_root, device, workers, image_size, limit)
    labels = ready.train["labels"].numpy()
    smallest = int(np.bincount(labels, minlength=ready.n_classes).min())
    arms = []
    for count in row_counts:
        if count > smallest:
            raise ValueError(f"{count} rows for each class, and a class has "
                             f"{smallest}")
        for draw in range(draws):
            out = arm(ready, few_rows(labels, count, ready.n_classes, draw), learners,
                      device, seed=1_000 * count + draw)
            out["draw"] = draw
            arms.append(out)
    full = arm(ready, None, full_learners, device, seed=SEED)
    full["draw"] = 0
    arms.append(full)
    return {"run": spec.name, "epoch": ready.epoch, "n_classes": ready.n_classes,
            "rows": {name: s.stop - s.start for name, s in ready.layout.items()},
            "settings": {"row_counts": row_counts, "draws": draws,
                         "learners": learners, "full_learners": full_learners,
                         "source": "train", "seed": SEED},
            "arms": arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--row-counts", type=int, nargs="+", default=list(ROW_COUNTS),
                        help="rows of each class in the few-row arms")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--learners", nargs="*", default=[*VERSIONS, *CONTROLS],
                        help="learners of D1 in the few-row arms")
    parser.add_argument("--full-learners", nargs="*", default=list(CONTROLS),
                        help="learners of D1 in the arm with all training rows")
    args = parser.parse_args()
    specs, paths = selected(args, parser)
    out_root, device = Path(args.out_root), torch.device(args.device)
    probe_root = Path(args.probe_root) if args.probe_root else out_root
    failures = []
    for spec in specs:
        target = run_dir(probe_root, spec.name) / "fewrow" / "results.json"
        if target.exists() and not args.force:
            print(f"[fewrow] {spec.name}: done, skipped", flush=True)
            continue
        began = time.time()
        try:
            out = fewrow_run(spec, paths, out_root, device, args.learners,
                             args.full_learners, args.row_counts, args.draws,
                             args.workers, args.image_size, args.limit)
        except Exception:  # noqa: BLE001 - continue with the next run
            failures.append(spec.name)
            print(f"[fewrow] {spec.name} FAILED\n{traceback.format_exc()}",
                  flush=True)
            continue
        out["seconds"] = round(time.time() - began, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out))
        errors = sum(len(one["errors"]) for one in out["arms"])
        print(f"[fewrow] {spec.name} epoch {out['epoch']}: {len(out['arms'])} arms, "
              f"{errors} errors, {out['seconds']}s", flush=True)
    print("[fewrow] failures: " + (", ".join(failures) if failures else "none"),
          flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
