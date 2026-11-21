"""Probe: OOD scores from a tabular foundation model on classifier features (HPC).

For each run, at its last checkpoint: extract the penultimate features, draw the
fixed probe rows, and compute the 23 roster scores and the new scores on them.

Direction 1 (one class at a time): for each class, a binary learner separates the
training rows of the class from a background with independent columns. D1 is the
probability of "class" under the context of the predicted class, and D1max is the
largest such probability over the classes.

Direction 2 (label swing): the test row enters a small context with each label in
turn. D2 is the negative of log(sum over labels of the probability of the forced
label). D2-0 is the largest class probability with no added row.

Each score is stored in a form that keeps the order of the rows where a
probability rounds to 1 (protocol, run record of 2026-11-21): the log-odds for D1
and D1max, minus the log of the probability of the other classes for D2-0, and D2
plus the constant log(number of classes). Each form is a monotone function of the
definition above. The TabPFN probabilities come from the raw logits in double
precision, because `predict_proba` works in single precision.

    python -m dynamics.tabular_probe --only A_seed5 --out-root $DYN_ROOT ...

Protocol: ../documentation/tabular_prior_probe_protocol_2026-11-17.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import (
    DEFAULT_PRECISION,
    IMAGE_SIZE,
    PENULTIMATE,
    STAGES,
    DataPaths,
    RunSpec,
    checkpoint_path,
    parse_run,
    run_dir,
)
from dynamics.data import evaluation_sets
from dynamics.evaluate import load_model
from dynamics.metrics import set_metrics
from dynamics.model import extract
from dynamics.scores import all_scores

NEAR_SETS = ("ssb_hard", "ninco", "imagewoof")
OOD_SETS = (*FAR_SETS, *NEAR_SETS)
#: Learner name to the model version of the `tabpfn` package.
VERSIONS = {"tabpfn_v3.5": "V3_5", "tabpfn_v3": "V3", "tabpfn_v2.5": "V2_5"}
PRIMARY = "tabpfn_v3.5"
CONTROLS = ("trees", "logistic")
SEED = 0
ROWS_PER_SET = 500
SWING_ROWS = 250
CONTEXT_PER_CLASS = 100
SWING_COLUMNS = (512, 128)
#: Largest difference between the double precision probabilities and `predict_proba`
#: that counts as agreement. `predict_proba` works in single precision.
PRECISION_TOLERANCE = 1e-4
#: Rows of the comparison with `predict_proba`.
PRECISION_ROWS = 256
#: Smallest positive double, the floor under a logarithm.
TINY = float(np.finfo(np.float64).tiny)
#: Marks results whose scores keep the order of the rows with a probability near 1.
SCORE_FORMS = "double precision: log-odds, rest of top class, D2 plus log(classes)"
Learner = Callable[[], Any]


def learner(name: str, device: str, swing: bool = False) -> Learner:
    """A factory of unfitted classifiers with `fit` and `predict_proba`.

    The label swing uses one estimator, as the protocol fixes.
    """
    if name in VERSIONS:
        # tabpfn builds the network again at each fit unless this cache is on.
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", str(len(VERSIONS)))

        def make() -> Any:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion

            options: dict[str, Any] = {"device": device, "random_state": SEED}
            if swing:
                options["n_estimators"] = 1
            return TabPFNClassifier.create_default_for_version(
                getattr(ModelVersion, VERSIONS[name]), **options)
        return make
    if name == "trees":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return lambda: HistGradientBoostingClassifier(random_state=SEED)
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return lambda: make_pipeline(StandardScaler(),
                                     LogisticRegression(C=1.0, max_iter=1000))
    raise ValueError(f"unknown learner: {name}")


def probe_rows(sizes: dict[str, int], val_labels: np.ndarray, n_classes: int,
               per_set: int = ROWS_PER_SET, seed: int = SEED
               ) -> dict[str, np.ndarray]:
    """Fixed rows of each set: `val` by class, every other set at random. Sorted."""
    out = {}
    for position, (name, size) in enumerate(sorted(sizes.items())):
        rng = np.random.default_rng([seed, position])
        if name == "val":
            share = per_set // n_classes
            rows = np.concatenate([
                rng.permutation(np.flatnonzero(val_labels == c))[:share]
                for c in range(n_classes)])
        else:
            rows = rng.choice(size, size=min(per_set, size), replace=False)
        out[name] = np.sort(rows)
    return out


def background(train: np.ndarray, count: int, rng: np.random.Generator
               ) -> np.ndarray:
    """Rows in which each column is drawn on its own from the training rows."""
    rows = rng.integers(0, len(train), size=(count, train.shape[1]))
    return train[rows, np.arange(train.shape[1])]


def probabilities(model: Any, rows: np.ndarray, gaps: list[float] | None = None
                  ) -> np.ndarray:
    """Class probabilities of `rows` in double precision, in the order of `classes_`.

    TabPFN applies its softmax in single precision, so a confident model gives
    exactly 1 to many rows, and their order is lost. Here the raw logits go through
    the post-processing of the model itself as a double tensor. With `gaps`, the
    first rows are compared with `predict_proba`: the largest difference is
    appended, and a difference above the tolerance gives `predict_proba` for all
    rows. A learner with no raw logits gives `predict_proba`.
    """
    raw_logits = getattr(model, "predict_raw_logits", None)
    convert = getattr(model, "logits_to_probabilities", None)
    if raw_logits is None or convert is None:
        return np.asarray(model.predict_proba(rows), dtype=np.float64)
    raw = torch.from_numpy(np.asarray(raw_logits(rows), dtype=np.float64))
    exact = np.asarray(convert(raw).detach().cpu().numpy(), dtype=np.float64)
    if gaps is not None:
        part = rows[:PRECISION_ROWS]
        single = np.asarray(model.predict_proba(part), dtype=np.float64)
        same_shape = exact[:len(part)].shape == single.shape
        gap = (float(np.abs(exact[:len(part)] - single).max()) if same_shape
               else float("inf"))
        gaps.append(gap)
        if not gap <= PRECISION_TOLERANCE:
            return np.asarray(model.predict_proba(rows), dtype=np.float64)
    return exact


def log_odds(model: Any, rows: np.ndarray, gaps: list[float] | None = None
             ) -> np.ndarray:
    """Log-odds of class 1 against class 0 of a fitted binary learner.

    The value is a monotone function of the probability of class 1. It keeps the
    order of the rows whose probability rounds to 1.
    """
    if not hasattr(model, "predict_raw_logits") and hasattr(model,
                                                            "decision_function"):
        return np.asarray(model.decision_function(rows), dtype=np.float64)
    table = probabilities(model, rows, gaps)
    classes = list(model.classes_)
    return (np.log(np.maximum(table[:, classes.index(1)], TINY))
            - np.log(np.maximum(table[:, classes.index(0)], TINY)))


def class_scores(make: Learner, train: np.ndarray, labels: np.ndarray,
                 queries: np.ndarray, n_classes: int, seed: int = SEED,
                 gaps: list[float] | None = None) -> np.ndarray:
    """Per query and class: the log-odds of "class" against the background."""
    out = np.empty((len(queries), n_classes))
    for c in range(n_classes):
        positive = train[labels == c]
        reference = background(train, len(positive),
                               np.random.default_rng([seed, c]))
        model = make()
        model.fit(np.vstack([positive, reference]),
                  np.r_[np.ones(len(positive), int), np.zeros(len(reference), int)])
        out[:, c] = log_odds(model, queries, gaps)
    return out


def balanced_context(labels: np.ndarray, n_classes: int,
                     per_class: int = CONTEXT_PER_CLASS, seed: int = SEED
                     ) -> np.ndarray:
    """Rows of a context with the same number of rows of each class. Sorted."""
    rng = np.random.default_rng([seed, 10_000])
    return np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(labels == c))[:per_class]
        for c in range(n_classes)]))


def rest_of(table: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Per row: the summed probability of every class except the one in `kept`."""
    others = table.copy()
    others[np.arange(len(table)), kept] = 0.0
    return others.sum(axis=1)


def plain_scores(make: Learner, context: np.ndarray, labels: np.ndarray,
                 queries: np.ndarray, gaps: list[float] | None = None) -> np.ndarray:
    """D2-0: minus the log of the probability that the top class leaves to the others.

    The value is a monotone function of the largest class probability with no added
    row. It keeps the order of the rows whose top probability rounds to 1.
    """
    model = make()
    model.fit(context, labels)
    table = probabilities(model, queries, gaps)
    rest = rest_of(table, table.argmax(axis=1))
    return -np.log(np.maximum(rest, TINY))


def swing_scores(make: Learner, context: np.ndarray, labels: np.ndarray,
                 queries: np.ndarray, n_classes: int) -> np.ndarray:
    """D2 plus log(n_classes): minus log(mean over labels of the forced probability).

    Each query enters the context once for each label. The value is log(n_classes)
    if only one label can be given to the row, and 0 if every label is accepted.
    The sum of the forced probabilities is close to n_classes for most rows, so the
    code adds up the deficits, which keep the order of those rows.
    """
    out = np.empty(len(queries))
    for i, row in enumerate(queries):
        deficit = 0.0
        for y in range(n_classes):
            model = make()
            model.fit(np.vstack([context, row]), np.r_[labels, y])
            table = probabilities(model, row[None])
            forced = np.array([list(model.classes_).index(y)])
            deficit += float(rest_of(table, forced)[0])
        out[i] = -np.log1p(-min(deficit / n_classes, 1.0 - 1e-16))
    return out


def metric_rows(scores: dict[str, np.ndarray], layout: dict[str, slice],
                correct: np.ndarray) -> list[dict]:
    """Metrics of every score for each OOD set against the validation rows."""
    rows = []
    for name, values in scores.items():
        for ood in (s for s in layout if s != "val"):
            rows.append({"score": name, "set": ood, **set_metrics(
                values[layout["val"]], correct, values[layout[ood]])})
    return rows


def reduce_columns(train: np.ndarray, columns: int) -> Callable[[np.ndarray], Any]:
    """The identity, or a PCA to `columns` components fitted on the training rows."""
    if columns >= train.shape[1]:
        return lambda rows: rows
    from sklearn.decomposition import PCA

    return PCA(n_components=columns, random_state=SEED).fit(train).transform


def note_gaps(name: str, found: list[float], gaps: dict[str, float],
              errors: dict[str, str]) -> None:
    """Record the comparison with `predict_proba`, and a fallback as an error."""
    if not found:
        return
    gaps[name] = max(found)
    if not gaps[name] <= PRECISION_TOLERANCE:
        errors[f"precision:{name}"] = (
            f"double precision differs from predict_proba by {gaps[name]:.3g}, "
            "so predict_proba was used")


def probe_run(spec: RunSpec, paths: DataPaths, out_root: Path, device: torch.device,
              learners: list[str], plain: list[str], swing: list[str],
              swing_columns: int, swing_rows: int, workers: int,
              image_size: int = IMAGE_SIZE, limit: int | None = None) -> dict:
    """All scores of one run on its probe rows, as a JSON-ready dict."""
    folder = run_dir(out_root, spec.name)
    epoch = max(int(p.stem.split("_")[1]) for p in (folder / "checkpoints").glob(
        "epoch_*.pt"))
    model, state = load_model(checkpoint_path(out_root, spec.name, epoch), device)
    n_classes = len(state["classes"])
    sets = evaluation_sets(paths, spec.held_out, image_size, limit, workers)
    wanted = ["train", "val", *[name for name in OOD_SETS if name in sets]]
    features = {name: extract(model, sets[name], device, workers=0)
                for name in wanted}
    train = features["train"]
    rows = probe_rows({name: len(features[name]["logits"]) for name in wanted[1:]},
                      features["val"]["labels"].numpy(), n_classes)
    order = ["val", *[name for name in wanted[2:]]]
    layout, start = {}, 0
    for name in order:
        layout[name] = slice(start, start + len(rows[name]))
        start += len(rows[name])
    pick = {name: torch.as_tensor(rows[name]) for name in order}
    probe = torch.cat([features[name][PENULTIMATE][pick[name]] for name in order])
    logits = torch.cat([features[name]["logits"][pick[name]] for name in order])
    val = features["val"]
    correct = (val["logits"].argmax(dim=1) == val["labels"]).numpy()[rows["val"]]
    w = model.fc.weight.detach().float().cpu()
    b = model.fc.bias.detach().float().cpu()
    roster, errors = all_scores(train[PENULTIMATE], train["labels"], train["logits"],
                                w, b, probe, logits, n_classes, device,
                                DEFAULT_PRECISION)
    scores: dict[str, np.ndarray] = dict(roster)
    x_train = train[PENULTIMATE].double().numpy()
    y_train = train["labels"].numpy()
    x_probe = probe.double().numpy()
    predicted = logits.argmax(dim=1).numpy()
    timing: dict[str, float] = {}
    gaps: dict[str, float] = {}
    for name in learners:
        began = time.time()
        found: list[float] = []
        try:
            table = class_scores(learner(name, str(device)), x_train, y_train,
                                 x_probe, n_classes, gaps=found)
            scores[f"D1:{name}"] = table[np.arange(len(table)), predicted]
            scores[f"D1max:{name}"] = table.max(axis=1)
        except Exception as error:  # noqa: BLE001 - keep the other scores
            errors[f"D1:{name}"] = f"{type(error).__name__}: {error}"
            print(traceback.format_exc(), flush=True)
        note_gaps(f"D1:{name}", found, gaps, errors)
        timing[f"D1:{name}"] = round(time.time() - began, 1)
    context = balanced_context(y_train, n_classes)
    reduce = reduce_columns(x_train, swing_columns)
    x_context, y_context = reduce(x_train[context]), y_train[context]
    for name in plain:
        began = time.time()
        found = []
        try:
            scores[f"D2-0:{name}"] = plain_scores(
                learner(name, str(device)), x_context, y_context, reduce(x_probe),
                gaps=found)
        except Exception as error:  # noqa: BLE001
            errors[f"D2-0:{name}"] = f"{type(error).__name__}: {error}"
        note_gaps(f"D2-0:{name}", found, gaps, errors)
        timing[f"D2-0:{name}"] = round(time.time() - began, 1)
    # The label swing runs on fewer rows: evenly spaced rows of val and of each
    # far set. The rows of val are in class order, so the classes stay balanced.
    swing_layout, swing_index, start = {}, [], 0
    for name in ("val", *[s for s in FAR_SETS if s in layout]):
        part = np.arange(layout[name].start, layout[name].stop)
        part = part[::max(1, len(part) // max(1, swing_rows))][:swing_rows]
        swing_layout[name] = slice(start, start + len(part))
        swing_index.append(part)
        start += len(part)
    chosen = np.concatenate(swing_index)
    swing_scores_by_name = {name: values[chosen] for name, values in scores.items()}
    for name in swing:
        began = time.time()
        try:
            swing_scores_by_name[f"D2:{name}"] = swing_scores(
                learner(name, str(device), swing=True), x_context, y_context,
                reduce(x_probe[chosen]), n_classes)
        except Exception as error:  # noqa: BLE001
            errors[f"D2:{name}"] = f"{type(error).__name__}: {error}"
            print(traceback.format_exc(), flush=True)
        timing[f"D2:{name}"] = round(time.time() - began, 1)
    np.savez_compressed(
        folder / "probe" / "scores.npz", names=np.array(list(scores)),
        values=np.stack(list(scores.values())).astype(np.float64),
        sets=np.array(order), bounds=np.array([layout[name].stop for name in order]),
        correct=correct, swing_rows=chosen,
        swing_names=np.array(list(swing_scores_by_name)),
        swing_values=np.stack(list(swing_scores_by_name.values())).astype(np.float64))
    return {
        "run": spec.name, "epoch": epoch, "n_classes": n_classes,
        "rows": {name: len(rows[name]) for name in order},
        "swing_rows": {name: s.stop - s.start for name, s in swing_layout.items()},
        "settings": {"learners": learners, "plain": plain, "swing": swing,
                     "swing_columns": swing_columns,
                     "context_per_class": CONTEXT_PER_CLASS, "seed": SEED,
                     "score_forms": SCORE_FORMS},
        "precision_gaps": gaps,
        "metrics": metric_rows(scores, layout, correct),
        "swing_metrics": metric_rows(swing_scores_by_name, swing_layout,
                                     correct[chosen[:swing_layout["val"].stop]]),
        "errors": errors, "timing": timing}


def measure(spec: RunSpec, paths: DataPaths, out_root: Path, device: torch.device,
            workers: int, passes: int = 20) -> None:
    """Time gate: seconds for one pass of the label swing, at each column count."""
    folder = run_dir(out_root, spec.name)
    epoch = max(int(p.stem.split("_")[1]) for p in (folder / "checkpoints").glob(
        "epoch_*.pt"))
    model, state = load_model(checkpoint_path(out_root, spec.name, epoch), device)
    sets = evaluation_sets(paths, spec.held_out, IMAGE_SIZE, None, workers)
    train = extract(model, sets["train"], device, workers=0)
    x, y = train[PENULTIMATE].double().numpy(), train["labels"].numpy()
    n_classes = len(state["classes"])
    context = balanced_context(y, n_classes)
    for columns in SWING_COLUMNS:
        reduce = reduce_columns(x, columns)
        queries = reduce(x[:max(1, passes // n_classes)])
        swing_scores(learner(PRIMARY, str(device), swing=True),
                     reduce(x[context]), y[context], queries[:1], n_classes)
        began = time.time()
        swing_scores(learner(PRIMARY, str(device), swing=True),
                     reduce(x[context]), y[context], queries, n_classes)
        seconds = (time.time() - began) / (len(queries) * n_classes)
        print(f"[probe] time gate, {columns} columns: {seconds:.3f} s for each "
              f"pass (limit 0.1)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--only", nargs="+", metavar="RUN")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--imagenette-dir", "--id-dir", required=True)
    parser.add_argument("--imagewoof-dir")
    parser.add_argument("--textures-dir")
    parser.add_argument("--extra-set", action="append", default=[],
                        metavar="NAME=DIR")
    parser.add_argument("--extra-limit", type=int, default=5000)
    parser.add_argument("--learners", nargs="*", default=[*VERSIONS, *CONTROLS],
                        help="learners of direction 1")
    parser.add_argument("--plain", nargs="*", default=list(VERSIONS),
                        help="learners of the plain prediction D2-0")
    parser.add_argument("--swing", nargs="*", default=[PRIMARY],
                        help="learners of the label swing D2; none skips it")
    parser.add_argument("--swing-columns", type=int, default=SWING_COLUMNS[0],
                        choices=SWING_COLUMNS)
    parser.add_argument("--swing-rows", type=int, default=SWING_ROWS)
    parser.add_argument("--measure", action="store_true",
                        help="time gate only: time the label swing on the first run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--limit", type=int, help="smoke runs only: images per set")
    args = parser.parse_args()
    if not args.stage and not args.only:
        parser.error("give --stage or --only")
    specs = STAGES[args.stage] if args.stage else []
    if args.only:
        named = [parse_run(name) for name in args.only]
        specs = [spec for spec in specs if spec in named] if args.stage else named
    extra = []
    for item in args.extra_set:
        name, _, folder = item.partition("=")
        extra.append((name, Path(folder)))
    paths = DataPaths(Path(args.imagenette_dir),
                      Path(args.imagewoof_dir) if args.imagewoof_dir else None,
                      Path(args.textures_dir) if args.textures_dir else None,
                      tuple(extra), args.extra_limit)
    out_root, device = Path(args.out_root), torch.device(args.device)
    if args.measure:
        measure(specs[0], paths, out_root, device, args.workers)
        return
    failures = []
    for spec in specs:
        target = run_dir(out_root, spec.name) / "probe" / "results.json"
        if target.exists() and not args.force:
            print(f"[probe] {spec.name}: done, skipped", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        began = time.time()
        try:
            out = probe_run(spec, paths, out_root, device, args.learners, args.plain,
                            args.swing, args.swing_columns, args.swing_rows,
                            args.workers, args.image_size, args.limit)
        except Exception:  # noqa: BLE001 - continue with the next run
            failures.append(spec.name)
            print(f"[probe] {spec.name} FAILED\n{traceback.format_exc()}", flush=True)
            continue
        out["seconds"] = round(time.time() - began, 1)
        target.write_text(json.dumps(out))
        print(f"[probe] {spec.name} epoch {out['epoch']}: {len(out['errors'])} "
              f"errors, {out['seconds']}s, timing {out['timing']}", flush=True)
    print("[probe] failures: " + (", ".join(failures) if failures else "none"),
          flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
