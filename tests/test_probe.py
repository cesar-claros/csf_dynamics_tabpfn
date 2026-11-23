import json

import numpy as np
import pytest
import torch

from dynamics import analyze_probe, run_stage
from dynamics import tabular_probe as tp
from dynamics.analyze_step0 import FAR_SETS
from dynamics.scores import ALL_CSFS


def clusters(seed: int = 0, per_class: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.array([[6.0, 0, 0, 0], [0, 6.0, 0, 0], [0, 0, 6.0, 0]])
    x = np.vstack([mean + rng.normal(0, 0.5, (per_class, 4)) for mean in means])
    return x, np.repeat(np.arange(3), per_class)


def test_probe_rows_are_fixed_and_balanced():
    labels = np.arange(300) % 10
    rows = tp.probe_rows({"val": 300, "places": 4000, "sun": 40}, labels, 10,
                         per_set=100)
    assert np.bincount(labels[rows["val"]]).tolist() == [10] * 10
    assert len(rows["places"]) == 100 and len(set(rows["places"])) == 100
    assert len(rows["sun"]) == 40 and (np.diff(rows["places"]) > 0).all()
    again = tp.probe_rows({"val": 300, "places": 4000, "sun": 40}, labels, 10, 100)
    assert all((rows[name] == again[name]).all() for name in rows)
    other = tp.probe_rows({"val": 300, "places": 4000, "sun": 40}, labels, 10, 100,
                          seed=1)
    assert not (rows["places"] == other["places"]).all()


def test_background_keeps_the_columns_and_breaks_their_link():
    rng = np.random.default_rng(0)
    first = rng.normal(size=2000)
    train = np.column_stack([first, first, rng.uniform(5, 6, 2000)])
    made = tp.background(train, 1000, np.random.default_rng(1))
    assert made.shape == (1000, 3)
    assert np.isin(made[:, 0], train[:, 0]).all() and made[:, 2].min() >= 5
    assert abs(np.corrcoef(made[:, 0], made[:, 1])[0, 1]) < 0.1
    assert np.corrcoef(train[:, 0], train[:, 1])[0, 1] == pytest.approx(1.0)


def test_class_scores_prefer_the_own_class():
    x, y = clusters()
    queries = np.array([[6.0, 0, 0, 0], [0, 6.0, 0, 0], [0, 0, 6.0, 0]])
    table = tp.class_scores(tp.learner("trees", "cpu"), x, y, queries, 3)
    assert table.shape == (3, 3) and np.isfinite(table).all()
    assert (table.argmax(axis=1) == np.arange(3)).all()
    assert (np.diag(table) > np.log(0.8 / 0.2)).all(), "log-odds of the own class"
    assert (table[~np.eye(3, dtype=bool)] < 0).all()


class KernelClassifier:
    """A Gaussian-kernel vote. Unlike a k-NN vote, it depends on the distance."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelClassifier":
        self.x, self.y, self.classes_ = x, y, np.unique(y)
        return self

    def predict_proba(self, queries: np.ndarray) -> np.ndarray:
        squared = ((queries[:, None, :] - self.x[None, :, :]) ** 2).sum(axis=2)
        weight = np.exp(-squared / 2)
        votes = np.stack([weight[:, self.y == c].sum(axis=1)
                          for c in self.classes_], axis=1)
        return votes / votes.sum(axis=1, keepdims=True)


def test_label_swing_is_larger_far_from_the_data():
    x, y = clusters()
    queries = np.array([[6.0, 0, 0, 0], [40.0, 40.0, 40.0, 40.0]])
    inside, far = tp.swing_scores(KernelClassifier, x, y, queries, 3)
    assert inside == pytest.approx(np.log(3), abs=0.1), "one label fits"
    assert far == pytest.approx(0.0, abs=1e-6), "every forced label is accepted"
    between = np.array([[6.0, 0, 0, 0], [3.0, 3.0, 0, 0]])
    plain = tp.plain_scores(KernelClassifier, x, y, between)
    table = KernelClassifier().fit(x, y).predict_proba(between)
    assert plain[1] == pytest.approx(-np.log(1 - table[1].max()), rel=1e-6)
    # At a centre the top probability is 1 minus about 1e-13, and 1 - max is not
    # exact any more. The score adds up the other classes in its place.
    assert plain[0] == pytest.approx(-np.log(np.sort(table[0])[:-1].sum()), rel=1e-9)
    assert plain[0] > plain[1], "the top class leaves less to the others at a centre"
    context = tp.balanced_context(y, 3, per_class=10)
    assert np.bincount(y[context]).tolist() == [10, 10, 10]


class SaturatedModel:
    """A stand-in with the output methods of TabPFN and a single precision softmax.

    One feature is the logit gap between the two classes. `predict_proba` rounds to
    1 above a gap of about 17, as a softmax in single precision does.
    """

    classes_ = np.array([0, 1])
    estimators = 2
    shift = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SaturatedModel":
        return self

    def predict_raw_logits(self, rows: np.ndarray) -> np.ndarray:
        gap = rows[:, 0].astype(np.float32)
        one = np.stack([np.zeros_like(gap), gap], axis=1)
        return np.stack([one] * self.estimators)

    def logits_to_probabilities(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.softmax(raw, dim=-1).mean(dim=0)

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        raw = torch.from_numpy(self.predict_raw_logits(rows))
        return (self.logits_to_probabilities(raw).numpy()
                + np.array([self.shift, -self.shift], dtype=np.float32))


def test_double_precision_keeps_the_order_that_single_precision_loses():
    rows = np.array([[20.0], [25.0], [30.0], [2.0]])
    model = SaturatedModel()
    single = model.predict_proba(rows)[:, 1]
    assert (single[:3] == 1.0).all(), "single precision ties the confident rows"
    gaps: list[float] = []
    table = tp.probabilities(model, rows, gaps)
    assert table.dtype == np.float64 and gaps[0] < 1e-6
    assert (np.diff(table[:3, 0]) < 0).all() and (table[:3, 0] > 0).all()
    assert tp.log_odds(model, rows) == pytest.approx([20.0, 25.0, 30.0, 2.0], rel=1e-5)
    scores = tp.plain_scores(SaturatedModel, rows, np.array([0, 1, 0, 1]), rows)
    assert scores == pytest.approx([20.0, 25.0, 30.0, 2.126928], rel=1e-5)
    assert len(np.unique(scores)) == 4


def test_a_mismatch_with_predict_proba_falls_back_and_is_recorded():
    rows = np.array([[1.0], [2.0]])
    model = SaturatedModel()
    model.shift = 0.01
    gaps: list[float] = []
    table = tp.probabilities(model, rows, gaps)
    assert gaps[0] == pytest.approx(0.01, rel=1e-3)
    assert table == pytest.approx(model.predict_proba(rows))
    noted: dict[str, float] = {}
    errors: dict[str, str] = {}
    tp.note_gaps("D1:stand_in", gaps, noted, errors)
    assert noted == {"D1:stand_in": gaps[0]} and "precision:D1:stand_in" in errors
    tp.note_gaps("D1:other", [], noted, errors)
    tp.note_gaps("D1:fine", [1e-7], noted, errors)
    assert set(noted) == {"D1:stand_in", "D1:fine"} and len(errors) == 1


def test_controls_use_the_decision_function():
    x, y = clusters()
    binary = (y == 0).astype(int)
    model = tp.learner("logistic", "cpu")().fit(x, binary)
    queries = np.array([[6.0, 0, 0, 0], [60.0, 0, 0, 0], [0, 6.0, 0, 0]])
    odds = tp.log_odds(model, queries)
    assert odds == pytest.approx(model.decision_function(queries))
    assert odds[1] > odds[0] > 0 > odds[2], "no tie for the two confident rows"


def test_tied_pairs_and_the_tie_table(tmp_path):
    assert analyze_probe.tied_pairs(np.array([1.0, 1.0, 0.5]),
                                    np.array([1.0, 0.2])) == pytest.approx(2 / 6)
    assert analyze_probe.tied_pairs(np.arange(4.0), np.arange(4.0) + 10) == 0.0
    folder = tmp_path / "runs" / "A_seed5" / "probe"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    sets = ["val", "places", "sun"]
    clean = rng.normal(size=30)
    tied = np.r_[np.ones(8), rng.normal(size=2), np.ones(5), rng.normal(size=15)]
    swing = np.r_[np.zeros(4), np.zeros(4)]
    np.savez_compressed(
        folder / "scores.npz", names=np.array(["MSR", "D1:trees", "D1:tabpfn_v3"]),
        values=np.stack([tied, clean, tied]), sets=np.array(sets),
        bounds=np.array([10, 20, 30]), swing_names=np.array(["D1:trees",
                                                             "D2:tabpfn_v3"]),
        swing_values=np.stack([swing, swing]))
    (folder / "results.json").write_text(json.dumps(
        {"run": "A_seed5", "swing_rows": {"val": 4, "places": 4}}))
    table = analyze_probe.tie_shares(tmp_path).set_index("score")["tied"]
    assert set(table.index) == {"D1:trees", "D1:tabpfn_v3", "D2:tabpfn_v3"}
    assert table["D1:trees"] == 0.0 and table["D2:tabpfn_v3"] == 1.0
    assert table["D1:tabpfn_v3"] == pytest.approx(8 * 5 / 100)


def test_unknown_learner_and_column_reduction():
    with pytest.raises(ValueError, match="unknown learner"):
        tp.learner("forest", "cpu")
    x, _ = clusters()
    assert tp.reduce_columns(x, 512)(x) is x
    assert tp.reduce_columns(x, 2)(x[:5]).shape == (5, 2)


def test_probe_on_a_tiny_run(tiny_data, tmp_path, monkeypatch):
    extra = tiny_data.imagenette.parent / "openood"
    common = ["--out-root", str(tmp_path), "--imagenette-dir",
              str(tiny_data.imagenette), "--imagewoof-dir", str(tiny_data.imagewoof),
              "--textures-dir", str(tiny_data.textures), "--image-size", "32",
              "--workers", "0", "--device", "cpu"]
    monkeypatch.setattr("sys.argv", ["run_stage", "--only", "B_seed0", "--epochs",
                                     "1", "--skip-eval", *common])
    run_stage.main()
    monkeypatch.setattr("sys.argv", [
        "tabular_probe", "--only", "B_seed0", "--learners", "logistic", "--plain",
        "logistic", "--swing", "logistic", "--swing-rows", "4", "--extra-limit", "20",
        "--extra-set", f"places={extra / 'places'}", "--extra-set",
        f"ssb_hard={extra / 'ssb_hard'}", *common])
    tp.main()
    folder = tmp_path / "runs" / "B_seed0" / "probe"
    out = json.loads((folder / "results.json").read_text())
    assert out["rows"]["val"] == 120 and out["rows"]["places"] == 20
    assert out["swing_rows"] == {"val": 4, "places": 4}
    names = {row["score"] for row in out["metrics"]}
    assert out["settings"]["score_forms"] == tp.SCORE_FORMS
    assert out["precision_gaps"] == {}, "the controls have no raw logits"
    assert set(ALL_CSFS) <= names
    assert {"D1:logistic", "D1max:logistic", "D2-0:logistic"} <= names
    assert {row["set"] for row in out["metrics"]} == {"places", "ssb_hard",
                                                      "imagewoof"}
    assert "D2:logistic" in {row["score"] for row in out["swing_metrics"]}
    saved = np.load(folder / "scores.npz")
    assert saved["values"].shape == (len(saved["names"]), 120 + 20 + 18 + 60)
    assert saved["values"].dtype == np.float64
    tp.main()          # a complete run is skipped
    # With --probe-root, the checkpoint is read in the run folder and the results
    # go to the other folder.
    other = tmp_path / "probe_only"
    (folder / "results.json").unlink()
    monkeypatch.setattr("sys.argv", [
        "tabular_probe", "--only", "B_seed0", "--learners", "logistic", "--plain",
        "--swing", "--extra-limit", "20", "--probe-root", str(other),
        "--extra-set", f"places={extra / 'places'}", *common])
    tp.main()
    moved = other / "runs" / "B_seed0" / "probe"
    assert sorted(path.name for path in moved.iterdir()) == ["results.json",
                                                             "scores.npz"]
    assert not (folder / "results.json").exists()
    assert not (other / "runs" / "B_seed0" / "checkpoints").exists()
    again = json.loads((moved / "results.json").read_text())
    assert "D1:logistic" in {row["score"] for row in again["metrics"]}
    assert "D2:logistic" not in {row["score"] for row in again["swing_metrics"]}


def write_results(root, run: str, far: dict[str, float], swing: dict[str, float]
                  ) -> None:
    folder = root / "runs" / run / "probe"
    folder.mkdir(parents=True)
    def rows(values: dict[str, float]) -> list[dict]:
        return [{"score": name, "set": ood, "auroc": value} for name, value
                in values.items() for ood in FAR_SETS]
    (folder / "results.json").write_text(json.dumps({
        "run": run, "metrics": rows(far), "swing_metrics": rows(swing)}))


def test_report_reads_a_lead_and_a_swing_that_adds(tmp_path, monkeypatch):
    recipes = analyze_probe.TEST_RECIPES
    for run in [f"{r}_seed{s}" for r in recipes for s in range(3)] + list(
            analyze_probe.DEVELOPMENT):
        recipe = run.split("_")[0]
        seed = int(run[-1])
        # Each roster score is good, except in one recipe of its own.
        far = {name: 0.80 - (0.06 if recipes[i % 6] == recipe else 0.0) + 0.001 * seed
               for i, name in enumerate(ALL_CSFS)}
        far |= {"D1:tabpfn_v3.5": 0.797, "D1max:tabpfn_v3.5": 0.79,
                "D1:tabpfn_v3": 0.775, "D1:tabpfn_v2.5": 0.797 - 0.001 * seed,
                "D1:trees": 0.76, "D1:logistic": 0.70,
                "D2-0:tabpfn_v3.5": 0.74, "D2-0:tabpfn_v3": 0.74}
        swing = dict(far) | {"D2:tabpfn_v3.5": 0.77 + 0.002 * seed}
        if seed == 2:
            swing.pop("D2:tabpfn_v3.5")
        write_results(tmp_path, run, far, swing)
    out = tmp_path / "probe_report.md"
    monkeypatch.setattr("sys.argv", ["analyze_probe", "--results-root", str(tmp_path),
                                     "--out", str(out)])
    analyze_probe.main()
    text = out.read_text()
    assert "20 runs with probe results: 18 test runs, 2 development runs" in text
    assert "D1:tabpfn_v3.5 pass, D2:tabpfn_v3.5 pass" in text
    assert "**D1 with TabPFN-3.5: lead:" in text
    assert "gradient-boosted trees: fail:" in text
    assert "12 test runs" in text and "**the swing adds.**" in text
    far = analyze_probe.wide(analyze_probe.read_results(tmp_path))
    versions = analyze_probe.version_table(
        far[far["recipe"].isin(recipes)], "D1").set_index(["first", "second"])
    assert versions.loc[("D1:tabpfn_v3.5", "D1:tabpfn_v3"), "differ"]
    assert not versions.loc[("D1:tabpfn_v3.5", "D1:tabpfn_v2.5"), "differ"]
    summary = analyze_probe.regret_summary(far[far["recipe"].isin(recipes)])
    assert summary.loc["D1:tabpfn_v3.5", "W"] < 1.0 < summary.loc["kNN", "W"]
