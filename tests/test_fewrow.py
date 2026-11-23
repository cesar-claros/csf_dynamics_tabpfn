import json

import numpy as np
import pytest

from dynamics import analyze_fewrow, fewrow_probe, run_stage
from dynamics.analyze_probe import DEVELOPMENT, TEST_RECIPES
from dynamics.analyze_step0 import FAR_SETS
from dynamics.paper_api import LOGIT_ONLY
from dynamics.scores import ALL_CSFS

D1 = analyze_fewrow.D1


def test_few_rows_are_balanced_and_fixed():
    labels = np.arange(300) % 10
    rows = fewrow_probe.few_rows(labels, 5, 10, draw=0)
    assert np.bincount(labels[rows]).tolist() == [5] * 10
    assert (np.diff(rows) > 0).all()
    assert (rows == fewrow_probe.few_rows(labels, 5, 10, draw=0)).all()
    assert not (rows == fewrow_probe.few_rows(labels, 5, 10, draw=1)).all()
    more = fewrow_probe.few_rows(labels, 8, 10, draw=0)
    assert np.bincount(labels[more]).tolist() == [8] * 10


def test_fewrow_on_a_tiny_run(tiny_data, tmp_path, monkeypatch):
    extra = tiny_data.imagenette.parent / "openood"
    common = ["--out-root", str(tmp_path), "--imagenette-dir",
              str(tiny_data.imagenette), "--imagewoof-dir", str(tiny_data.imagewoof),
              "--textures-dir", str(tiny_data.textures), "--image-size", "32",
              "--workers", "0", "--device", "cpu"]
    monkeypatch.setattr("sys.argv", ["run_stage", "--only", "B_seed0", "--epochs",
                                     "1", "--skip-eval", *common])
    run_stage.main()
    other = tmp_path / "probe_only"
    monkeypatch.setattr("sys.argv", [
        "fewrow_probe", "--only", "B_seed0", "--row-counts", "2", "3", "--draws", "2",
        "--learners", "logistic", "--full-learners", "logistic", "--extra-limit",
        "20", "--probe-root", str(other), "--extra-set",
        f"places={extra / 'places'}", *common])
    fewrow_probe.main()
    out = json.loads((other / "runs" / "B_seed0" / "fewrow" / "results.json"
                      ).read_text())
    assert [(arm["rows_per_class"], arm["draw"]) for arm in out["arms"]] == [
        (2, 0), (2, 1), (3, 0), (3, 1), (fewrow_probe.ALL_ROWS, 0)]
    assert out["rows"]["val"] == 120 and out["settings"]["source"] == "train"
    for arm in out["arms"]:
        names = {row["score"] for row in arm["metrics"]}
        assert {"D1:logistic", "D1max:logistic", "MSR"} <= names
        assert all(0 <= row["tied"] <= 1 for row in arm["metrics"])
        assert set(arm["errors"]).isdisjoint(names)
    first, second = out["arms"][0]["metrics"], out["arms"][1]["metrics"]
    value = {(row["score"], row["set"]): row["auroc"] for row in first}
    other_draw = {(row["score"], row["set"]): row["auroc"] for row in second}
    assert value[("MSR", "places")] == other_draw[("MSR", "places")]
    assert not (tmp_path / "runs" / "B_seed0" / "fewrow").exists()
    fewrow_probe.main()          # a complete run is skipped
    monkeypatch.setattr("sys.argv", [
        "fewrow_probe", "--only", "B_seed0", "--row-counts", "500", "--force",
        "--probe-root", str(other), *common])
    with pytest.raises(SystemExit):
        fewrow_probe.main()      # a class has fewer rows than the count


def write_run(root, run: str, scores_at, counts=(10, 30, 100), draws: int = 3,
              seconds: float = 600.0, drop: tuple = ()) -> None:
    """One result file. `scores_at(count)` gives the far AUROC of each score."""
    arms = []
    for count in (*counts, fewrow_probe.ALL_ROWS):
        for draw in range(1 if count == fewrow_probe.ALL_ROWS else draws):
            values = scores_at(count)
            arms.append({
                "rows_per_class": count, "draw": draw, "timing": {},
                "errors": {name: "failed" for name, c, d in drop
                           if (c, d) == (count, draw)},
                "metrics": [{"score": name, "set": ood, "auroc": value, "tied": 0.0}
                            for name, value in values.items() for ood in FAR_SETS
                            if (name, count, draw) not in drop]})
    folder = root / "runs" / run / "fewrow"
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps({
        "run": run, "seconds": seconds, "settings": {"draws": draws}, "arms": arms}))


def roster(recipe: str, count: int, seed: int) -> dict[str, float]:
    """The data-free scores do not change with the rows. The others gain with rows."""
    recipes = list(TEST_RECIPES)
    gain = {10: 0.0, 30: 0.02, 100: 0.04, fewrow_probe.ALL_ROWS: 0.08}[count]
    out = {}
    for i, name in enumerate(ALL_CSFS):
        weak = 0.06 if recipe in recipes and recipes[i % 6] == recipe else 0.0
        out[name] = (0.78 if name in LOGIT_ONLY else 0.74 + gain) - weak + 0.001 * seed
    return out


def run_report(root, monkeypatch) -> str:
    out = root / "fewrow_report.md"
    monkeypatch.setattr("sys.argv", ["analyze_fewrow", "--results-root", str(root),
                                     "--out", str(out)])
    analyze_fewrow.main()
    return out.read_text()


def test_report_reads_a_lead_and_a_prior_that_helps(tmp_path, monkeypatch):
    runs = [f"{r}_seed{s}" for r in TEST_RECIPES for s in range(3)] + list(DEVELOPMENT)
    for run in runs:
        recipe, seed = run.split("_")[0], int(run[-1])

        def scores_at(count, recipe=recipe, seed=seed):
            values = roster(recipe, count, seed)
            if count != fewrow_probe.ALL_ROWS:
                values[D1] = 0.778
            values["D1:logistic"] = 0.75 + 0.002 * (seed % 3)
            values["D1:trees"] = 0.74
            return values
        write_run(tmp_path, run, scores_at)
    text = run_report(tmp_path, monkeypatch)
    assert "20 runs with few-row results: 18 test runs, 2 development runs" in text
    assert "pass" in text and "continue" in text and "inside the limit" in text
    assert f"**{D1}: lead:" in text
    assert "**The prior helps.**" in text
    assert "Rule 1 at 10 rows for each class (primary)" in text
    assert "Rule 1 at 100 rows for each class (secondary)" in text
    assert "A gate on the development runs does not allow the test" not in text
    free = analyze_fewrow.data_free(analyze_fewrow.read_results(tmp_path)[0])
    assert free == sorted(LOGIT_ONLY)
    # The fitted scores are 0.04 below the data-free ones at 10 rows, level at 100
    # rows, and above them with all rows.
    assert "is at least 1.0 point above GEN: none of the few-row counts" in text


def test_futility_gate_stops_and_time_gate_cuts_the_draws(tmp_path, monkeypatch):
    for run in DEVELOPMENT:
        def scores_at(count):
            values = roster("A", count, 0)
            values[D1], values["D1:logistic"], values["D1:trees"] = 0.74, 0.74, 0.73
            return values
        write_run(tmp_path, run, scores_at, seconds=1500.0)
    text = run_report(tmp_path, monkeypatch)
    assert "STOP, no test" in text
    assert "so the draws go to 1" in text
    assert "A gate on the development runs does not allow the test" in text
    assert "Mean far AUROC over the development runs" in text


def test_a_score_needs_every_draw(tmp_path):
    def scores_at(count):
        return roster("C1", count, 0) | {D1: 0.7}
    write_run(tmp_path, "C1_seed0", scores_at, drop=(("ViM", 10, 1),))
    results, runs = analyze_fewrow.read_results(tmp_path)
    table = analyze_fewrow.wide(results, 10)
    assert np.isnan(table.loc["C1_seed0", "ViM"])
    assert table.loc["C1_seed0", "MLS"] == pytest.approx(78.0)
    kept, missing = analyze_fewrow.complete(table)
    assert missing == ["ViM"] and "ViM" not in kept
    assert runs.loc[0, "errors"][10] == ["ViM"]
    assert not np.isnan(analyze_fewrow.wide(results, 30).loc["C1_seed0", "ViM"])
