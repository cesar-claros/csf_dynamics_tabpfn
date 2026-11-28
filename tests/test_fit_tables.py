import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from dynamics import fit_tables as ft


def write_tables(path, seed: int = 0, content: str = "abc") -> dict:
    """A tables file with two beds, three fits, and two targets."""
    rng = np.random.default_rng(seed)
    arrays, manifest = {}, {}
    for bed, folds in (("P", 1), ("S1", 2)):
        manifest[bed] = {"title": bed, "csfs": ["one", "two"], "folds": [
            {"name": f"f{n}", "test_ids": [f"t{i}" for i in range(5)]}
            for n in range(folds)]}
        for number in range(folds):
            x = rng.normal(size=(30, 4))
            arrays[f"{bed}.{number:03d}.x_fit"] = x
            arrays[f"{bed}.{number:03d}.y_fit"] = np.column_stack(
                [x[:, 0] + 0.1 * rng.normal(size=30), np.sin(x[:, 1])])
            arrays[f"{bed}.{number:03d}.weights"] = rng.uniform(0.5, 2.0, 30)
            arrays[f"{bed}.{number:03d}.x_test"] = rng.normal(size=(5, 4))
    np.savez_compressed(path, manifest=np.array(json.dumps(manifest)),
                        content_sha256=np.array(content), **arrays)
    return arrays


def run(monkeypatch, tables, out, *extra: str) -> None:
    monkeypatch.setattr("sys.argv", ["fit_tables", "--tables", str(tables), "--out",
                                     str(out), "--device", "cpu", *extra])
    ft.main()


def test_ridge_is_the_weighted_ridge_with_no_penalty_on_the_offset():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 5)), rng.normal(size=(40, 3))
    weights, new = rng.uniform(0.5, 2.0, 40), rng.normal(size=(7, 5))
    found = ft.ridge(x, y, weights, new)
    offset = (weights[:, None] * y).sum(axis=0) / weights.sum()
    scaled = weights * len(weights) / weights.sum()
    model = Ridge(alpha=ft.PENALTY, fit_intercept=False).fit(x, y - offset,
                                                             sample_weight=scaled)
    assert found == pytest.approx(model.predict(new) + offset)


def test_fits_are_written_resumed_and_tied_to_their_tables(tmp_path, monkeypatch,
                                                           capsys):
    tables, out = tmp_path / "tables.npz", tmp_path / "predictions.npz"
    arrays = write_tables(tables)
    run(monkeypatch, tables, out, "--learners", "ridge", "--beds", "S1")
    saved = np.load(out)
    assert sorted(k for k in saved.files if "." in k) == ["ridge.S1.000",
                                                          "ridge.S1.001"]
    assert str(saved["content_sha256"]) == "abc"
    assert saved["ridge.S1.001"] == pytest.approx(ft.ridge(
        *(arrays[f"S1.001.{part}"] for part in ("x_fit", "y_fit", "weights",
                                                "x_test"))))
    run(monkeypatch, tables, out, "--learners", "ridge", "forest")
    text = capsys.readouterr().out
    assert "ridge: 1 new fits of 3" in text and "forest: 3 new fits of 3" in text
    saved = np.load(out)
    assert list(saved["learners"]) == ["forest", "ridge"]
    assert saved["forest.P.000"].shape == (5, 2)
    assert np.isfinite(saved["forest.S1.001"]).all()
    run(monkeypatch, tables, out, "--learners", "ridge", "forest")
    assert "forest: 0 new fits of 3" in capsys.readouterr().out
    write_tables(tables, content="other")
    with pytest.raises(SystemExit, match="other tables"):
        run(monkeypatch, tables, out, "--learners", "ridge")


def test_unknown_learner():
    with pytest.raises(ValueError, match="unknown learner"):
        ft.regressor("spline", "cpu")
