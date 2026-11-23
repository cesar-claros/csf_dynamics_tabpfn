"""Probe analysis: OOD scores from a tabular foundation model against the roster.

Reads <root>/runs/<run>/probe/results.json and applies the decision rules of the
protocol: the regret of each new score against the best of the 23 roster scores,
its worst-recipe regret W and its mean regret M, the reading of direction 1 and
of direction 2, and the comparison of the model versions.

    python -m dynamics.analyze_probe --results-root dyn_results

Protocol: ../documentation/tabular_prior_probe_protocol_2026-11-17.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics.analyze_c import interval
from dynamics.analyze_e import block
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import parse_run
from dynamics.scores import ALL_CSFS
from dynamics.tabular_probe import NEAR_SETS, PRIMARY, VERSIONS, tied_pairs

HERE = Path(__file__).resolve().parent
DEVELOPMENT = ("A_seed5", "B_seed5")
TEST_RECIPES = ("C1", "C2", "C3", "C4", "C5", "C6")
SANITY = 60.0
#: Share of tied pairs of one ID row and one OOD row above which a score is marked.
TIE_LIMIT = 0.01
LEAD_W = 1.0
LEAD_M = 0.5
BELOW_COUNTERPART = 1.0
VERSION_GAP = 1.0
D1, D2, PLAIN = f"D1:{PRIMARY}", f"D2:{PRIMARY}", f"D2-0:{PRIMARY}"
COUNTERPART = {"D1": "MahaPP", "D2": "pNML"}


def read_results(root: Path, key: str = "metrics") -> pd.DataFrame:
    """Per run and score: the mean AUROC over the far sets and over the near sets."""
    rows = []
    for path in sorted(Path(root).glob("runs/*/probe/results.json")):
        out = json.loads(path.read_text())
        frame = pd.DataFrame(out[key])
        if frame.empty:
            continue
        for group, sets in (("far", FAR_SETS), ("near", NEAR_SETS)):
            part = frame[frame["set"].isin(sets)]
            for score, value in (part.groupby("score")["auroc"].mean() * 100).items():
                rows.append({"run": out["run"], "recipe": parse_run(out["run"]).recipe,
                             "group": group, "score": score, "auroc": value})
    return pd.DataFrame(rows)


def tie_shares(root: Path) -> pd.DataFrame:
    """Per run and new score: the largest share of tied pairs over the far sets.

    An AUROC counts a tied pair as a half. A large share means that the score has
    lost the order of many rows, for example by a probability that rounds to 1.
    """
    rows = []
    for path in sorted(Path(root).glob("runs/*/probe/scores.npz")):
        run = path.parent.parent.name
        saved = np.load(path, allow_pickle=False)
        sets = [str(name) for name in saved["sets"]]
        starts = np.r_[0, saved["bounds"][:-1]]
        layout = {name: slice(int(a), int(b)) for name, a, b
                  in zip(sets, starts, saved["bounds"], strict=True)}
        swing = json.loads((path.parent / "results.json").read_text())["swing_rows"]
        swing_layout, start = {}, 0
        for name, count in swing.items():
            swing_layout[name] = slice(start, start + count)
            start += count
        for names, values, where in ((saved["names"], saved["values"], layout),
                                     (saved["swing_names"], saved["swing_values"],
                                      swing_layout)):
            for name, row in zip(names, values, strict=True):
                # The table of the swing rows repeats the other scores: keep D2.
                if ":" not in str(name) or (
                        where is swing_layout and not str(name).startswith("D2:")):
                    continue
                far = [tied_pairs(row[where["val"]], row[where[ood]])
                       for ood in FAR_SETS if ood in where]
                if far:
                    rows.append({"run": run, "score": str(name), "tied": max(far)})
    return pd.DataFrame(rows)


def wide(results: pd.DataFrame, group: str = "far") -> pd.DataFrame:
    """Runs as rows and scores as columns, for one group of OOD sets."""
    part = results[results["group"] == group]
    table = part.pivot(index="run", columns="score", values="auroc")
    table.insert(0, "recipe", [parse_run(run).recipe for run in table.index])
    return table


def regret_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per score: mean AUROC, mean regret M, and worst-recipe regret W.

    The regret of a score in a run is the AUROC of the best roster score minus the
    AUROC of the score. A recipe has the mean regret of its runs.
    """
    roster = [name for name in ALL_CSFS if name in table.columns]
    scores = table.drop(columns="recipe")
    regret = scores.rsub(scores[roster].max(axis=1), axis=0)
    by_recipe = regret.groupby(table["recipe"]).mean()
    return pd.DataFrame({"auroc": scores.mean(), "M": by_recipe.mean(),
                         "W": by_recipe.max(),
                         "worst_recipe": by_recipe.idxmax()}).rename_axis("score")


def read_direction(summary: pd.DataFrame, score: str, counterpart: str) -> str:
    """Lead, fail, or no gain, by the rules of the protocol."""
    if score not in summary.index:
        return "not computed"
    roster = summary.loc[[name for name in ALL_CSFS if name in summary.index]]
    w_fixed, m_fixed = roster["W"].min(), roster["M"].min()
    row = summary.loc[score]
    if row["W"] <= w_fixed - LEAD_W and row["M"] <= m_fixed + LEAD_M:
        return (f"lead: W {row['W']:.2f} against {w_fixed:.2f} for the best fixed "
                f"score, M {row['M']:.2f} against {m_fixed:.2f}")
    below = summary.loc[counterpart, "auroc"] - row["auroc"]
    if row["W"] > w_fixed + LEAD_W or below > BELOW_COUNTERPART:
        return (f"fail: W {row['W']:.2f} against {w_fixed:.2f}, and {below:.2f} "
                f"points below {counterpart}")
    return f"no gain: W {row['W']:.2f} against {w_fixed:.2f} for the best fixed score"


def paired(table: pd.DataFrame, first: str, second: str) -> dict:
    """Mean of first minus second over the runs that have both, with an interval."""
    both = table[[first, second]].dropna()
    mean, low, high = interval((both[first] - both[second]).to_numpy())
    return {"first": first, "second": second, "runs": len(both), "difference": mean,
            "low": low, "high": high}


def version_table(table: pd.DataFrame, family: str) -> pd.DataFrame:
    """Every pair of model versions for one score family, on the same runs."""
    names = [f"{family}:{name}" for name in VERSIONS if f"{family}:{name}"
             in table.columns]
    rows = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            row = paired(table, first, second)
            row["differ"] = bool(abs(row["difference"]) >= VERSION_GAP
                                 and (row["low"] > 0 or row["high"] < 0))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "probe_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    results = read_results(root)
    if results.empty:
        raise SystemExit(f"no probe results under {root}/runs")
    swing = read_results(root, "swing_metrics")
    far = wide(results)
    test = far[far["recipe"].isin(TEST_RECIPES)]
    develop = far[far.index.isin(DEVELOPMENT)]
    new = [name for name in far.columns if ":" in name]
    sanity = {name: bool((develop[name].dropna() >= SANITY).all())
              for name in (D1,) if name in develop}
    if not swing.empty:
        swing_far = wide(swing)
        swing_develop = swing_far[swing_far.index.isin(DEVELOPMENT)]
        if D2 in swing_develop and swing_develop[D2].notna().any():
            sanity[D2] = bool((swing_develop[D2].dropna() >= SANITY).all())
    sections = [("Runs", (
        f"{len(far)} runs with probe results: {len(test)} test runs, "
        f"{len(develop)} development runs.\n\n**Sanity gate (far AUROC of at "
        f"least {SANITY:.0f} on the development runs):** "
        + (", ".join(f"{name} {'pass' if ok else 'FAIL'}"
                     for name, ok in sanity.items()) or "no development run")
        + ".\n\n" + block(develop[["recipe", *[n for n in new if n in develop]]]
                          .reset_index(), 1)))]
    ties = tie_shares(root)
    if not ties.empty:
        worst = ties.groupby("score")["tied"].max().sort_values(ascending=False)
        marked = [name for name, share in worst.items() if share > TIE_LIMIT]
        sections.append(("Ties", (
            "Largest share of tied pairs of one ID row and one OOD row, over the "
            f"runs and the far sets. A share above {TIE_LIMIT:.0%} means that the "
            "AUROC of the score is not reliable: "
            + (", ".join(f"**{name}**" for name in marked) if marked else "none")
            + ".\n\n" + block(worst.rename("tied").reset_index(), 3))))
    if not test.empty:
        summary = regret_summary(test)
        shown = summary.loc[[*new, *summary.loc[[n for n in ALL_CSFS
                                                 if n in summary.index]]
                             .sort_values("W").index[:6]]]
        reading = read_direction(summary, D1, COUNTERPART["D1"])
        trees = read_direction(summary, "D1:trees", COUNTERPART["D1"])
        sections.append(("Direction 1: one class at a time", (
            "Far target, test runs. `M` is the mean regret over the recipes, and "
            "`W` is the regret of the worst recipe. The six roster scores with the "
            "lowest `W` follow the new scores:\n\n" + block(shown.reset_index(), 2)
            + f"\n\n**D1 with TabPFN-3.5: {reading}.**\n\nThe same task with "
            f"gradient-boosted trees: {trees}.\n\nBy recipe, AUROC:\n\n"
            + block(test.groupby("recipe")[[n for n in new if n.startswith("D1")]
                                           ].mean().reset_index(), 1))))
        near = wide(results, "near")
        near = near[near["recipe"].isin(TEST_RECIPES)]
        if not near.empty:
            sections.append(("Near target, reported", block(
                regret_summary(near).loc[[n for n in new if n in near.columns]]
                .reset_index(), 2)))
        versions = pd.concat([version_table(test, family)
                              for family in ("D1", "D1max", "D2-0")])
        sections.append(("Task T2: model versions, direction 1 and plain", (
            "Paired differences of the far AUROC over the test runs. Two versions "
            f"differ if the difference is at least {VERSION_GAP} point and the "
            "interval excludes 0:\n\n" + block(versions, 2))))
    if not swing.empty:
        table = wide(swing)
        table = table[table["recipe"].isin(TEST_RECIPES)]
        with_swing = table[table[D2].notna()] if D2 in table else table.iloc[:0]
        if not with_swing.empty:
            summary = regret_summary(with_swing)
            adds = paired(with_swing, D2, PLAIN)
            verdict = ("the swing adds" if adds["difference"] > 0 and adds["low"] > 0
                       else "the swing does not add")
            sections.append(("Direction 2: label swing", (
                f"Far target, the rows of the swing, {len(with_swing)} test runs."
                "\n\n" + block(summary.loc[[n for n in summary.index if ":" in n
                                           or n in ("pNML", "MSR", "MLS")]]
                               .reset_index(), 2)
                + f"\n\nD2 minus D2-0: {adds['difference']:.2f} "
                f"({adds['low']:.2f} to {adds['high']:.2f}). **{verdict}.**\n\n"
                f"**D2 with TabPFN-3.5: "
                f"{read_direction(summary, D2, COUNTERPART['D2'])}.**")))
            sections.append(("Task T2: model versions, label swing",
                             block(version_table(table, "D2"), 2)))
    text = ("# Probe: OOD scores from a tabular foundation model\n\nThe rules were "
            "fixed before these results.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
