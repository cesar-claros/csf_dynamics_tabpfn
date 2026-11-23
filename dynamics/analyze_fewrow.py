"""Few-row probe analysis: one-class scores against the scores that few rows allow.

Reads <root>/runs/<run>/fewrow/results.json and applies the gates and the rules of
the protocol: the sanity, futility, and time gates on the development runs, the
reading of D1 against the 23 available scores at each row count, the comparison
with the controls, and the loss of each fitted score from all rows to few rows.

    python -m dynamics.analyze_fewrow --results-root dyn_results

Protocol: ../documentation/few_row_probe_protocol_2026-11-23.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dynamics.analyze_e import block
from dynamics.analyze_probe import (
    DEVELOPMENT,
    SANITY,
    TEST_RECIPES,
    TIE_LIMIT,
    paired,
    read_direction,
    regret_summary,
)
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import parse_run
from dynamics.fewrow_probe import ALL_ROWS, PRIMARY, PRIMARY_ROWS
from dynamics.scores import ALL_CSFS
from dynamics.tabular_probe import NEAR_SETS

HERE = Path(__file__).resolve().parent
D1 = f"D1:{PRIMARY}"
CONTROLS = ("D1:logistic", "D1:trees")
#: The data-free score with the lowest mean regret in the known results.
COUNTERPART = "GEN"
FUTILITY = 2.0
PRIOR_GAIN = 1.0
FITTED_GAIN = 1.0
TIME_LIMIT_HOURS = 6.0
TEST_RUNS = 18


def read_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per run, row count, draw, and score: the mean AUROC of each group of sets.

    Returns:
        The table of the scores, and one row for each run with its seconds and its
        errors by row count.
    """
    rows, runs = [], []
    for path in sorted(Path(root).glob("runs/*/fewrow/results.json")):
        out = json.loads(path.read_text())
        errors: dict[int, set[str]] = {}
        for one in out["arms"]:
            count = one["rows_per_class"]
            errors.setdefault(count, set()).update(one["errors"])
            frame = pd.DataFrame(one["metrics"])
            if frame.empty:
                continue
            for group, sets in (("far", FAR_SETS), ("near", NEAR_SETS)):
                part = frame[frame["set"].isin(sets)].groupby("score")
                for score, value in (part["auroc"].mean() * 100).items():
                    rows.append({"run": out["run"], "rows": count, "draw": one["draw"],
                                 "group": group, "score": score, "auroc": value,
                                 "tied": part["tied"].max()[score]})
        runs.append({"run": out["run"], "seconds": out.get("seconds", float("nan")),
                     "draws": out["settings"]["draws"],
                     "errors": {count: sorted(names)
                                for count, names in errors.items()}})
    return pd.DataFrame(rows), pd.DataFrame(runs)


def wide(results: pd.DataFrame, count: int, group: str = "far",
         value: str = "auroc") -> pd.DataFrame:
    """Runs as rows and scores as columns at one row count: the mean over the draws.

    A score has a value for a run only if every draw of the run has it.
    """
    part = results[(results["group"] == group) & (results["rows"] == count)]
    if part.empty:
        return pd.DataFrame()
    draws = part.groupby("run")["draw"].nunique()
    cells = part.groupby(["run", "score"])[value]
    table = cells.mean().where(cells.count() == draws.reindex(
        cells.count().index.get_level_values("run")).to_numpy()).unstack("score")
    table.insert(0, "recipe", [parse_run(run).recipe for run in table.index])
    return table


def complete(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop the roster scores that some run does not have. Return their names too."""
    missing = [name for name in ALL_CSFS if name not in table.columns
               or table[name].isna().any()]
    return table.drop(columns=[name for name in missing if name in table.columns]
                      ), missing


def data_free(results: pd.DataFrame) -> list[str]:
    """Roster scores whose AUROC is the same in every arm of each run."""
    part = results[(results["group"] == "far") & results["score"].isin(ALL_CSFS)]
    spread = part.groupby(["run", "score"])["auroc"].agg(lambda v: v.max() - v.min())
    widest = spread.groupby("score").max()
    return sorted(widest.index[widest < 1e-9])


def gates(results: pd.DataFrame, runs: pd.DataFrame) -> tuple[str, bool]:
    """The three gates on the development runs. Returns the text and "continue"."""
    table = wide(results, PRIMARY_ROWS)
    develop = table[table.index.isin(DEVELOPMENT)] if not table.empty else table
    if develop.empty or D1 not in develop or COUNTERPART not in develop:
        return "No development run with the primary score.", False
    shown = develop[["recipe", D1, *[c for c in CONTROLS if c in develop],
                     COUNTERPART]]
    sane = bool((develop[D1] >= SANITY).all())
    below = develop[COUNTERPART] - develop[D1]
    futile = bool((below > FUTILITY).all())
    seconds = runs[runs["run"].isin(DEVELOPMENT)]["seconds"].mean()
    hours = seconds * TEST_RUNS / 3600
    lines = [
        f"At {PRIMARY_ROWS} rows for each class, far AUROC:\n\n"
        + block(shown.reset_index(), 1),
        f"- **Sanity ({D1} at least {SANITY:.0f}):** {'pass' if sane else 'FAIL'}.",
        f"- **Futility (more than {FUTILITY} points below {COUNTERPART} on every "
        f"development run):** {'STOP, no test' if futile else 'continue'}. The "
        "differences are " + ", ".join(f"{v:.1f}" for v in below) + ".",
        f"- **Time:** {seconds:.0f} s for a development run, so about {hours:.1f} "
        f"GPU-hours for {TEST_RUNS} test runs. "
        + (f"This is above {TIME_LIMIT_HOURS:.0f}, so the draws go to 1."
           if hours > TIME_LIMIT_HOURS else "This is inside the limit.")]
    return "\n".join(lines), sane and not futile


def rule_one(table: pd.DataFrame) -> str:
    """The table and the verdict of rule 1 for the runs in `table`."""
    kept, missing = complete(table)
    summary = regret_summary(kept)
    new = [name for name in summary.index if ":" in name]
    best = summary.loc[[n for n in ALL_CSFS if n in summary.index]
                       ].sort_values("W").index[:6]
    shown = summary.loc[[*new, *[n for n in best if n not in new]]]
    return (block(shown.reset_index(), 2)
            + f"\n\n**{D1}: {read_direction(summary, D1, COUNTERPART)}.**"
            + ("\n\nRoster scores that a run could not compute, left out: "
               + ", ".join(missing) + "." if missing else ""))


def rule_two(table: pd.DataFrame) -> str:
    """The prior helps if D1 with TabPFN beats both controls by the fixed margin."""
    found = [paired(table, D1, control) for control in CONTROLS if control in table]
    if not found or D1 not in table:
        return "not computed"
    helps = all(row["difference"] >= PRIOR_GAIN and row["low"] > 0 for row in found)
    return (block(pd.DataFrame(found), 2) + "\n\n**The prior "
            + ("helps" if helps else "does not help") + ".**")


def efficiency(results: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    """Mean far AUROC of chosen scores at each row count, over the given runs."""
    free = data_free(results)
    rows = []
    for count in sorted(results["rows"].unique(), key=lambda c: (c == ALL_ROWS, c)):
        table = wide(results, count)
        table = table[table.index.isin(runs)]
        if table.empty:
            continue
        means = table.drop(columns="recipe").mean()
        fitted = means[[n for n in ALL_CSFS if n in means.index and n not in free
                        and not table[n].isna().any()]]
        row: dict[str, object] = {"rows": "all" if count == ALL_ROWS else str(count)}
        for name in (D1, *CONTROLS, "MahaPP", "kNN", COUNTERPART):
            row[name] = means.get(name, float("nan"))
        row["best fitted"] = fitted.max() if len(fitted) else float("nan")
        row["which"] = fitted.idxmax() if len(fitted) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "fewrow_report.md"))
    args = parser.parse_args()
    results, runs = read_results(Path(args.results_root))
    if results.empty:
        raise SystemExit(f"no few-row results under {args.results_root}/runs")
    recipes = {run: parse_run(run).recipe for run in runs["run"]}
    test_runs = [run for run, recipe in recipes.items() if recipe in TEST_RECIPES]
    text, go = gates(results, runs)
    sections = [("Runs and gates", (
        f"{len(runs)} runs with few-row results: {len(test_runs)} test runs, "
        f"{sum(run in DEVELOPMENT for run in recipes)} development runs.\n\n" + text
        + "\n\nScores that need no fit rows (the same AUROC in every arm): "
        + ", ".join(data_free(results)) + "."))]
    counts = sorted(c for c in results["rows"].unique() if c != ALL_ROWS)
    if test_runs:
        for count in sorted(counts, key=lambda c: c != PRIMARY_ROWS):
            table = wide(results, count)
            table = table[table.index.isin(test_runs)]
            role = "primary" if count == PRIMARY_ROWS else "secondary"
            sections.append((f"Rule 1 at {count} rows for each class ({role})", (
                "Far target, test runs. `M` is the mean regret over the recipes, "
                "and `W` is the regret of the worst recipe. The six roster scores "
                "with the lowest `W` follow the new scores:\n\n" + rule_one(table))))
            sections.append((f"Rule 2 at {count} rows for each class ({role})", (
                "Paired differences of the far AUROC over the test runs:\n\n"
                + rule_two(table))))
    chosen = test_runs or [run for run in recipes if run in DEVELOPMENT]
    table = efficiency(results, chosen)
    gain = table[(table["rows"] != "all") & (table["best fitted"] - table[COUNTERPART]
                                             >= FITTED_GAIN)]
    sections.append(("Question 3: rows that a fitted score needs", (
        f"Mean far AUROC over the {'test' if test_runs else 'development'} runs. "
        f"The arm with all rows has D1 for the controls only.\n\n"
        + block(table, 1) + "\n\nSmallest row count at which the best fitted score "
        f"is at least {FITTED_GAIN} point above {COUNTERPART}: "
        + (gain["rows"].iloc[0] if len(gain) else "none of the few-row counts")
        + ".")))
    ties = results[(results["group"] == "far") & results["score"].str.contains(":")]
    worst = ties.groupby("score")["tied"].max().sort_values(ascending=False)
    marked = [name for name, share in worst.items() if share > TIE_LIMIT]
    sections.append(("Ties", (
        "Largest share of tied pairs of one ID row and one OOD row. A share above "
        f"{TIE_LIMIT:.0%} marks a score whose AUROC holds many ties: "
        + (", ".join(f"**{name}**" for name in marked) if marked else "none")
        + ".\n\n" + block(worst.rename("tied").reset_index(), 3))))
    failed = {count: sorted({name for errors in runs["errors"]
                             for name in errors.get(count, [])})
              for count in [*counts, ALL_ROWS]}
    sections.append(("Scores with an error", "\n".join(
        f"- **{'all' if count == ALL_ROWS else count} rows:** "
        + (", ".join(names) if names else "none") + "."
        for count, names in failed.items())))
    text = ("# Few-row probe\n\nThe rules were fixed before these results. "
            + ("" if go else "**A gate on the development runs does not allow the "
               "test.**") + "\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
