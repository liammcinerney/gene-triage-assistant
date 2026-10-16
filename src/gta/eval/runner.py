"""
Eval runner.

    python -m src.gta.eval.runner --mode record      # first time: hit real APIs, save fixtures
    python -m src.gta.eval.runner                    # replay fixtures, deterministic checks + judge
    python -m src.gta.eval.runner --no-judge         # free: deterministic checks only
    python -m src.gta.eval.runner --baseline eval_results/<file>.json   # show what changed
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.gta.agent.loop import MODEL, AgentConfigError, run_agent
from src.gta.agent.prompts import SYSTEM_PROMPT
from src.gta.agent.tool_specs import TOOL_FUNCTIONS
from src.gta.eval.cases import CASES, CASES_BY_ID
from src.gta.eval.checks import run_checks
from src.gta.eval.recording import wrap_tools

RESULTS_DIR = Path("eval_results")


def run_case(case, *, mode, system_prompt, judge_model, fresh_judge=False):
    misses = []
    tools = wrap_tools(
        TOOL_FUNCTIONS,
        mode,
        fail_tools=case.fail_tools,
        output_hooks=case.output_hooks,
        misses=misses,
    )
    started = time.monotonic()
    run = run_agent(case.query, temperature=0, tool_functions=tools, system_prompt=system_prompt)
    elapsed = round(time.monotonic() - started, 1)

    base = {"id": case.id, "category": case.category, "seconds": elapsed, "agent_usage": run["usage"]}

    if misses:
        # The model asked for something we never recorded. Scoring this would
        # measure our fixtures, not the agent, so say so instead.
        return {**base, "outcome": "replay_miss", "misses": [f"{t}({a})" for t, a in misses]}

    checks = run_checks(case, run)
    verdict = None
    if judge_model and run["answer"]:
        from src.gta.eval.judge import JudgeError, evidence_from_trace, judge_answer

        try:
            verdict = judge_answer(
                case.query, evidence_from_trace(run["trace"]), run["answer"], case.behavior,
                model=judge_model, use_cache=not fresh_judge,
            )
        except JudgeError as e:
            # One unjudgeable case must not sink the batch; surface it instead.
            return {**base, "outcome": "judge_error", "error": str(e), "checks": [vars(c) for c in checks]}

    deterministic_ok = all(c.passed for c in checks)
    judge_ok = True if verdict is None else (verdict["grounded"] and verdict["behavior_pass"])
    return {
        **base,
        "outcome": "pass" if (deterministic_ok and judge_ok) else "fail",
        "checks": [vars(c) for c in checks],
        "verdict": verdict,
        "answer": run["answer"],
        "trace": run["trace"],
        "rounds": run["rounds"],
    }


def summarize(results):
    scored = [r for r in results if r["outcome"] in ("pass", "fail")]
    judged = [r for r in scored if r["verdict"]]
    out = {
        "cases": len(results),
        "passed": sum(r["outcome"] == "pass" for r in results),
        "replay_misses": sum(r["outcome"] == "replay_miss" for r in results),
        "judge_errors": sum(r["outcome"] == "judge_error" for r in results),
        "tool_checks_passed": sum(all(c["passed"] for c in r["checks"]) for r in scored),
        "scored": len(scored),
    }
    if judged:
        out["grounded"] = sum(r["verdict"]["grounded"] for r in judged)
        out["behavior_ok"] = sum(r["verdict"]["behavior_pass"] for r in judged)
        out["judged"] = len(judged)
        out["major_issues"] = sum(r["verdict"]["major_issues"] for r in judged)
        out["minor_issues"] = sum(r["verdict"]["minor_issues"] for r in judged)
    return out


def print_report(results, summary):
    print()
    for r in results:
        if r["outcome"] == "replay_miss":
            print(f"MISS  {r['id']:<30} no recording for {r['misses']}")
            continue
        if r["outcome"] == "judge_error":
            print(f"JERR  {r['id']:<30} {r['error']}")
            continue
        failed = [c["name"] for c in r["checks"] if not c["passed"]]
        v = r["verdict"]
        judge = ""
        if v:
            judge = f" judge: {v['counts']['supported']}/{v['total']} supported, {v['major_issues']} major, {v['minor_issues']} minor"
            if not v["behavior_pass"]:
                judge += ", BEHAVIOR FAIL"
        print(f"{r['outcome'].upper():<5} {r['id']:<30} tools:{'ok' if not failed else 'FAIL ' + ','.join(failed)}{judge}")
    print()
    print(f"passed {summary['passed']}/{summary['cases']}  |  tool+citation checks {summary['tool_checks_passed']}/{summary['scored']}", end="")
    if "judged" in summary:
        print(f"  |  grounded {summary['grounded']}/{summary['judged']}  |  behavior {summary['behavior_ok']}/{summary['judged']}  |  issues: {summary['major_issues']} major, {summary['minor_issues']} minor", end="")
    print()
    if summary["replay_misses"]:
        print(f"{summary['replay_misses']} case(s) hit a replay miss: re-run with --mode record")


def print_diff(baseline_path, results):
    old = {r["id"]: r["outcome"] for r in json.loads(Path(baseline_path).read_text())["results"]}
    changes = [(r["id"], old.get(r["id"], "absent"), r["outcome"]) for r in results if old.get(r["id"]) != r["outcome"]]
    print(f"\nvs baseline {baseline_path}:")
    print("  no change in any case outcome" if not changes else "")
    for cid, before, after in changes:
        print(f"  {cid}: {before} -> {after}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["replay", "record", "live"], default="replay")
    ap.add_argument("--cases", help="comma-separated case ids (default: all)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--fresh-judge", action="store_true", help="bypass the judge verdict cache")
    ap.add_argument("--baseline", help="earlier results JSON to diff outcomes against")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cases = [CASES_BY_ID[i] for i in args.cases.split(",")] if args.cases else CASES
    judge_model = None if args.no_judge else args.judge_model

    results = []
    try:
        for case in cases:
            print(f"running {case.id} ...", flush=True)
            results.append(run_case(case, mode=args.mode, system_prompt=SYSTEM_PROMPT, judge_model=judge_model, fresh_judge=args.fresh_judge))
    except AgentConfigError as e:
        raise SystemExit(f"\nAborting eval run: {e}")

    summary = summarize(results)
    print_report(results, summary)

    agent_tokens = sum(r["agent_usage"]["input_tokens"] + r["agent_usage"]["output_tokens"] for r in results)
    judge_tokens = sum(r["verdict"]["usage"]["input_tokens"] + r["verdict"]["usage"]["output_tokens"] for r in results if r.get("verdict"))
    print(f"tokens: agent {agent_tokens:,} ({MODEL}) | judge {judge_tokens:,} ({judge_model})")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{stamp}{'_' + args.label if args.label else ''}.json"
    path.write_text(json.dumps({"settings": vars(args), "summary": summary, "results": results}, indent=2, default=str))
    print(f"saved {path}")

    if args.baseline:
        print_diff(args.baseline, results)


if __name__ == "__main__":
    main()
