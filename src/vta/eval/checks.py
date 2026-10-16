"""
Deterministic checks on one agent run: tool-call correctness and citation validity.

Everything here is computed from the trace and the answer text. No model is
involved, so these are exact, free and reproducible. What they cannot do is
judge whether a claim is faithful to its source -- that is the judge's job.
"""

import re
from dataclasses import dataclass

from src.vta.eval.cases import EvalCase

MAX_TOOL_CALLS = 4  # a run needing more than this is probably flailing

# "PMID: 123456", "(PMID 123456)", "PMIDs 123456, 234567 and 345678"
_PMID_GROUP = re.compile(r"PMIDs?[:\s#]*(\d{6,9}(?:\s*(?:,|;|and|&)\s*(?:PMID\s*)?\d{6,9})*)", re.IGNORECASE)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def extract_pmids(text: str) -> set[str]:
    """PMIDs the answer cites. Only numbers introduced by 'PMID' count, so
    '10,000 patients' or a year is never mistaken for a citation."""
    found = set()
    for match in _PMID_GROUP.finditer(text or ""):
        found.update(re.findall(r"\d{6,9}", match.group(1)))
    return found


def retrieved_pmids(trace: list[dict]) -> set[str]:
    """PMIDs the agent was actually shown by a successful literature_search."""
    pmids = set()
    for step in trace:
        output = step["output"]
        if step["tool"] == "literature_search" and not step["is_error"] and isinstance(output, dict) and output.get("found"):
            pmids.update(str(r["pmid"]) for r in output["results"])
    return pmids


def _arg_check_passes(check, trace: list[dict]) -> bool:
    for step in trace:
        if step["tool"] != check.tool or check.arg not in step["input"]:
            continue
        value = str(step["input"][check.arg]).lower()
        if check.equals is not None and value.strip() != check.equals.lower():
            continue
        if not all(term.lower() in value for term in check.contains_all):
            continue
        return True
    return False


def run_checks(case: EvalCase, run: dict) -> list[Check]:
    trace = run["trace"]
    called = [step["tool"] for step in trace]
    checks = [Check("run_completed", run["status"] == "ok", f"status={run['status']}" + (f": {run['error']}" if run["error"] else ""))]

    for tool in case.must_call:
        checks.append(Check(f"called:{tool}", tool in called, f"calls={called}"))
    for tool in case.must_not_call:
        checks.append(Check(f"not_called:{tool}", tool not in called, f"calls={called}"))
    for ac in case.arg_checks:
        want = f"{ac.arg}=={ac.equals}" if ac.equals else f"{ac.arg} contains {list(ac.contains_all)}"
        seen = [s["input"].get(ac.arg) for s in trace if s["tool"] == ac.tool]
        checks.append(Check(f"arg:{ac.tool}.{want}", _arg_check_passes(ac, trace), f"saw={seen}"))

    if not case.fail_tools:
        errors = [s["tool"] for s in trace if s["is_error"]]
        checks.append(Check("no_tool_errors", not errors, f"errored={errors}"))
    checks.append(Check("tool_budget", len(trace) <= MAX_TOOL_CALLS, f"{len(trace)} calls"))

    cited = extract_pmids(run["answer"] or "")
    retrieved = retrieved_pmids(trace)
    fabricated = cited - retrieved
    checks.append(Check("citations_valid", not fabricated, f"cited-but-never-retrieved={sorted(fabricated)}"))
    if case.requires_citations:
        checks.append(Check("citations_present", bool(cited & retrieved), f"valid citations={sorted(cited & retrieved)}"))

    return checks
