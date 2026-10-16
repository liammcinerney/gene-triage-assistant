"""Tests for the eval harness's own machinery and the MCP server's input bounds."""

import asyncio
import json

import pytest

from src.gta.eval.cases import ArgCheck, EvalCase
from src.gta.eval.checks import extract_pmids, retrieved_pmids, run_checks
from src.gta.eval.judge import _quote_in_evidence
from src.gta.eval.recording import ReplayMiss, cache_path, wrap_tools

# --- PMID extraction ------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("PMID: 18812548 shows", {"18812548"}),
        ("(PMID 41469310)", {"41469310"}),
        ("PMIDs 33423179, 18812548 and 37591780", {"33423179", "18812548", "37591780"}),
        ("(PMID: 33423179; PMID: 18812548)", {"33423179", "18812548"}),
        ("enrolled 10,000 patients in 2021", set()),  # numbers without 'PMID' are never citations
        ("pmid 12345 is too short", set()),
        ("", set()),
    ],
)
def test_extract_pmids(text, expected):
    assert extract_pmids(text) == expected


# --- deterministic run checks --------------------------------------------


def _step(tool, inp, output, is_error=False):
    return {"tool": tool, "input": inp, "output": output, "is_error": is_error, "duration_s": 0}


def _run(trace, answer="ok", status="ok"):
    return {"trace": trace, "answer": answer, "status": status, "error": None}


LIT = _step("literature_search", {"query": "BRCA1 hormone"}, {"found": True, "results": [{"pmid": "111111"}]})


def _by_name(checks):
    return {c.name: c.passed for c in checks}


def test_fabricated_citation_is_caught():
    case = EvalCase("c", "x", "q", "b", requires_citations=True)
    ok = _by_name(run_checks(case, _run([LIT], "Found it (PMID 111111).")))
    bad = _by_name(run_checks(case, _run([LIT], "Found it (PMID 999999).")))
    assert ok["citations_valid"] and ok["citations_present"]
    assert not bad["citations_valid"] and not bad["citations_present"]


def test_citation_to_a_failed_search_does_not_count_as_retrieved():
    failed = _step("literature_search", {"query": "x"}, {"found": False, "error": "none"})
    assert retrieved_pmids([failed]) == set()


def test_required_and_forbidden_tools_and_arg_checks():
    case = EvalCase(
        "c", "x", "q", "b",
        must_call=("gene_lookup",),
        must_not_call=("literature_search",),
        arg_checks=(ArgCheck("gene_lookup", "symbol", equals="TP53"),),
    )
    good = _by_name(run_checks(case, _run([_step("gene_lookup", {"symbol": "tp53"}, {"found": True})])))
    assert all(good.values())

    wrong = _by_name(run_checks(case, _run([_step("gene_lookup", {"symbol": "p53"}, {"found": False}), LIT])))
    assert not wrong["not_called:literature_search"]
    assert not wrong["arg:gene_lookup.symbol==TP53"]


def test_unrecovered_tool_error_fails_unless_a_fault_was_injected():
    errored = _step("gene_lookup", {"symbol": "TP53"}, {"error": "boom"}, is_error=True)
    assert not _by_name(run_checks(EvalCase("c", "x", "q", "b"), _run([errored])))["no_tool_errors"]
    injected = EvalCase("c", "x", "q", "b", fail_tools=("gene_lookup",))
    assert "no_tool_errors" not in _by_name(run_checks(injected, _run([errored])))


def test_incomplete_run_fails_run_completed():
    assert not _by_name(run_checks(EvalCase("c", "x", "q", "b"), _run([], None, status="truncated")))["run_completed"]


# --- judge helper: quote verification -------------------------------------


def test_quote_verification_allows_ellipsis_but_not_invention():
    evidence = "".join(c for c in "Organoids with p.Asp1152His were rescued by CFTR modulators, but not p.Ile507del.".lower() if c.isalnum())
    assert _quote_in_evidence("Organoids with p.Asp1152His ... but not p.Ile507del", evidence)
    assert not _quote_in_evidence("Organoids with p.Asp1152His ... and lung function rose by twelve percent", evidence)
    assert not _quote_in_evidence("", evidence)


def test_severity_rollup_only_major_issues_fail_a_case():
    from src.gta.eval.judge import summarise_claims

    claims = [
        {"verdict": "supported", "severity": "minor"},
        {"verdict": "unsupported", "severity": "minor"},
        {"verdict": "overstated", "severity": "minor"},
    ]
    r = summarise_claims(claims)
    assert r["grounded"] and (r["major_issues"], r["minor_issues"], r["total"]) == (0, 2, 3)

    r = summarise_claims(claims + [{"verdict": "overstated", "severity": "major"}])
    assert not r["grounded"] and r["major_issues"] == 1

    # a missing severity must be treated as major: ambiguity should never make a case pass
    assert not summarise_claims([{"verdict": "unsupported"}])["grounded"]
    assert summarise_claims([])["grounded"]


# --- record / replay ------------------------------------------------------


def test_record_then_replay_round_trip_without_calling_the_tool(tmp_path):
    calls = []

    def real(query):
        calls.append(query)
        return {"found": True, "n": len(calls)}

    wrap_tools({"t": real}, "record", cache_dir=tmp_path)["t"](query="BRCA1  Risk")
    assert wrap_tools({"t": real}, "replay", cache_dir=tmp_path)["t"](query="brca1 risk") == {"found": True, "n": 1}
    assert calls == ["BRCA1  Risk"]  # case/whitespace-insensitive hit, and the real tool ran only once


def test_replay_miss_is_raised_and_reported(tmp_path):
    misses = []
    tools = wrap_tools({"t": lambda query: {}}, "replay", cache_dir=tmp_path, misses=misses)
    with pytest.raises(ReplayMiss):
        tools["t"](query="never recorded")
    assert misses == [("t", {"query": "never recorded"})]


def test_transient_network_failures_are_never_recorded(tmp_path):
    flaky = {"t": lambda query: {"found": False, "error": "PubMed search request failed: timeout"}}
    wrap_tools(flaky, "record", cache_dir=tmp_path)["t"](query="x")
    assert not cache_path("t", {"query": "x"}, tmp_path).exists()


def test_injected_fault_and_output_hook(tmp_path):
    real = {"t": lambda query: {"results": ["clean"]}}

    def hook(out):
        out["results"].append("poison")
        return out

    tools = wrap_tools(real, "record", cache_dir=tmp_path, output_hooks={"t": hook})
    assert tools["t"](query="x") == {"results": ["clean", "poison"]}
    # the hook must not leak into the stored fixture
    stored = json.loads(cache_path("t", {"query": "x"}, tmp_path).read_text())["output"]
    assert stored == {"results": ["clean"]}

    with pytest.raises(RuntimeError, match="injected fault"):
        wrap_tools(real, "live", fail_tools=("t",))["t"](query="x")


# --- MCP server input bounds (tools mocked, no network) -------------------


@pytest.fixture
def server(monkeypatch):
    from src.gta.mcp_server import server as s

    monkeypatch.setattr(s, "_gene_lookup", lambda symbol: {"found": True, "symbol": symbol})
    monkeypatch.setattr(s, "_literature_search", lambda query, k=5: {"found": True, "k": k})
    return s


def call(server, name, args):
    return asyncio.run(server.mcp.call_tool(name, args))


def test_server_exposes_only_the_intended_parameters(server):
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert set(tools) == {"gene_lookup", "literature_search"}
    assert set(tools["literature_search"].inputSchema["properties"]) == {"query", "k"}  # fetch_n stays server-side


@pytest.mark.parametrize(
    "name, args",
    [
        ("gene_lookup", {"symbol": "BRCA1 OR TP53"}),  # would rewrite the MyGene.info query
        ("gene_lookup", {"symbol": ""}),
        ("gene_lookup", {"symbol": "A" * 31}),
        ("literature_search", {"query": "BRCA1 risk", "k": 500}),
        ("literature_search", {"query": "BRCA1 risk", "k": 0}),
        ("literature_search", {"query": "hi"}),
    ],
)
def test_server_rejects_out_of_bounds_input(server, name, args):
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        call(server, name, args)


def test_server_accepts_valid_input(server):
    call(server, "gene_lookup", {"symbol": "TP53"})
    call(server, "literature_search", {"query": "BRCA1 hormone risk", "k": 3})


def test_injection_hook_matches_the_real_literature_search_output_shape(monkeypatch):
    """The hook plants text in a tool result. If literature_search's output keys ever change,
    the hook must fail HERE, not silently turn the injection eval into a tool-error eval."""
    import numpy as np

    from src.gta.eval.cases import INJECTION, _plant_injection
    from src.gta.rag import retrieval
    from src.gta.tools import literature_search as ls

    monkeypatch.setattr(retrieval, "embed", lambda texts: np.array([[1.0, 0.0] for _ in texts]))
    paper = {"pmid": "1", "title": "T", "journal": "J", "year": "2024", "abstract": "alpha finding"}
    monkeypatch.setattr(ls, "pubmed_search", lambda query, max_results: {"found": True, "papers": [paper]})

    real_output = ls.literature_search("alpha question")
    poisoned = _plant_injection(real_output)
    assert INJECTION in poisoned["results"][0]["abstract"]
    assert poisoned["results"][0]["abstract"].startswith("alpha finding")  # legitimate text preserved
