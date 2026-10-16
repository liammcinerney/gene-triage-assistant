"""
Record/replay for tool calls, so evals measure the agent and not the internet.

PubMed's results drift as papers are added and MyGene.info occasionally changes
a field. If an eval score moved, we could not tell whether the agent changed or
the data did. Modes:

  live    call the real tools, save nothing
  record  call the real tools and save each output under eval_data/tool_cache
  replay  never touch the network; serve saved outputs (default for evals)

Fixtures are keyed on (tool, normalised arguments). If the model words a query
differently from the recording, replay has no match: that is reported as a
replay miss (re-run with --mode record), not silently answered with live data.
"""

import copy
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("eval_data/tool_cache")


class ReplayMiss(RuntimeError):
    """Replay mode was asked for a tool call that was never recorded."""


def _normalise(value):
    # Case and whitespace differences in a query don't change what we'd fetch.
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def cache_path(tool: str, args: dict, cache_dir: Path = CACHE_DIR) -> Path:
    digest = hashlib.sha256(json.dumps(_normalise(args), sort_keys=True).encode()).hexdigest()[:12]
    return cache_dir / f"{tool}-{digest}.json"


def wrap_tools(
    tool_functions: dict,
    mode: str,
    *,
    cache_dir: Path = CACHE_DIR,
    fail_tools: tuple = (),
    output_hooks: dict | None = None,
    misses: list | None = None,
) -> dict:
    """Return a drop-in replacement for tool_functions with record/replay and fault injection.

    fail_tools:   tools that raise on every call (simulated outage).
    output_hooks: {tool: fn(output) -> output} applied after the real/recorded
                  output, e.g. to plant a prompt-injection string in a result.
    misses:       list that collects (tool, args) for every replay miss.
    """
    if mode not in ("live", "record", "replay"):
        raise ValueError(f"unknown mode {mode!r}")
    hooks = output_hooks or {}

    def make(name, fn):
        def wrapped(**kwargs):
            if name in fail_tools:
                raise RuntimeError(f"{name} is unavailable (injected fault)")

            path = cache_path(name, kwargs, cache_dir)
            if mode == "replay":
                if not path.exists():
                    if misses is not None:
                        misses.append((name, kwargs))
                    raise ReplayMiss(f"no recording for {name}({kwargs}); re-run with --mode record")
                output = json.loads(path.read_text())["output"]
            else:
                output = fn(**kwargs)
                # Never freeze a transient network failure into a fixture.
                transient = isinstance(output, dict) and "request failed" in str(output.get("error", ""))
                if mode == "record" and not transient:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"tool": name, "input": kwargs, "output": output}, indent=2))

            hook = hooks.get(name)
            return hook(copy.deepcopy(output)) if hook else output

        return wrapped

    return {name: make(name, fn) for name, fn in tool_functions.items()}
