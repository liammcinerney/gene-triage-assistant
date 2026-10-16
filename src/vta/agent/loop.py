import json
import time

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

from src.vta.agent.prompts import SYSTEM_PROMPT
from src.vta.agent.tool_specs import TOOL_FUNCTIONS, TOOLS

load_dotenv()

# The SDK's default read timeout is 600s, far too long for an interactive
# tool. It already retries transient failures (429, 5xx, dropped connections)
# max_retries times with backoff, so we don't hand-roll that here.
client = Anthropic(timeout=60.0, max_retries=2)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 2048

# Safety cap: if Claude never settles on a final answer (a confused loop,
# or a bug in a tool that keeps returning something that looks incomplete),
# this stops us from burning API calls indefinitely.
MAX_ITERATIONS = 5

# A tool result is pasted into the model's context on every later round, so
# an oversized one costs tokens repeatedly. Cap it.
MAX_TOOL_RESULT_CHARS = 20_000


class AgentConfigError(RuntimeError):
    """A problem the caller must fix (e.g. a revoked API key).

    Raised instead of returned: retrying won't help, and a batch job such as
    the eval harness should abort immediately rather than fail every case.
    """


def _execute_tool(block, tool_functions: dict):
    """Run one tool call. Returns (output, is_error).

    Three different things can go wrong and they are NOT the same:
      - the tool ran and found nothing: {"found": False, ...}. A valid result,
        returned as normal content by the tool itself (is_error stays False).
      - the model asked for something broken (unknown tool, bad arguments) or
        the tool crashed: is_error=True, so the model can see it and recover.
      - the harness itself is broken: not caught here, it should be loud.
    """
    function = tool_functions.get(block.name)
    if function is None:
        return {"error": f"Unknown tool '{block.name}'. Available: {sorted(tool_functions)}"}, True
    try:
        return function(**block.input), False
    except Exception as e:  # tool boundary: report to the model, don't crash the loop
        return {"error": f"{type(e).__name__}: {e}"}, True


def _tool_result_text(output) -> str:
    text = json.dumps(output, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "... [truncated: result exceeded size limit]"
    return text


def run_agent(
    user_message: str,
    max_iterations: int = MAX_ITERATIONS,
    *,
    model: str = MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    tools: list = TOOLS,
    tool_functions: dict = TOOL_FUNCTIONS,
    temperature: float | None = None,
) -> dict:
    """Run the agent loop until Claude stops calling tools or we hit a limit.

    The keyword-only parameters exist so the eval harness can swap the model,
    the prompt, or wrap the tools (record/replay) without touching this code.

    Returns:
        {
          "status": "ok" | "max_iterations" | "truncated" | "refusal"
                    | "unexpected_stop" | "api_error",
          "answer": final text (partial if truncated), or None,
          "error": explanation when status != "ok", else None,
          "trace": [{"tool", "input", "output", "is_error", "duration_s"}, ...],
          "rounds": number of API calls made,
          "usage": {"input_tokens", "output_tokens"} summed over all rounds,
        }

    Raises:
        AgentConfigError: the API key was rejected (401/403).
    """
    messages = [{"role": "user", "content": user_message}]
    trace = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    succeeded_calls = set()
    rounds = 0

    def result(status, answer=None, error=None):
        return {
            "status": status,
            "answer": answer,
            "error": error,
            "trace": trace,
            "rounds": rounds,
            "usage": usage,
        }

    for _ in range(max_iterations):
        kwargs = dict(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            response = client.messages.create(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise AgentConfigError(
                f"Anthropic rejected the API key (HTTP {e.status_code}). "
                "Check ANTHROPIC_API_KEY in .env."
            ) from e
        except anthropic.APIError as e:
            # Connection, timeout, rate limit, 5xx: the SDK already retried.
            return result("api_error", error=f"{type(e).__name__}: {e}")

        rounds += 1
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason in ("end_turn", "stop_sequence"):
                return result("ok", answer=text)
            if response.stop_reason == "max_tokens":
                return result("truncated", answer=text, error=f"Hit max_tokens={MAX_TOKENS}; answer is cut off")
            if response.stop_reason == "refusal":
                return result("refusal", answer=text or None, error="Model refused to answer")
            return result("unexpected_stop", answer=text or None, error=f"stop_reason={response.stop_reason}")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            call_key = (block.name, json.dumps(block.input, sort_keys=True))
            started = time.monotonic()
            if call_key in succeeded_calls:
                # Same call, same arguments, already succeeded: re-running adds
                # nothing but cost. Tell the model so it moves on.
                output, is_error = {"error": "Identical call already succeeded; reuse its earlier result or change the query."}, True
            else:
                output, is_error = _execute_tool(block, tool_functions)
                failed_softly = isinstance(output, dict) and output.get("found") is False
                if not is_error and not failed_softly:
                    succeeded_calls.add(call_key)

            trace.append(
                {
                    "tool": block.name,
                    "input": block.input,
                    "output": output,
                    "is_error": is_error,
                    "duration_s": round(time.monotonic() - started, 2),
                }
            )

            tool_result = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _tool_result_text(output),
            }
            if is_error:
                tool_result["is_error"] = True
            tool_results.append(tool_result)

        messages.append({"role": "user", "content": tool_results})

    return result("max_iterations", error=f"Hit max_iterations={max_iterations} without a final answer")


if __name__ == "__main__":
    print("Variant Triage Assistant -- ask about a gene (blank line to quit)\n")

    while True:
        question = input("> ").strip()
        if not question:
            break

        try:
            r = run_agent(question)
        except AgentConfigError as e:
            raise SystemExit(f"\nConfiguration error: {e}")

        print()
        print(f"=== {r['rounds']} round(s), {len(r['trace'])} tool call(s), "
              f"{r['usage']['input_tokens']} in / {r['usage']['output_tokens']} out tokens ===")
        for step in r["trace"]:
            flag = "  [ERROR]" if step["is_error"] else ""
            print(f"  {step['tool']}({step['input']}){flag}")
        print()

        if r["status"] != "ok":
            print(f"[{r['status']}] {r['error']}")
        if r["answer"]:
            print(r["answer"])
        print()
