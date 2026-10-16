import json

from anthropic import Anthropic
from dotenv import load_dotenv

from src.vta.agent.prompts import SYSTEM_PROMPT
from src.vta.agent.tool_specs import TOOL_FUNCTIONS, TOOLS

load_dotenv()

client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"

# Safety cap: if Claude never settles on a final answer (a confused loop,
# or a bug in a tool that keeps returning something that looks incomplete),
# this stops us from burning API calls indefinitely.
MAX_ITERATIONS = 5


def run_agent(user_message: str, max_iterations: int = MAX_ITERATIONS) -> dict:
    """Run the agent loop until Claude stops calling tools or we hit the cap.

    Returns:
        {
          "answer": final text, or None if the iteration cap was hit,
          "trace": [{"tool", "input", "output"}, ...] for every tool call,
          "rounds": number of API calls made,
        }
    """
    messages = [{"role": "user", "content": user_message}]
    trace = []

    for round_num in range(1, max_iterations + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"answer": final_text, "trace": trace, "rounds": round_num}

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            function = TOOL_FUNCTIONS[block.name]
            result = function(**block.input)
            trace.append({"tool": block.name, "input": block.input, "output": result})

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": None,
        "trace": trace,
        "rounds": max_iterations,
        "error": f"Hit max_iterations={max_iterations} without a final answer",
    }


if __name__ == "__main__":
    print("Variant Triage Assistant -- ask about a gene (blank line to quit)\n")

    while True:
        question = input("> ").strip()
        if not question:
            break

        result = run_agent(question)

        print()
        print(f"=== {result['rounds']} round(s), {len(result['trace'])} tool call(s) ===")
        for step in result["trace"]:
            print(f"  {step['tool']}({step['input']})")
        print()

        if result["answer"]:
            print(result["answer"])
        else:
            print("ERROR:", result["error"])
        print()
