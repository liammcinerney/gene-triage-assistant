"""
Minimal tool calling with the raw Anthropic SDK: one tool, two API calls,
no loop and no helpers.

This is the mechanism that src/gta/agent/loop.py generalises, so it is the
quickest way to understand that file:
  1. Send the question plus the tool's schema.
  2. If stop_reason is "tool_use", run the function ourselves and send back a
     tool_result tied to the call's id.
  3. The model then answers using the result.

Run from the repo root:  python -m examples.raw_tool_calling
"""

import json

from anthropic import Anthropic
from dotenv import load_dotenv

from src.gta.tools.gene_lookup import gene_lookup

load_dotenv()

client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"

# The tool schema: this is the ONLY information Claude has about gene_lookup.
# It never sees the Python source. Everything it knows about when and how to
# call this tool comes from "description" and "input_schema" below.
GENE_LOOKUP_TOOL = {
    "name": "gene_lookup",
    "description": (
        "Look up metadata for a human gene by its official HGNC symbol "
        "(e.g. 'BRCA1', 'TP53', 'CFTR'). Returns the gene's full name, a "
        "functional summary, known aliases, genomic location, and gene type. "
        "Only accepts official gene symbols, not full gene names, rsIDs, or "
        "protein names. Only covers human genes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Official HGNC gene symbol, e.g. 'BRCA1'.",
            }
        },
        "required": ["symbol"],
    },
}


# Dispatch table: tool name (as Claude will refer to it) -> the Python
# function that actually does the work. This is the one place that couples
# "what Claude can ask for" to "what code runs."
TOOL_FUNCTIONS = {
    "gene_lookup": gene_lookup,
}


def main():
    question = input("Ask about a gene: ").strip()
    if not question:
        question = "What does the BRCA1 gene do?"

    messages = [{"role": "user", "content": question}]

    # --- Round 1: give Claude the question and the tool, see what it wants ---
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[GENE_LOOKUP_TOOL],
        messages=messages,
    )

    print("=== round 1 ===")
    print("stop_reason:", response.stop_reason)

    if response.stop_reason != "tool_use":
        # Claude answered directly, nothing to execute.
        print(response.content[0].text)
        return

    # The assistant's turn (including the tool_use block(s)) must be echoed
    # back verbatim as conversation history. The API rejects a tool_result
    # that doesn't correspond to a tool_use it can see in the prior turn.
    messages.append({"role": "assistant", "content": response.content})

    # A turn can contain multiple tool_use blocks (parallel tool calls).
    # We execute every one and collect a matching tool_result for each.
    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue

        print(f"executing tool: {block.name}({block.input})")
        function = TOOL_FUNCTIONS[block.name]
        result = function(**block.input)

        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,  # ties this result to that specific call
                "content": json.dumps(result),
            }
        )

    # Tool results go back as a "user" turn -- from the API's perspective,
    # tool output is external input, the same category as a human message.
    messages.append({"role": "user", "content": tool_results})

    # --- Round 2: Claude has the tool output, now it can actually answer ---
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[GENE_LOOKUP_TOOL],
        messages=messages,
    )

    print()
    print("=== round 2 ===")
    print("stop_reason:", response.stop_reason)
    print()
    print(response.content[0].text)


if __name__ == "__main__":
    main()
