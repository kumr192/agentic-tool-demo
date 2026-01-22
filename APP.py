import os
import json
import datetime as dt
import streamlit as st
from openai import OpenAI

# ---------- Tools you expose to the model ----------
def get_time() -> str:
    return dt.datetime.now().isoformat()

def calc(expression: str) -> str:
    # Very small safety guard: only allow a restricted set of chars
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        return "Rejected: expression contains disallowed characters."
    try:
        # WARNING: eval is dangerous in general. This is a demo with tight filtering.
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current server time in ISO format.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a basic arithmetic expression (+,-,*,/,% and parentheses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression, e.g. '(12*3)+4'."}
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_MAP = {
    "get_time": lambda args: get_time(),
    "calc": lambda args: calc(args.get("expression", "")),
}

# ---------- Agent loop ----------
def run_agent(user_text: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when useful. If you use a tool, do it explicitly."},
        {"role": "user", "content": user_text},
    ]

    tool_log = []

    # Simple loop: allow up to 3 tool calls max
    for _ in range(3):
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=messages,
            tools=TOOLS,
        )

        # Collect tool calls (if any)
        tool_calls = [item for item in resp.output if item.type == "tool_call"]
        if not tool_calls:
            # Final answer
            final_text = ""
            for item in resp.output:
                if item.type == "output_text":
                    final_text += item.text
            return final_text.strip(), tool_log

        # Execute each tool call, then feed results back
        for tc in tool_calls:
            name = tc.name
            args = tc.arguments or "{}"
            try:
                args_obj = json.loads(args)
            except Exception:
                args_obj = {}

            tool_log.append({"tool": name, "arguments": args_obj})

            if name not in TOOL_MAP:
                tool_result = f"Unknown tool: {name}"
            else:
                tool_result = TOOL_MAP[name](args_obj)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(tool_result),
            })

    return "Stopped: too many tool calls.", tool_log


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Agentic Tool-Calling Demo", page_icon="🛠️", layout="centered")
st.title("Agentic AI Demo (Tool Calling)")

st.write("Ask something like: 'What time is it?' or 'Compute (19*7)+3' or 'Use tools if needed.'")

user_text = st.text_input("Your prompt", placeholder="Try: What is (19*7)+3 and what time is it?")

if st.button("Run"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY environment variable.")
    elif not user_text.strip():
        st.warning("Type something first.")
    else:
        answer, tool_log = run_agent(user_text)

        if tool_log:
            st.subheader("Tool calls")
            st.json(tool_log)

        st.subheader("Answer")
        st.write(answer)
