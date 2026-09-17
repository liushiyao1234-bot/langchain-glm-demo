## main.py

`main.py` demonstrates a basic LangChain agent with tool calling and middleware-based error handling.

It includes:

- A GLM model connected through an OpenAI-compatible API.
- Multiple tools for retrieving customer information, product information, and generating recommendations.
- Agent-driven tool selection using `create_agent`.
- Sequential and fallback tool calling.
- A custom middleware layer that:
  - Logs tool calls.
  - Records execution time.
  - Retries `TimeoutError` failures.
  - Converts tool failures into a standardized structured result.
- A fallback recommendation tool that is used when the primary recommendation service fails.
- Message-level execution output showing `HumanMessage`, `AIMessage`, `ToolMessage`, and tool call IDs.

The purpose of this file is to demonstrate how LangChain manages the agent loop, tool selection, multi-step tool calling, and tool failure recovery.


## workflow_demo.py

`workflow_demo.py` demonstrates a deterministic LangGraph workflow built on top of the same tool and middleware concepts.

Instead of allowing the LLM to decide every execution step, the workflow explicitly defines the execution path:

```text
START
  ↓
Get Customer Status
  ↓
Primary Recommendation
  ↓
Success?
 ├─ Yes → Primary Result
 └─ No  → Fallback Recommendation
              ↓
       Generate Final Response
              ↓
             END

All customer names, product names, and business scenarios in this repository are anonymized or synthetic and are used only for demonstration purposes.