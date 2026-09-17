import time
import os
import json

from dotenv import load_dotenv
from functools import wraps
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# =========================
# Model
# =========================

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("GLM_MODEL"),
    api_key=os.getenv("GLM_API_KEY"),
    base_url=os.getenv("GLM_BASE_URL"),
)


# =========================
# State
# =========================

class RecommendationState(TypedDict, total=False):
    customer_name: str
    customer_status: str
    primary_result: dict
    final_recommendation: str
    final_answer: str


# =========================
# Middleware
# =========================

def tool_middleware(tool_func):

    @wraps(tool_func)
    def wrapped(*args, **kwargs):
        tool_name = tool_func.__name__

        print(f"\n[MIDDLEWARE] Calling tool: {tool_name}")
        print(f"[MIDDLEWARE] args={args}, kwargs={kwargs}")

        max_retries = 2
        attempt = 0

        while True:
            attempt += 1
            start = time.time()

            try:
                result = tool_func(*args, **kwargs)

                duration = time.time() - start

                print(f"[MIDDLEWARE] Tool succeeded: {tool_name}")
                print(f"[MIDDLEWARE] Attempt: {attempt}")
                print(f"[MIDDLEWARE] Duration: {duration:.3f}s")

                return {
                    "success": True,
                    "data": result,
                    "error": None,
                }

            except TimeoutError as e:
                duration = time.time() - start

                print(f"[MIDDLEWARE] Timeout: {tool_name}")
                print(f"[MIDDLEWARE] Attempt: {attempt}")
                print(f"[MIDDLEWARE] Duration: {duration:.3f}s")

                if attempt <= max_retries:
                    print("[MIDDLEWARE] Retrying...")
                    continue

                return {
                    "success": False,
                    "data": None,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "retryable": True,
                    },
                }

            except Exception as e:
                duration = time.time() - start

                print(f"[MIDDLEWARE] Tool failed: {tool_name}")
                print(f"[MIDDLEWARE] Error: {e}")
                print(f"[MIDDLEWARE] Duration: {duration:.3f}s")

                return {
                    "success": False,
                    "data": None,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "retryable": False,
                    },
                }

    return wrapped


# =========================
# Tools
# =========================

@tool_middleware
def get_customer_status(customer_name: str) -> str:
    """Get the current status of a customer."""

    demo_data = {
        "Customer A": "Interested in cloud modernization and AI use cases.",
        "Customer B": "Existing cloud customer with hybrid-cloud workloads.",
    }

    return demo_data.get(
        customer_name,
        "No customer information found.",
    )


@tool_middleware
def recommend_product(customer_status: str) -> str:
    """Recommend a cloud product based on the customer's current status."""

    # Simulate a temporary service failure
    raise TimeoutError("Recommendation service timed out.")


@tool_middleware
def fallback_recommend_product(customer_status: str) -> str:
    """Provide a conservative fallback recommendation when the primary service is unavailable."""

    if "AI" in customer_status:
        return "ModelArts"

    return "No fallback recommendation available."


# =========================
# LangGraph Nodes
# =========================

def get_status_node(state: RecommendationState):
    result = get_customer_status(
        customer_name=state["customer_name"]
    )

    return {
        "customer_status": result["data"]
    }


def primary_recommendation_node(state: RecommendationState):
    result = recommend_product(
        customer_status=state["customer_status"]
    )

    return {
        "primary_result": result
    }


def route_after_primary(state: RecommendationState):
    if state["primary_result"]["success"]:
        return "success"

    return "fallback"


def primary_success_node(state: RecommendationState):
    return {
        "final_recommendation":
            state["primary_result"]["data"]
    }


def fallback_node(state: RecommendationState):
    result = fallback_recommend_product(
        customer_status=state["customer_status"]
    )

    return {
        "final_recommendation": result["data"]
    }


def generate_response_node(state: RecommendationState):
    prompt = f"""
You are a presales assistant.

Customer: {state['customer_name']}
Customer status: {state['customer_status']}
Recommended product: {state['final_recommendation']}

Explain concisely why the recommended product could be relevant
based only on the provided information.

Do not claim that the product is an ideal or perfect fit.
Do not invent customer requirements, architecture, budget, or business outcomes.
Distinguish clearly between known facts and inferred relevance.
"""

    response = model.invoke(prompt)

    return {
        "final_answer": response.content
    }


# =========================
# Build Workflow
# =========================

builder = StateGraph(RecommendationState)

builder.add_node(
    "get_status",
    get_status_node,
)

builder.add_node(
    "primary_recommendation",
    primary_recommendation_node,
)

builder.add_node(
    "primary_success",
    primary_success_node,
)

builder.add_node(
    "fallback",
    fallback_node,
)

builder.add_node(
    "generate_response",
    generate_response_node,
)


builder.add_edge(
    START,
    "get_status",
)

builder.add_edge(
    "get_status",
    "primary_recommendation",
)

builder.add_conditional_edges(
    "primary_recommendation",
    route_after_primary,
    {
        "success": "primary_success",
        "fallback": "fallback",
    },
)

builder.add_edge(
    "primary_success",
    "generate_response",
)

builder.add_edge(
    "fallback",
    "generate_response",
)

builder.add_edge(
    "generate_response",
    END,
)


# =========================
# Checkpoint
# =========================

checkpointer = InMemorySaver()

workflow = builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "demo-001"
    }
}


# =========================
# Run
# =========================

result = workflow.invoke(
    {
        "customer_name": "Customer A"
    },
    config=config,
)


# =========================
# Output
# =========================

state = workflow.get_state(config)

print("\n=== BUSINESS STATE ===")
print(
    json.dumps(
        state.values,
        indent=2,
        ensure_ascii=False,
    )
)

print("\n=== CHECKPOINT INFO ===")
print("thread_id:", state.config["configurable"]["thread_id"])
print("checkpoint_id:", state.config["configurable"]["checkpoint_id"])
print("next:", state.next)

print("\n=== FINAL ANSWER ===")
print(result["final_answer"])