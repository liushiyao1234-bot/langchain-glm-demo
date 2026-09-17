import os
import time
from functools import wraps

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


load_dotenv()

model = ChatOpenAI(
    model=os.getenv("GLM_MODEL"),
    api_key=os.getenv("GLM_API_KEY"),
    base_url=os.getenv("GLM_BASE_URL"),
)


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
def get_product_info(product_name: str) -> str:
    """Get basic information about a cloud product."""

    demo_data = {
        "Product A": "A distributed relational database designed for high availability and scalability.",
        "Product B": "An AI development platform for model training, deployment, and lifecycle management.",
    }

    return demo_data.get(
        product_name,
        "No product information found.",
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
        return "Product B"

    return "No fallback recommendation available."


agent = create_agent(
    model=model,
    tools=[
        get_customer_status,
        get_product_info,
        recommend_product,
        fallback_recommend_product,
    ],
    system_prompt=(
        "You are a presales assistant. "
        "Use tools to obtain factual customer and product information. "

        "When asked to recommend a product for a customer, "
        "first retrieve the customer's current status, "
        "then use the recommendation tool. "

        "Product recommendations MUST come from recommend_product. "
        "If recommend_product fails, call fallback_recommend_product. "
        "Do not generate, infer, or suggest a recommendation yourself. "
        "Do not speculate about when a failed service will recover."
    ),
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Based on Customer A's current status, recommend a suitable product."
            }
        ]
    }
)


for message in result["messages"]:
    print("\n---")
    print(type(message).__name__)
    print(message)