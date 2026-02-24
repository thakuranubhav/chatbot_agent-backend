from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from ddgs import DDGS

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()


def _build_retry_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


http_session = _build_retry_session()

# ==========================
# LLM SETUP
# ==========================

eval_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="conversation"
)

gen_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

gen_model = ChatHuggingFace(llm=gen_llm)
model = ChatHuggingFace(llm=eval_llm)


# ==========================
# STATE DEFINITION
# ==========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================
# TOOLS
# ==========================

@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return str(results)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation {operation}"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price using Alpha Vantage."""
    symbol = symbol.strip().upper()
    if not symbol:
        return {"error": "Stock symbol is required"}

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=7TPUMPXXVKZK9BAJ"
    )

    try:
        response = http_session.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {"error": f"Stock API request failed: {exc}", "symbol": symbol}
    except ValueError:
        return {"error": "Stock API returned invalid JSON", "symbol": symbol}

    quote = data.get("Global Quote", {})
    if not quote:
        return {"error": "No quote data returned", "symbol": symbol, "raw": data}

    return {"symbol": symbol, "quote": quote}


# ==========================
# TOOL BINDING
# ==========================

tools = [get_stock_price, search_tool, calculator]

model_tool = gen_model.bind_tools(tools)
tool_node = ToolNode(tools)


# ==========================
# CHAT NODE
# ==========================

def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = model_tool.invoke(messages)
    return {"messages": [response]}


# ==========================
# MEMORY SAVER (IN-MEMORY)
# ==========================

checkpointer = MemorySaver()


# ==========================
# GRAPH STRUCTURE
# ==========================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ==========================
# THREAD LISTING (OPTIONAL)
# ==========================

def rec_all_thread():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_thread)
