import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
model = init_chat_model(MODEL, model_provider="groq")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool()
def multiply_my_number(a: int, b: int) -> int:
    """
    Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

tools = [multiply_my_number]
llm_with_tools = model.bind_tools(tools)
tool_node = ToolNode(tools=[multiply_my_number])

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    return graph_builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    res = graph.invoke({"messages": [("user", "what is water?")]})
    # res = graph.invoke({"messages": [("user", "what is 2*5?")]})
    result = res["messages"][-1].content
    print(result)
