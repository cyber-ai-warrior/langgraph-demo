import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
model = init_chat_model(MODEL, model_provider="groq")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}

def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()
    return graph

def save_graph_png_matplotlib(graph, filename="chatbot_graph"):
    """Draw LangGraph with matplotlib (pure Python)."""
    g = nx.DiGraph()

    # Extract nodes & edges
    nodes = list(graph.get_graph().nodes)
    edges = list(graph.get_graph().edges)

    g.add_nodes_from(nodes)
    for src, dst, _, _ in edges:  # Unpack tuple
        g.add_edge(src, dst)

    # Draw
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(g, seed=42)  # Fixed layout for consistency
    nx.draw(
        g, pos,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        font_size=10,
        arrows=True
    )
    # Save
    output_dir = Path("graph-diagram")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"{filename}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Graph PNG saved at: {path}")


if __name__ == "__main__":
    graph = build_graph()
    save_graph_png_matplotlib(graph)
    result = graph.invoke({"messages": [{"role": "user", "content": "what is 5+5"}]})
    print("output:- "+ result['messages'][-1].content)
