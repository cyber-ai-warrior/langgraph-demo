import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

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

if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"messages": [{"role": "user", "content": "what is 5+5"}]})
    print("output:- "+ result['messages'][-1].content)
