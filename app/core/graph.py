# app/core/graph.py
from langgraph.graph import StateGraph, START, END
from app.core.state import AgentState
from app.core.agents.planner import planner_node
from app.core.agents.retriever import retriever_node  # Import the new node
from app.core.agents.executor import executor_node

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)       # Add the new node
workflow.add_node("executor", executor_node)

# Rewire the edges sequentially
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")            # Planner goes to Retriever
workflow.add_edge("retriever", "executor")           # Retriever goes to Executor
workflow.add_edge("executor", END)

def get_compiled_graph(checkpointer=None):
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()