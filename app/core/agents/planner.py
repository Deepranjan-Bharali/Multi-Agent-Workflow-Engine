# app/core/agents/planner.py
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from app.core.state import AgentState

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

async def planner_node(state: AgentState):
    # Retrieve all messages in the thread history
    messages = state.get("messages", [])

    system_prompt = SystemMessage(
        content="You are a planning agent. Review the conversation history and the user's latest message, then break down the task into a concise plan. Return ONLY the plan."
    )
    
    # Pass system_prompt followed by ALL messages in conversation history
    response = await llm.ainvoke([system_prompt] + messages)
    
    return {"current_plan": response.content}