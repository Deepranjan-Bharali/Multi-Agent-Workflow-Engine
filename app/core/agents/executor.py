# app/core/agents/executor.py
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from app.core.state import AgentState

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

async def executor_node(state: AgentState):
    plan = state.get("current_plan", "")
    documents = state.get("documents", [])
    messages = state.get("messages", [])

    # Format the retrieved documents into a single string
    context = "\n\n".join(documents) if documents else "No external context available."

    system_prompt = SystemMessage(
        content=f"""You are the execution agent. Use the conversation history, the plan, and the knowledge base context to answer the user.
        
        Plan: {plan}
        
        Knowledge Base Context:
        {context}
        """
    )
    
    response = await llm.ainvoke([system_prompt] + messages)
    
    return {
        "final_answer": response.content,
        "messages": [response]
    }