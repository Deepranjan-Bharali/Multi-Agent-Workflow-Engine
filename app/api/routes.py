# app/api/routes.py
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from app.schemas.requests import WorkflowRequest
from app.core.graph import get_compiled_graph
from app.core.memory import get_checkpointer, log_workflow_start, log_workflow_complete

router = APIRouter()

@router.post("/execute")
async def execute_workflow(request: WorkflowRequest):
    
    # 1. Create the log in Django's DB and get the ID
    log_id = await log_workflow_start(
        thread_id=request.thread_id, 
        project_id=request.project_id, 
        prompt=request.prompt
    )

    try:
        checkpointer = await get_checkpointer()
        app_graph = get_compiled_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": request.thread_id}}
        state_update = {"messages": [HumanMessage(content=request.prompt)]}
        
        # Execute the workflow
        final_state = await app_graph.ainvoke(state_update, config=config)
        
        # 2. Update the log to 'success'
        final_answer = final_state.get("final_answer", "")
        await log_workflow_complete(log_id=log_id, final_answer=final_answer, status="success")
        
        return {
            "status": "success", 
            "data": {
                "plan": final_state.get("current_plan"),
                "answer": final_answer,
                "total_messages": len(final_state.get("messages", []))
            }
        }
        
    except Exception as e:
        # 3. If LangGraph crashes, update the log to 'failed'
        await log_workflow_complete(log_id=log_id, final_answer=str(e), status="failed")
        raise HTTPException(status_code=500, detail=str(e))