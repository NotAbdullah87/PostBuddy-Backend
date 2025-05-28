from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import time
import json
from workflow import graph_app,graph_app2,graph_app1,graph_app3
from db import QdrantDB
from graph_state import GraphState, GraphState1,GraphState2,GraphState3
from pydantic import BaseModel

from typing import TypedDict, Annotated, Tuple, Dict,Any
import uuid
import asyncio
class RetrievePostRequest(BaseModel):
    user_id: str
    user_query: str


class ChatWithPostRequest(BaseModel):
    user_id: str
    post_id: str
    
    user_query: str
   
 

app = FastAPI()


# Function to run GraphState workflow
async def  run_post_analysis_workflow():
    print("🔄 Running run_post_analysis_workflow ..")

    initial_state: GraphState = {
        "posts": {},
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "db_handler": "",
        "collection_name": "my_collection",
        "added_in_Qdrant": False
    }

    try:
        result = await graph_app.ainvoke(initial_state)

        # ✅ Save the state excluding db_handler
        json_state = {key: value for key, value in initial_state.items() if key != "db_handler"}

        with open("graph_state.json", "w", encoding="utf-8") as f:
            json.dump(json_state, f, indent=4, ensure_ascii=False)

        print("✅ run_post_analysis_workflow executed and results saved.")
    except Exception as e:
        print(f"❌ Error during post analysis workflow: {e}")
        raise

# ✅ Function to run GraphState1 workflow
async def run_auto_categorization_workflow():
    print("🔄 Running run_auto_categorization_workflow ...")

    initial_state1: GraphState1 = {
        "posts": {},
        "model_name": "llama-3.3-70b-versatile"
    }

    try:
        result = await graph_app1.ainvoke(initial_state1)

        # ✅ Save the state
        with open("graph_state1.json", "w", encoding="utf-8") as f:
            json.dump(initial_state1, f, indent=4, ensure_ascii=False)

        print("✅ run_auto_categorization_workflow executed and results saved.")
    except Exception as e:
        print(f"❌ Error during auto categorization workflow: {e}")
        raise

def run_post_analysis_sync():
    asyncio.run(run_post_analysis_workflow())  # ✅ Safe to use here

def run_auto_categorization_sync():
    asyncio.run(run_auto_categorization_workflow())  # ✅ Safe to use here
 

# ✅ Create Background Scheduler
scheduler = BackgroundScheduler()

# # ✅ Schedule both workflows to run at intervals
# scheduler.add_job(run_post_analysis_workflow, "interval", minutes=1)  
# scheduler.add_job(run_auto_categorization_workflow, "interval", minutes=2)  


scheduler.add_job(run_post_analysis_sync, "interval", minutes=1)
scheduler.add_job(run_auto_categorization_sync, "interval", minutes=2)


# ✅ FastAPI Startup Event (Starts Scheduler)
@app.on_event("startup")
def start_scheduler():
    print("✅ Starting scheduler...")
    scheduler.start()

# ✅ FastAPI Shutdown Event (Stops Scheduler Gracefully)
@app.on_event("shutdown")
def shutdown_scheduler():
    print("❌ Stopping scheduler...")
    scheduler.shutdown()

# ✅ API Endpoints (Optional)
@app.get("/")
def home():
    return {"message": "FastAPI server with APScheduler is running!"}

@app.post("/run-post-analysis")
async def run_post_analysis():
    await run_post_analysis_workflow()
    return {"status": "success", "message": "Post analysis workflow executed"}

@app.post("/run-auto-categorization")
async def run_auto_categorization():
    await run_auto_categorization_workflow()
    return {"status": "success", "message": "Auto categorization workflow executed"}



@app.post("/retrieve-posts")
async def retrieve_posts_endpoint(payload: RetrievePostRequest):
    print("📥 Received request for retrieve-posts")

    initial_state: GraphState2 = {
        "posts": {},
        "post_ids": [],
        "topk_ids": [],
        "results": [],
        "topk": 5,  # or make this dynamic from request
        "user_id": payload.user_id,
        "user_query": payload.user_query,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "db_handler": "",
        "collection_name": "my_collection"
    }

    final_state = await graph_app2.ainvoke(initial_state)

    # Optional: Save the state for debugging
    with open("graph_state2.json", "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=4, ensure_ascii=False, default=str)

    if not final_state["results"]:
        print("❌ No matching posts found.")
        return {
            "status": "empty",
            "message": "No matching posts found for the given query."
        }
    else:
        return {
            "status": "success",
            "matched_posts": final_state["results"]
        }


@app.post("/chat-with-post")
async def chat_with_post(payload: ChatWithPostRequest):
    user_id = payload.user_id
    post_id = payload.post_id
    user_query = payload.user_query
    print("📥 Received request for chat-with-post")
   


    initial_state: GraphState3 = {
        "post_id": post_id,
        "user_id": user_id,
        "user_query": user_query,
        "system_response": "",
        "conversation_history": [],
        "post_description": "",
        "chat_id": ""
    }
    
    try:
        final_state = await graph_app3.ainvoke(initial_state)

        # Optional: Save the state for debugging
        with open("graph_state3.json", "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=4, ensure_ascii=False, default=str)

        response = final_state["system_response"]
        chat_id = final_state["chat_id"]
        return {"status": "success", "chat_id": chat_id, "system_response": response}
    
    except Exception as e:
        print(f"❌ Error during chat_with_post: {e}")
        return {"status": "error", "message": str(e)}
