from langgraph.graph import StateGraph, END, START
from db import QdrantDB
from graph_state import GraphState,GraphState1,GraphState2,GraphState3
from nodes import  analyze_post, load_posts ,storeInDB,refine_analysis,create_embedding,categorize_posts,store_in_MONGO,load_posts1,retrieve_posts,retrieve_post_from_mongo,generate_response # Import node functions
from langchain_core.runnables.graph import (
    CurveStyle,
    Edge,
    MermaidDrawMethod,
    Node,
    NodeStyles,
)
# Initialize QdrantDB
db_handler = QdrantDB()
if not db_handler.is_running():
    print("Qdrant is not running.")

# Create the workflow
workflow = StateGraph(GraphState)


workflow.add_node("post_analysis", analyze_post)
workflow.add_node("storeInDB", storeInDB)
workflow.add_node("posts_loader", load_posts)
workflow.add_node("refine_analysis", refine_analysis)
workflow.add_node("embed_Description",create_embedding)
# Define edges (workflow execution path)
workflow.add_edge(START, "posts_loader")

workflow.add_edge("posts_loader", "post_analysis")
workflow.add_edge("post_analysis", "refine_analysis")
workflow.add_edge("refine_analysis", "embed_Description")
workflow.add_edge("embed_Description", "storeInDB")
workflow.add_edge("storeInDB", END)


graph_app = workflow.compile()



try:
    image_data = graph_app.get_graph().draw_mermaid_png()
    with open("post_processing.png", "wb") as f:
        f.write(image_data)
except Exception as e:
        print(f"Could not generate graph diagram: {str(e)}")




# Create the workflow
workflow1 = StateGraph(GraphState1)


workflow1.add_node("posts_loader", load_posts1)
workflow1.add_node("categorize_posts", categorize_posts)
workflow1.add_node("store_in_MONGO",store_in_MONGO)
# Define edges (workflow execution path)
workflow1.add_edge(START, "posts_loader")

workflow1.add_edge("posts_loader", "categorize_posts")
workflow1.add_edge("categorize_posts", "store_in_MONGO")
workflow1.add_edge("store_in_MONGO", END)


graph_app1 = workflow1.compile()



try:
    image_data = graph_app1.get_graph().draw_mermaid_png()
    with open("categorize_posts.png", "wb") as f:
        f.write(image_data)
except Exception as e:
        print(f"Could not generate graph diagram: {str(e)}")






# Create the workflow
workflow2 = StateGraph(GraphState2)
workflow2.add_node("retrieve_posts", retrieve_posts)
workflow2.add_edge("retrieve_posts", END)
workflow2.add_edge(START, "retrieve_posts")


graph_app2 = workflow2.compile()



try:
    image_data = graph_app2.get_graph().draw_mermaid_png()
    with open("retrieve_posts.png", "wb") as f:
        f.write(image_data)
except Exception as e:
        print(f"Could not generate graph diagram: {str(e)}")



# Create the workflow
workflow3 = StateGraph(GraphState3)
workflow3.add_node("retrieve_post", retrieve_post_from_mongo)
workflow3.add_node("generate_response", generate_response)
workflow3.add_edge(START, "retrieve_post")
workflow3.add_edge("retrieve_post", "generate_response")
workflow3.add_edge("generate_response", END)


graph_app3 = workflow3.compile()



try:
    image_data = graph_app3.get_graph().draw_mermaid_png()
    with open("chat_with_post.png", "wb") as f:
        f.write(image_data)
except Exception as e:
        print(f"Could not generate graph diagram: {str(e)}")