from workflow import graph_app,graph_app1
from db import QdrantDB
from graph_state import GraphState,GraphState1
from utils import latest_value_with_timestamp, get_timestamp
# Initialize QdrantDB
import json
db_handler = QdrantDB()

# # Initial state for the workflow
# initial_state: GraphState = {
#     "posts": {},
#     "embedding_model":"BAAI/bge-small-en-v1.5",
#     "embedding_dimension":384,
#     "db_handler":"",
#     "collection_name": "my_collection",
#     "added_in_Qdrant":False

#     # "question": ("", 0),
#     # "db_handler": db_handler  # Pass QdrantDB instance
# }


# # Run the workflow
# result = graph_app.invoke(initial_state)

# # Print the result
# # print(f"We are back in the main.py file {result}")

# # Define the output file path
# output_file = "output_langgraph.json"

# json_state = {key: value for key, value in initial_state.items() if key != "db_handler"}

# # Write to JSON file
# with open("graph_state.json", "w", encoding="utf-8") as f:
#     json.dump(json_state, f, indent=4, ensure_ascii=False)




# print(f"Results saved to {output_file}")



# Initial state for the workflow
initial_state1: GraphState1 = {
    "posts": {},
    "model_name":"llama-3.3-70b-versatile"
    
}


# Run the workflow
result = graph_app1.invoke(initial_state1)





