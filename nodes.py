import os
from bson import ObjectId  # Import ObjectId

import requests
import time
import google.generativeai as genai
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Tuple, Dict,Any
from enum import Enum
# Load environment variables (if using .env)
load_dotenv()
from graph_state import GraphState,GraphState1,GraphState2,GraphState3  # Import GraphState
import tempfile
from utils import remove_formatting,get_timestamp,latest_value_with_timestamp,get_qdrant,download_image,extract_and_validate_category
from PIL import Image
import os
import logging
from pymongo import MongoClient, errors
from dotenv import load_dotenv
import json
from enum import Enum
# Load environment variables (if using .env)
import ollama
from groq import Groq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from utils import generate_chat_id,store_conversation,get_conversation,store_system_message
load_dotenv()
# Get MongoDB URI from environment variable
uri = os.getenv("MONGO_URI")
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)



if not uri:
    print("MONGO_URI is not set. Please configure it in environment variables.")
    raise ValueError("MONGO_URI is not set. Please configure it in your environment variables.")

try:
    # Connect to MongoDB
    client = MongoClient(uri, serverSelectionTimeoutMS=50000)
    client.admin.command("ping")  
    print("Connected to MongoDB successfully.")

    # Select database and collection
    db = client["postbuddy"]
    collection = db["posts"]
    collection1=db["user"]  # for searching within user collection


except errors.ServerSelectionTimeoutError as e:
    logging.error("Failed to connect to MongoDB: Server selection timeout.", exc_info=True)
    raise ConnectionError("Failed to connect to MongoDB.") from e
except errors.ConnectionFailure as e:
    logging.error("MongoDB connection failed.", exc_info=True)
    raise ConnectionError("MongoDB connection failed.") from e
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    raise RuntimeError(f"An unexpected error occurred: {e}") from e




def analyze_post(state: GraphState) -> GraphState:
    """Traverses posts, downloads images, analyzes them using Gemini, and updates state."""
    print("Analyzing posts one by one...")

    # Load API Key
    KEY = os.getenv("GEMINI_API_KEY")
    if not KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please check your environment variables.")

    genai.configure(api_key=KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Create a temporary directory for storing images
    with tempfile.TemporaryDirectory() as temp_dir:
        for post_id, post_data in state["posts"].items():
            image_url = post_data.get("image_url")
            if not image_url:
                print(f"No image URL for post {post_id}, marking as 'No image provided'.")
                state["posts"][post_id]["image_analysis"] = ""
                continue

            # Download the image
            image_path = os.path.join(temp_dir, f"{post_id}.jpg")
            if not download_image(image_url, image_path):
                print(f"Failed to download image for post {post_id}, marking as 'Image download failed'.")
                state["posts"][post_id]["image_analysis"] = "" # storing nothing
                continue

            try:
                # Open the image inside a context manager to ensure it closes properly
                with Image.open(image_path) as image:
                    response = model.generate_content(["Describe this image:", image])
                    analysis_result = response.text if response else "" # storing space in the field if something goes wrong

                # Store the analysis result
                state["posts"][post_id]["image_analysis"] = analysis_result
                # print(f"Analysis for post {post_id}: {analysis_result}")

            except Exception as e:
                print(f"Error processing image for post {post_id}: {e}")
                state["posts"][post_id]["image_analysis"] = f"Error: {str(e)}"

    return state


def refine_analysis(state: GraphState) -> GraphState:
    
    
    posts=state["posts"]
    for post_id, post_data in posts.items():
        post_title = post_data.get("post_title", "No title available")
        post_description = post_data.get("description", "No description available")
        image_analysis = post_data.get("image_analysis", "No image analysis available")


     
        system_prompt = """You are a specialized post summarization assistant. Your task is to refine post descriptions to make them optimally retrievable through natural language queries. Focus on extracting and highlighting key information, entities, topics, and emotions present in the post. Format the refined description in a clear, searchable manner while preserving all important details."""

        # Handle missing image analysis
        image_analysis_text = image_analysis.strip() if image_analysis and image_analysis.strip() else "No image is present in this post."

        # Unified description
        UNIFIED_DESCRIPTION = f"""
        Post Title: {post_title}
        Post Description: {post_description}
        Image Analysis: {image_analysis_text}
        """

        # User prompt for refinement
        user_prompt = f""" Below is a unified description of a saved post containing the post title, description, and image content (if present). Refine this description to optimize it for retrieval when users search using natural language queries.

        Original Unified Description:
        {UNIFIED_DESCRIPTION}

        Please create a refined version that:
        1. Highlights key topics, entities, and concepts
        2. Preserves important details and context
        3. Structures information in a way that's optimized for semantic search
        4. Maintains the post's original intent and meaning"""

        # Generate refined description from Groq API
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=1500
        )

        final_description = response.choices[0].message.content.strip()  # Extract response content
        final_description=remove_formatting(final_description)
     
        post_data["Refined Description"] = final_description


    state["posts"] = posts

    return state


def create_embedding(state: GraphState) -> GraphState:
    print("🔹 Inside the embedding creation function...")

    # return state # for the time being
    model = state["embedding_model"]  # Get model name from state
    embed_model = FastEmbedEmbeddings(model_name=model)  # Initialize embedding model

    posts = state["posts"]  # Access posts dictionary


    for post_id, post_data in posts.items():
        refined_description = post_data.get("Refined Description", "").strip()

        if refined_description:
            # Generate embedding
            embedding = embed_model.embed_query(refined_description)

            # Store embedding in dictionary
            post_data["Embedding"] = embedding 

            print(f"✅ Created embedding for Post ID: {post_id}")

        else:
            print(f"⚠️ Skipping Post ID {post_id} (No Refined Description found)")

    # ✅ Update state with embeddings
    state["posts"] = posts  


    print(" All embeddings stored successfully!")
    return state

# Node to retrieve relevant chunks from QdrantDB
def storeInDB(state: GraphState) -> GraphState: # store the payload in qdrant and update the flag and desription in the mongo db
   
    
    print("in store in db function")

    state["db_handler"]=get_qdrant()
    db_handler=state["db_handler"]

    if not db_handler.is_running():
        # return jsonify({"error": f"{vectorDB} service unavailable"}), 500
        return state
        
    if db_handler:
        try:
            db_handler.check_collection(state)
            print(f"the type of db handler is {type(db_handler)}")

            db_handler.remove_data(state)
            db_handler.add_data(state)


            print("Added in qdrant ")

            # time.sleep(5)

            updated_ids = []  # Track successful updates

            if state["added_in_Qdrant"]:
                posts_dict = state["posts"]

                for post_id, post_data in posts_dict.items():
                    object_id = ObjectId(post_id)
                    refined_description = post_data.get("Refined Description", "No description available")

                    result = collection.update_one(
                        {"_id": object_id},
                        {"$set": {"processed": True, "refinedDescription": refined_description}}
                    )

                    if result.matched_count > 0:
                        updated_ids.append(object_id)  # Store successfully updated documents
                    else:
                        raise Exception(f"No document found for Post ID {post_id}")  # Trigger rollback

            print("✅ All updates applied successfully.")

        except Exception as e:
            # Rollback updates by setting `processed` back to False
            if updated_ids:
                collection.update_many(
                    {"_id": {"$in": updated_ids}}, {"$set": {"processed": False, "refinedDescription": ""}}
                )
                print("♻️ Rolled back updates due to error.")

            print(f"❌ Error occurred: {str(e)}")

        return state



    return state  # Return updated GraphState
# Node to load document
def load_posts(state: GraphState) -> GraphState:


    print("Retrieving posts from MongoDB... in load_posts function")

    documents = collection.find({"processed": False})  # Fetch unprocessed posts

    # Dictionary to store posts
    posts_dict: Dict[str, Dict[str, Any]] = {}

    # documents=documents[0:4]
    # Loop through documents and extract fields
    for doc in documents:
        
        post_id = str(doc["_id"])  # Convert ObjectId to string
        post_title = doc["postTitle"]
        description = doc["description"]
        post_url = doc["postUrl"]
        image_url = doc.get("imageUrl", "No Image")
        platform = doc["platform"]
        category = doc["category"]
        processed=doc["processed"]
        refinedDescription=doc["refinedDescription"]

        # Store post information in dictionary
        posts_dict[post_id] = {
            "post_title": post_title,
            "description": description,
            "post_url": post_url,
            "image_url": image_url,
            "platform": platform,
            "category": category,
            "processed": processed,
            "Refined Description": refinedDescription
        }

        # Print extracted values
        print(f"Post ID: {post_id}")
        print(f"Title: {post_title}")


    output_file = "posts.json"

    

    print(f"Results saved to {output_file}")
    # Update GraphState with posts
    state["posts"] = posts_dict
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(state["posts"], f, indent=4, ensure_ascii=False)
 
    return state


#functions for the categorization langgraph



def load_posts1(state: GraphState1) -> GraphState1:


    print("Retrieving posts from MongoDB... in load_posts function")

    documents = collection.find({"processed": True, "category": ""})  # Fetch processed posts

    # Dictionary to store posts
    posts_dict: Dict[str, Dict[str, Any]] = {}

    # documents=documents[12:25]
    # Loop through documents and extract fields
    for doc in documents:
        
        post_id = str(doc["_id"])  # Convert ObjectId to string
        post_title = doc["postTitle"]
        description = doc["description"]
        post_url = doc["postUrl"]
        image_url = doc.get("imageUrl", "No Image")
        platform = doc["platform"]
        category = doc["category"]
        processed=doc["processed"]
        refinedDescription=doc["refinedDescription"]

        # Store post information in dictionary
        posts_dict[post_id] = {
            "post_title": post_title,
            "description": description,
            "post_url": post_url,
            "image_url": image_url,
            "platform": platform,
            "category": category,
            "processed": processed,
            "Refined Description": refinedDescription
        }

        # Print extracted values
        print(f"Post ID: {post_id}")
        print(f"Title: {post_title}")


    output_file = "posts.json"

    

    print(f"Results saved to {output_file}")
    # Update GraphState with posts
    state["posts"] = posts_dict
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(state["posts"], f, indent=4, ensure_ascii=False)
 
    return state


def categorize_posts(state: GraphState1):
    print("🔍 Categorizing posts...")

    categories_list = [
        "Personal Updates", "Educational & Learning", "Business & Work", 
        "Entertainment & Pop Culture", "News & Current Affairs", "Promotions & Marketing", 
        "Lifestyle & Wellness", "Motivational & Inspirational", "Creativity & Arts", 
        "Humor & Fun", "Technology & Science", "Finance & Investing", "Social & Activism"
    ]

    system_part = (
            "You are an expert in content categorization. Your role is to analyze the Refined Description of post, determine its primary theme, and classify it based on a provided list of categories. Use the content's tone, focus, and keywords to make an informed decision."
        )  

    posts_dict = state["posts"]
    if not posts_dict:
        print("⚠️ No posts found in state.")
        return state

    

    for post_id, post_data in posts_dict.items():
        # ✅ Check if post is processed but doesn't have a category
        if post_data.get("processed") and not post_data.get("category"):
            refined_description = post_data.get("Refined Description", "No description available")

            user_query = f"""
            Below is the refined description of a post. Your task is to classify it into one of the following categories:

            Categories: {", ".join(categories_list)}

            Post Description: {refined_description}

            Please return only the category name as the output.
            """

            # Call LLM to get the category
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_query}
                ],
                model=state["model_name"],
                max_tokens=50
            )

            category = response.choices[0].message.content.strip()

            category=extract_and_validate_category(category, categories_list) # writing just in case if we do not get only category as response from the llm

            if category is not None:
                # ✅ Update the category in the posts dictionary
                posts_dict[post_id]["category"] = category

            
            else:
                print(f"⚠️ Skipping Post ID {post_id} due to invalid category response.")

    # ✅ Return updated state
    state["posts"] = posts_dict

    output_file = "categorize_posts.json"

    # ✅ Load existing data if the file already exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Existing JSON file is corrupted. Starting fresh.")
                existing_data = {}  # Start with an empty dictionary
    else:
        existing_data = {}  # If file doesn't exist, start fresh

    # ✅ Merge existing data with new categorized posts
    existing_data.update(state["posts"])

    # ✅ Save merged data back to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Results appended to {output_file}")
    return state


def store_in_MONGO(state: GraphState1):
    
    print("📌 In store_in_MONGO function...")


    posts_dict = state["posts"]
    if not posts_dict:
        print("⚠️ No posts found in state.")
        return state

    for post_id, post_data in posts_dict.items():
        if post_data.get("processed") and "category" in post_data:
            category = post_data["category"]

            try:
                object_id = ObjectId(post_id)
                result = collection.update_one(
                    {"_id": object_id},
                    {"$set": {"category": category}}
                )

                if result.matched_count > 0:
                    print(f"✅ Categorized Post ID {post_id} as {category} and updated in MongoDB.")
                else:
                    print(f"⚠️ No document found for Post ID {post_id} in MongoDB.")

            except Exception as e:
                print(f"❌ Error updating MongoDB for Post ID {post_id}: {e}")

    return state  # ✅ Return updated state






def retrieve_posts(state:GraphState2):
    print("in retrieve_posts function")

    state["db_handler"]=get_qdrant()
    db_handler=state["db_handler"]

    if not db_handler.is_running():
        # return jsonify({"error": f"{vectorDB} service unavailable"}), 500
        return state
        
    if db_handler:
        try:
            db_handler.check_collection(state)
            # user_id = ObjectId("67c75778bf5515f0b0ccddc2")

            
            user_id = ObjectId(state["user_id"])

            user_doc = collection1.find_one(
                {"_id": user_id},
                {"posts": 1, "_id": 0}
            )

            if user_doc and "posts" in user_doc:

                state["post_ids"]=user_doc["posts"]
                # print(state["post_ids"])
            else:
                print("No posts found or user does not exist.")
                
                return state

                

            
            collection_name=state["collection_name"]

            allowed_post_ids = state["post_ids"]  # List of post IDs from MongoDB

            # Ensure all post_ids are strings
            post_ids = [str(pid) for pid in allowed_post_ids]

            print("No of posts present against the user are ",len(post_ids))
            model=state["embedding_model"]
            embed_model = FastEmbedEmbeddings(model_name=model)  # Initialize embedding model
            embedding = embed_model.embed_query(state["user_query"])

            # Store embedding in dictionary
            query_vector = embedding 
            topk = state["topk"]
            collection_name=state["collection_name"]
            matched_ids = db_handler.search(
                collection_name=collection_name,
                query_vector=query_vector,
                topk=topk,
                post_ids=post_ids
            )


            state["topk_ids"]=matched_ids

            matched_posts = list(collection.find( ## find post objects in mongo against matched post ids
            {"_id": {"$in": [ObjectId(post_id) for post_id in matched_ids]}}
        ))

            for post in matched_posts:
                post["_id"] = str(post["_id"])  # returning the id as string in post objects

            state["results"] = matched_posts

            print("No of matched posts are ",len(matched_posts))





        except Exception as e:
            print("error occured in retrive posts function")

        
        
        return state

        


def retrieve_post_from_mongo(state: GraphState3):
    print("in retrieve_post_from_mongo function")

    post_id = state["post_id"]
    user_id = state["user_id"]

    # Convert string post_id to ObjectId
    object_id = ObjectId(post_id)

    # Retrieve the post from MongoDB
    post = collection.find_one({"_id": object_id})

    if post:
        # Convert ObjectId to string for JSON serialization
        post["_id"] = str(post["_id"])
        state["post"] = post

        # Store refinedDescription if available
        refined_description = post.get("refinedDescription")
        if refined_description:
            state["post_description"] = refined_description
            print("Refined description stored in state.")
        else:
            print("Refined description not found in post.")

        # print(f"Post retrieved successfully: {post}")
    else:
        print(f"No post found with ID: {post_id}")

    return state


def generate_response(state: GraphState3):
    print("in generate_response function")

    post_id = state["post_id"]
    user_id = state["user_id"]  
    user_query = state["user_query"]
    description = state["post_description"]

    # Generate or reuse chat_id
    chat_id = generate_chat_id(user_id, post_id)
    state["chat_id"] = chat_id

    # System prompt
    system_prompt = f"""
    You are a specialized assistant for a social media platform. Your task is to provide detailed and informative responses based on the user's query and the post's refined description. Use the refined description to generate a comprehensive answer that addresses the user's question while maintaining the context of the post.
    
    Refined Description for Post:
    {description}
    """

    # store_system_message(chat_id, system_message=system_prompt)

    # Get conversation history from Redis
    raw_history = get_conversation(chat_id)
    
    # ✅ Convert stored messages into OpenAI-style messages
    messages = [{"role": "system", "content": system_prompt}]
    for item in raw_history:
        # Safely handle both JSON strings and already-loaded dicts
        if isinstance(item, str):
            parsed = json.loads(item)
        else:
            parsed = item  # Already a dict

        if "user" in parsed and "assistant" in parsed:
            messages.append({"role": "user", "content": parsed["user"]})
            messages.append({"role": "assistant", "content": parsed["assistant"]})

    # Add current user message
    messages.append({"role": "user", "content": user_query})

    # Debug print
    print(f"Formatted messages: {messages}")

    # Send to LLM
    response = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        max_tokens=1500
    )

    answer = response.choices[0].message.content
    state["system_response"] = answer

    # Store this turn in Redis
    store_conversation(chat_id, user_query, answer)

    return state
