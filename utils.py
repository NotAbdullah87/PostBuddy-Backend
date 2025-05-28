import time
import re
import redis
import json
import requests
import time


# Initialize Redis client (assuming Redis is running locally)
r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

def generate_chat_id(user_id: str, post_id: str) -> str:
    """Generate a unique chat_id based on user_id and post_id"""
    return f"{user_id}:{post_id}"


def store_conversation(chat_id: str, user_query: str, system_response: str):


    # Store the user and assistant message as usual
    conversation_entry = {
        "user": user_query,
        "assistant": system_response
    }
    r.rpush(chat_id, json.dumps(conversation_entry))


def store_system_message(chat_id: str, system_message: str):
    existing_length = r.llen(chat_id)

    if existing_length == 0:
        # First message: store system message + user query + assistant response
        initial_system_entry = {
            "system": system_message
        }
        r.rpush(chat_id, json.dumps(initial_system_entry))


def get_conversation(chat_id: str) -> list:
    """Retrieve the conversation history from Redis"""
    conversation_data = r.lrange(chat_id, 0, -1)  # Get all items in the list
    return [json.loads(entry) for entry in conversation_data]  # Convert JSON strings to dictionaries

def process_user_query(user_query: str, chat_id: str) -> str:
    """Process the user query and generate a response (simulated)"""
    # Here, you can integrate your AI model for generating a response
    # For demonstration, we're returning a fixed response
    response = f"Response to: {user_query}"

    # Store the query and response in the conversation history
    store_conversation(chat_id, user_query, response)

    return response


def remove_formatting(text: str) -> str:
    """Removes markdown-style bold, headers, and unnecessary prefixes like 'Refined Description'."""

    # Remove markdown headers (###, ##, #) with 'Refined Description' variations
    text = re.sub(r"^#+\s*(Refined (Unified )?Description|Refined Post Description)[:\-]?\s*", "", text, flags=re.IGNORECASE)

  
    # Remove variations of 'Refined Description
    text = re.sub(r"(Refined (Unified )?Description|Refined Post Description)[:\-]?\s*", "", text, flags=re.IGNORECASE)

    # Remove markdown-style **bold**
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    return text.strip()  # Remove leading/trailing spaces


def download_image(url: str, save_path: str) -> str:
    """Downloads an image from a URL and saves it to the specified path."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return save_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None
def latest_value_with_timestamp(a, b):
    """Returns the value with the latest timestamp."""
    
    if not isinstance(a, tuple) or len(a) != 2:
        a = (None, 0)  # Default value with an old timestamp
    if not isinstance(b, tuple) or len(b) != 2:
        b = (None, 0)

    # Ensure we only keep meaningful values with the latest timestamp
    if b[1] > a[1]:
        if b[0] is not None and (not isinstance(b[0], (str, list)) or len(b[0]) > 0):
            return b
    return a

def get_timestamp():
    """Returns the current timestamp in microseconds."""
    return int(time.time() * 1_000_000)

def get_qdrant():
    from db import QdrantDB  # Lazy import (only runs when function is called) due to circular import issue btw nodes.ps and db.py
    return QdrantDB()
import ollama
def generate_ans(user_prompt, system_prompt, model_name):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
 
    response = ollama.chat(
        model=model_name, 
        messages=messages, 
        options={"temperature": 0.7, "num_ctx": 7000}
    )
    return response['message']['content']
   


def extract_and_validate_category(llm_response, categories_list):
    """
    Extracts a category from the LLM response and ensures it's valid.
    
    :param llm_response: The raw response from the LLM.
    :param categories_list: List of predefined categories.
    :return: Valid category or None if extraction fails.
    """
    # Normalize categories for case-insensitive matching
    categories_lower = {category.lower(): category for category in categories_list}

    # Extract category using regex
    match = re.search(r"\b(" + "|".join(re.escape(category) for category in categories_list) + r")\b", llm_response, re.IGNORECASE)

    if match:
        extracted_category = match.group(1).strip()

        # ✅ Check if the extracted category is in our predefined list
        if extracted_category.lower() in categories_lower:
            return categories_lower[extracted_category.lower()]  # Return properly formatted category
    
    print(f"❌ No valid category found in LLM response: {llm_response}")
    return None  # Return None if no valid category is found

import uuid
