from typing import TypedDict, Annotated, Tuple, Dict,Any,List
from utils import latest_value_with_timestamp, get_timestamp
# Define GraphState globally
class GraphState(TypedDict): # for processing of posts
    posts: Dict[str, Dict[str, Any]]
    embedding_model:str
    embedding_dimension:int
    db_handler:str
    collection_name:str
    added_in_Qdrant:bool



class GraphState1(TypedDict): # for categorization of posts
    posts: Dict[str, Dict[str, Any]]
    model_name:str
   

class GraphState2(TypedDict): # for retreival of posts
    posts: Dict[str, Dict[str, Any]]
    post_ids:list
    topk_ids:list
    results:list[Dict[str, Any]] #json like objects for post
    topk:int
    user_id:str # from api
    user_query:str  # from api
    embedding_model:str
    embedding_dimension:int
    db_handler:str
    collection_name:str
    


class GraphState3(TypedDict): # for chatting with posts
    post_id:str
    user_id:str # from api
    user_query:str  # from api
    system_response:str
    conversation_history: list[Dict[str, Any]]
    post_description:str
    chat_id:str

    

