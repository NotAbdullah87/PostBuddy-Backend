from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
import socket
import uuid
from graph_state import GraphState


class QdrantDB():

    def __init__(self):
        self.client = self.create_connection()

    # def search(self,collection_name,query_vector,topk,user_id):

    #     print("inside the search function in qdrant Db")

    #     results_from_db = self.client.query_points(
    #         collection_name=collection_name,
    #         query=query_vector,
    #         with_payload=True,
    #         limit=topk,
    #         query_filter=models.Filter(
    #             must=[
    #                 models.FieldCondition(
    #                     key="metadata.user_id",
    #                     match=models.MatchValue(value=user_id),
    #                 )
    #             ]
    #         )
    #         )
    #     context=[]
    #     context.extend([point.payload['text'] for point in results_from_db.points])  
    #     print("inside the search function in qdrant Db, fitlering only on the basis of user id")
    #     print(f"the results returned after applying filtering is {results_from_db}")
    #     print(f" the context being returned now is {context}")
        
    #     return context


    def search(self, collection_name, query_vector, topk, post_ids):
        """
        Perform vector search in Qdrant restricted to given post_ids and return top-k post_ids.

        Args:
            collection_name (str): Qdrant collection name.
            query_vector (List[float]): The query embedding vector.
            topk (int): Number of top results to retrieve.
            post_ids (List[str]): List of allowed post IDs to restrict search to.

        Returns:
            List[str]: List of top-k matched post IDs.
        """
        print("Inside the search function in Qdrant DB")

        
        for i in post_ids:
            print(f"Post ID: {i}")
        # Perform vector search filtered by the allowed post_ids
        results_from_db = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            with_payload=True,
            limit=topk,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.post_id",  # Direct key, not nested under "metadata"
                        match=models.MatchAny(any=post_ids)
                    )
                ]
            )
        )

        matched_post_ids=[]
        # for point in results_from_db:
        #     matched_post_ids.append(point.payload["metadata"]["post_id"])
            # print(point.payload["metadata"]["post_id"])

        for point in results_from_db:
            matched_post_ids.append(point.payload["metadata"]["post_id"])
            post_id = point.payload["metadata"]["post_id"]
            score = point.score
            print(f"Post ID: {post_id}, Score: {score}")

        # print("Filtering based on post_ids")
        # print(f"Results returned after applying filter: {results_from_db}")
        # print(f"Post IDs being returned: {matched_post_ids}")



        return matched_post_ids



    def create_connection(self):
        return QdrantClient(
                url="https://7275c8a8-926f-4091-b72c-d4c0a934fae4.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uJ58UMUyVnJl0sOU3_ft0QsfHLkPQMEnpketnd_D_js",
                timeout=60  # Increase timeout (default is ~5s)
            )
    
    def is_running(self):
        # try:
        # # Try connecting to the Qdrant server
        #     with socket.create_connection(("localhost", 6333), timeout=60):
        #         return True
        # except (socket.timeout, ConnectionRefusedError):
        #     return False
        try:
            host = "7275c8a8-926f-4091-b72c-d4c0a934fae4.us-west-2-0.aws.cloud.qdrant.io"
            port = 6333
            with socket.create_connection((host, port), timeout=60):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False
        
    def check_collection(self,state: GraphState):

        collections = self.client.get_collections()
        print("Available collections:", [col.name for col in collections.collections])
        collection_name=state["collection_name"]
        embedding_dim=state["embedding_dimension"]
        if not self.client.collection_exists(collection_name):

       
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )


    def add_data(self, state: GraphState):
        print("🔹 Adding posts to Qdrant database...")

        collection_name = state["collection_name"]  # Get collection name
        posts = state["posts"]  # Extract posts dictionary

        try:
            payloads = []
            embeddings = []
            ids = []

            for post_id, post_data in posts.items():
                refined_description = post_data.get("Refined Description", "").strip()
                embedding = post_data.get("Embedding", None)

                if not refined_description:
                    print(f"⚠️ Skipping Post ID {post_id} (No Refined Description)")
                    continue

                if embedding is None:
                    print(f"⚠️ Skipping Post ID {post_id} (No Embedding)")
                    continue

                # Construct payload with metadata
                payload = {
                    "text": refined_description,
                    "metadata": {
                        "post_id": post_id,
                        "post_url": post_data.get("post_url", "Unknown URL"),
                        "platform": post_data.get("platform", "Unknown Platform"),
                    }
                }

                payloads.append(payload)
                embeddings.append(embedding)
                ids.append(str(uuid.uuid4()))  # Generate unique ID for each embedding

            if not payloads:
                print("⚠️ No valid posts to insert into Qdrant.")
                return state

            # Insert into Qdrant
            print(f"✅ Inserting {len(payloads)} posts into Qdrant...")
            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads
                )
            )
            state["added_in_Qdrant"] = True  # Mark as added in Qdrant      
            print(f"✅ Successfully inserted {len(payloads)} posts into Qdrant!")

            return state

        except Exception as e:
            print(f"❌ Error inserting into Qdrant: {str(e)}")
            raise  # Re-raise exception for debugging

    




    def remove_data(self, state: GraphState):
        print("🔹 Inside remove_data function...")

        # Extract posts from state
        posts_dict = state["posts"]
        if not posts_dict:
            print("⚠️ No posts found in state.")
            return

        # Extract post IDs from state
        post_ids = list(posts_dict.keys())  # These are string post IDs

        # ✅ Remove posts from Qdrant using post_id
        collection_name = state["collection_name"]

        for post_id in post_ids:
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.post_id",
                                match=models.MatchValue(value=post_id),
                            )
                        ]
                    ),
                )
                print(f"🗑️ Removed Post ID {post_id} from Qdrant.") # because it had processed flag as false in the mongo db 
            except Exception as e:
                print(f"❌ Error removing Post ID {post_id} from Qdrant: {str(e)}")


      





# search fun on the basis of list of posts ids present against a user in mongo, first filter the posts from qdrarnt and then the query on filtered posts

# from qdrant_client import models

# def search_with_post_ids(self, collection_name, query_vector, current_chunk_count, post_ids):
#     """
#     Search Qdrant but restrict results to a specific list of post_ids.
#     """
#     print("🔹 Running filtered search in Qdrant...")

#     try:
#         # Query Qdrant using post_id list as filter
#         results_from_db = self.client.query_points(
#             collection_name=collection_name,
#             query=query_vector,
#             with_payload=True,
#             limit=current_chunk_count,
#             query_filter=models.Filter(
#                 must=[
#                     models.FieldCondition(
#                         key="metadata.post_id",
#                         match=models.MatchAny(values=post_ids),  # Filter only specific post IDs
#                     )
#                 ]
#             )
#         )

#         if not results_from_db.points:
#             print("⚠️ No matching posts found for the given post IDs.")
#             return []

#         # Extract relevant details
#         retrieved_posts = []
#         for point in results_from_db.points:
#             payload = point.payload
#             retrieved_posts.append({
#                 "Post ID": payload.get("metadata", {}).get("post_id", "Unknown"),
#                 "Post Title": payload.get("metadata", {}).get("post_title", "No Title"),
#                 "Post URL": payload.get("metadata", {}).get("post_url", "No URL"),
#                 "Platform": payload.get("metadata", {}).get("platform", "Unknown"),
#                 "Refined Description": payload.get("text", "No Description Available")
#             })

#         print(f"✅ Retrieved {len(retrieved_posts)} posts matching given post IDs.")
#         return retrieved_posts

#     except Exception as e:
#         print(f"❌ Error in search function: {str(e)}")
#         return []
