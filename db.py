from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
import socket
import uuid
from graph_state import GraphState


class QdrantDB():

    def __init__(self):
        self.client = self.create_connection()

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

        if not post_ids:
            print("No post IDs. Skipping search.")
            return []

        # Perform vector search filtered by the allowed post_ids
        try:
            results_from_db = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                with_payload=True,
                limit=topk,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.post_id",
                            match=models.MatchAny(any=post_ids)
                        )
                    ]
                )
            )

            matched_post_ids = []
            for point in results_from_db:
                matched_post_ids.append(point.payload["metadata"]["post_id"])
                post_id = point.payload["metadata"]["post_id"]
                score = point.score
                print(f"Post ID: {post_id}, Score: {score}")

            return matched_post_ids

        except Exception as e:
            print(f"Error during search: {str(e)}")
            # Fallback to brute-force search if index is missing
            print("Attempting fallback search without filter...")
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                with_payload=True,
                limit=topk * 3  # Get more results to filter locally
            )
            
            matched_post_ids = []
            for point in results:
                if point.payload["metadata"]["post_id"] in post_ids:
                    matched_post_ids.append(point.payload["metadata"]["post_id"])
                    if len(matched_post_ids) >= topk:
                        break
            
            return matched_post_ids[:topk]

    def create_connection(self):
        return QdrantClient(
            url="https://7275c8a8-926f-4091-b72c-d4c0a934fae4.us-west-2-0.aws.cloud.qdrant.io:6333", 
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uJ58UMUyVnJl0sOU3_ft0QsfHLkPQMEnpketnd_D_js",
            timeout=60
        )
    
    def is_running(self):
        try:
            host = "7275c8a8-926f-4091-b72c-d4c0a934fae4.us-west-2-0.aws.cloud.qdrant.io"
            port = 6333
            with socket.create_connection((host, port), timeout=60):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False
        
    def check_collection(self, state: GraphState):
        collections = self.client.get_collections()
        print("Available collections:", [col.name for col in collections.collections])
        collection_name = state["collection_name"]
        embedding_dim = state["embedding_dimension"]
        
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
            
        # Ensure index exists for post_id filtering
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.post_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            print("Created index for metadata.post_id")
        except Exception as e:
            print(f"Index may already exist: {str(e)}")

    def add_data(self, state: GraphState):
        print("🔹 Adding posts to Qdrant database...")

        collection_name = state["collection_name"]
        posts = state["posts"]

        try:
            payloads = []
            embeddings = []
            ids = []

            for post_id, post_data in posts.items():
                refined_description = post_data.get("Refined Description", "").strip()
                embedding = post_data.get("Embedding", None)

                if not refined_description or embedding is None:
                    print(f"⚠️ Skipping Post ID {post_id}")
                    continue

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
                ids.append(str(uuid.uuid4()))

            if not payloads:
                print("⚠️ No valid posts to insert into Qdrant.")
                return state

            print(f"✅ Inserting {len(payloads)} posts into Qdrant...")
            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads
                )
            )
            state["added_in_Qdrant"] = True
            print(f"✅ Successfully inserted {len(payloads)} posts into Qdrant!")
            return state

        except Exception as e:
            print(f"❌ Error inserting into Qdrant: {str(e)}")
            raise

    def remove_data(self, state: GraphState):
        print("🔹 Inside remove_data function...")
        posts_dict = state["posts"]
        if not posts_dict:
            print("⚠️ No posts found in state.")
            return

        post_ids = list(posts_dict.keys())
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
                print(f"🗑️ Removed Post ID {post_id} from Qdrant.")
            except Exception as e:
                print(f"❌ Error removing Post ID {post_id} from Qdrant: {str(e)}")