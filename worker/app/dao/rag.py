import os

from fastapi import Depends

from app.utils.secrets import get_open_ai_client, get_pinecone_client


class RagDAO:
    def __init__(self, openai_client, pinecone_client):
        self.__pinecone_client = pinecone_client
        self.__openai_client = openai_client
        self.__chattest_index = self.__pinecone_client.Index(f"chat-{os.getenv('ENV')}")

    def upsert_vectors(self, vectors: list):
        self.__chattest_index.upsert(vectors=vectors)

    def query_data(self):
        pass

    def get_embedding(self, text):
        response = self.__openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding

    def upsert_embedding(self, vectors):
        self.__chattest_index.upsert(vectors=vectors)

    def query_index_with_filter(self, query_text, user_id, top_k=2, score_threshold=0.3) -> list[dict]:
        # Get the embedding for the query text
        query_embedding = self.get_embedding(query_text)

        # Query the index with user_id filter
        results = self.__chattest_index.query(
            vector=query_embedding,
            filter={"user_id": str(user_id)},  # Add filter for user_id
            top_k=top_k,
            include_metadata=True
        )

        # Apply score threshold filter
        filtered_results: list[dict] = [
            match for match in results['matches']
            if match['score'] >= score_threshold
        ]

        return filtered_results


def get_rag_dao(openai_client=Depends(get_open_ai_client),
                pinecone_client=Depends(get_pinecone_client())):
    return RagDAO(openai_client, pinecone_client)
