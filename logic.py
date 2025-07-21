import hashlib
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from chat.models import PromptRequest, PromptResponse
from cache.redis_backend import RedisCache
from cache.config import CacheConfig

# Load environment variables
load_dotenv()

# OpenAI client (v1.0+)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cache = RedisCache()

USE_SEMANTIC_CACHE = getattr(CacheConfig, "USE_SEMANTIC_CACHE", False)
SIMILARITY_THRESHOLD = getattr(CacheConfig, "SIMILARITY_THRESHOLD", 0.9)


def generate_key(session_id: str, message: str) -> str:
    hash_value = hashlib.sha256(message.encode()).hexdigest()
    return f"chat:{session_id}:{hash_value}"


def embed_text(text: str) -> np.ndarray:
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm != 0 else 0.0


def bot_logic(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(error generating response: {e})"


def find_semantic_match(session_id: str, input_embedding: np.ndarray):
    matches = []
    for key in cache.list_keys():
        if key.startswith(f"chat:{session_id}:") and key.endswith(":vec"):
            vec_data = cache.get(key)
            response_key = key.replace(":vec", ":response")
            response_text = cache.get(response_key)
            if vec_data and response_text:
                vec = np.array(vec_data)
                score = cosine_similarity(input_embedding, vec)
                if score >= SIMILARITY_THRESHOLD:
                    matches.append((score, response_text))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]
    return None


def process_prompt(req: PromptRequest) -> PromptResponse:
    # Semantic match first (if enabled)
    if USE_SEMANTIC_CACHE:
        embedding = embed_text(req.message)
        if embedding is not None:
            match = find_semantic_match(req.session_id, embedding)
            if match:
                return PromptResponse(session_id=req.session_id, response=match, from_cache=True)

    # Fallback to exact match
    key = generate_key(req.session_id, req.message)
    cached = cache.get(key)
    if cached:
        return PromptResponse(session_id=req.session_id, response=cached, from_cache=True)

    # Generate fresh response
    response = bot_logic(req.message)
    cache.set(key, response)

    if USE_SEMANTIC_CACHE and embedding is not None:
        vec_key = f"{key}:vec"
        res_key = f"{key}:response"
        cache.set(vec_key, embedding.tolist())
        cache.set(res_key, response)

    return PromptResponse(session_id=req.session_id, response=response, from_cache=False)
