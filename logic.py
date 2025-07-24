import hashlib
import os
import json
import numpy as np
from dotenv import load_dotenv
#from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Optional

from chat.models import PromptRequest, PromptResponse
from cache.redis_backend import RedisCache
from cache.config import CacheConfig

# Load environment variables
load_dotenv()
local_embedder = SentenceTransformer("all-MiniLM-L6-v2")


# OpenAI client
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cache = RedisCache()

USE_SEMANTIC_CACHE = getattr(CacheConfig, "USE_SEMANTIC_CACHE", False)
#SIMILARITY_THRESHOLD = getattr(CacheConfig, "SIMILARITY_THRESHOLD", 0.9)


def generate_key(session_id: str, message: str) -> str:
    hash_value = hashlib.sha256(message.encode()).hexdigest()
    return f"chat:{session_id}:{hash_value}"


def embed_text(text: str) -> np.ndarray:
    try:
        embedding = local_embedder.encode([text])[0]
        return np.array(embedding)
    except Exception as e:
        print(f"[Local Embedding Error] {e}")
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



def find_semantic_match(session_id: str, query_embedding: List[float]) -> Optional[str]:
    # Scan for versioned vector keys
    pattern = f"chat:v1:{session_id}:*:vec"
    keys = list(cache.scan_iter(match=pattern))
    best_key = None
    best_score = -1.0

    for key in keys:
        # Normalize key to string
        key_str = key.decode() if isinstance(key, bytes) else str(key)
        # Retrieve the stored embedding
        embedding = cache.get(key_str)
        if not embedding:
            continue

        try:
            # Convert embedding to numpy array
            stored_vec = np.array(embedding, dtype=np.float32)
            score = cosine_similarity(query_embedding, stored_vec)
            print(f"[Debug] Compared with {key_str} → Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_key = key_str
        except Exception as e:
            print(f"Error comparing embeddings for {key_str}: {e}")
            continue

    if best_score >= CacheConfig.SIMILARITY_THRESHOLD:
        print(f"[Semantic Match] Key: {best_key}, Score: {best_score:.4f}")
        return best_key

    print(f"[Semantic Miss] Best score: {best_score:.4f} (Threshold: {CacheConfig.SIMILARITY_THRESHOLD})")
    return None

def classify_prompt(prompt: str) -> str:
    prompt = prompt.lower()

    # AI / Machine Learning
    if any(term in prompt for term in ["transformer", "bert", "gpt", "llm", "attention mechanism"]):
        return "AI.NLP.TransformerModels"
    if any(term in prompt for term in ["classification", "regression", "supervised", "unsupervised"]):
        return "AI.ML.ModelTypes"
    if "embedding" in prompt or "vector search" in prompt:
        return "AI.NLP.SemanticSearch"

    # Finance
    if any(term in prompt for term in ["revenue", "recognition", "accrual", "invoice"]):
        return "Finance.Accounting.RevenueRecognition"
    if any(term in prompt for term in ["credit score", "risk", "loan"]):
        return "Finance.Risk.CreditScoring"
    if "tax" in prompt or "filing" in prompt:
        return "Finance.Tax.Compliance"

    # Medicine / Healthcare
    if any(term in prompt for term in ["diagnosis", "symptom", "treatment"]):
        return "Medicine.General.Diagnosis"
    if "oncology" in prompt or "cancer" in prompt:
        return "Medicine.Oncology.TreatmentPlan"
    if any(term in prompt for term in ["insurance", "authorization", "payer"]):
        return "Medicine.Admin.PriorAuth"

    # Tech / Databases
    if any(term in prompt for term in ["sql", "join", "query", "select", "index"]):
        return "Tech.Databases.SQL"
    if any(term in prompt for term in ["nosql", "redis", "cache", "memory store"]):
        return "Tech.Systems.Caching"
    if any(term in prompt for term in ["api", "endpoint", "rest", "postman"]):
        return "Tech.Backend.API"

    # General / Fallback
    return "General.Uncategorized"

def process_prompt(req: PromptRequest) -> PromptResponse:
    message = req.message or req.prompt
    embedding = None

    if USE_SEMANTIC_CACHE:
        embedding = embed_text(message)
        if embedding is not None:
            match = find_semantic_match(req.session_id, embedding)
            if match:
                full_entry = cache.get(match.replace(":vec", ""))
                if isinstance(full_entry, dict):
                    response = full_entry.get("response") or ""
                    label = full_entry.get("label") or ""
                else:
                    # fallback if structure hasn't been updated yet
                    response = full_entry or ""
                    label = None

                stored_vec = np.array(cache.get(match), dtype=np.float32)
                score = float(cosine_similarity(embedding, stored_vec))
                print(f"[CACHE HIT] Semantic match: {match}, Score: {score:.4f}")
                return PromptResponse(
                    session_id=req.session_id,
                    response=response,
                    from_cache=True,
                    similarity=score,
                    label=label
                )

    # exact-match fallback
    key = generate_key(req.session_id, message)
    cached = cache.get(key)
    if cached:
        print(f"[CACHE HIT] Exact match: {key}")
        if isinstance(cached, dict):
            return PromptResponse(
                session_id=req.session_id,
                response=cached.get("response") or "",
                from_cache=True,
                similarity=None,
                label=cached.get("label") or ""
            )
        else:
            return PromptResponse(
                session_id=req.session_id,
                response=cached or "",
                from_cache=True,
                similarity=None
            )

    # generate fresh
    response = f"Answer to: {message}" 
    label = classify_prompt(message)

    redis_payload = {
        "prompt": message,
        "response": response,
        "label": label
    }
    cache.set(key, redis_payload)

    if USE_SEMANTIC_CACHE and embedding is not None:
        vec_key = f"{key}:vec"
        cache.set(vec_key, json.dumps(embedding.tolist()))

    print(f"[CACHE MISS] Freshly generated: {message}")
    return PromptResponse(
        session_id=req.session_id,
        response=response,
        from_cache=False,
        similarity=None,
        label=label
    )
