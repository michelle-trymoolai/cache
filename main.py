import json
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from cache.redis_backend import RedisCache
from cache.config import CacheConfig
from chat.models import PromptRequest
from chat.logic import process_prompt
from chat.logic import embed_text, bot_logic  
from chat.models import CacheWarmRequest
import hashlib
from fastapi import APIRouter
import time


import io

app = FastAPI(title="FastAPI CacheBot")

cache = RedisCache()

@app.get("/cache/stats")
def stats(): return cache.stats()

@app.post("/cache/clear")
def clear(): cache.clear(); return {"message": "Cache cleared"}

@app.post("/cache/enable")
def enable(): CacheConfig.ENABLED = True; return {"message": "Cache enabled"}

@app.post("/cache/disable")
def disable(): CacheConfig.ENABLED = False; return {"message": "Cache disabled"}

@app.get("/cache/list")
def list_keys(): return {"keys": cache.list_keys()}

@app.get("/cache/export/json")
def export_json(): return cache.export_json()

@app.get("/cache/export/csv")
def export_csv():
    content = cache.export_csv()
    return StreamingResponse(io.StringIO(content), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=cache.csv"})

@app.get("/cache/threshold")
def get_threshold(): return {"threshold": CacheConfig.SIMILARITY_THRESHOLD}

@app.post("/cache/threshold")
def set_threshold(value: float): CacheConfig.SIMILARITY_THRESHOLD = value; return {"message": "Threshold updated"}

@app.get("/cache/ttl")
def get_ttl(): return {"ttl": CacheConfig.CACHE_TTL}

@app.post("/cache/ttl")
def set_ttl(value: int): CacheConfig.CACHE_TTL = value; return {"message": "TTL updated"}

@app.get("/cache/analytics")
def get_cache_analytics():
    return cache.stats()

start_time = time.time()

@app.get("/health")
def health_check():
    try:
        # Test Redis connection
        cache.client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"

    uptime_seconds = int(time.time() - start_time)
    uptime = f"{uptime_seconds // 3600} hours {(uptime_seconds % 3600) // 60} minutes"

    return {
        "redis": redis_status,
        "uptime": uptime,
        "keys": len(cache.list_keys()),
        "cache_enabled": CacheConfig.ENABLED
    }

@app.post("/cache/warm")
def warm_cache(data: CacheWarmRequest):
    session_id = data.session_id or "warmup"
    prompts = data.prompts or []
    full = data.mode == "full"

    for prompt in prompts:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        base_key = f"chat:{session_id}:{prompt_hash}"

        # Cache embedding
        embedding = embed_text(prompt)
        if embedding is not None:
            cache.set(f"{base_key}:vec", json.dumps(embedding.tolist()))

        # Cache response (optional)
        if full:
            req = PromptRequest(
                prompt=prompt,
                session_id=session_id,
                message=prompt
            )
            response = process_prompt(req)
            cache.set(base_key, {
                "prompt": prompt,
                "response": response.response or "",  # safety
                "label": response.label
            })
        else:
            cache.set(base_key, {
                "prompt": prompt,
                "response": "<WARMED>",
                "label": None
            })

    return {
        "message": f"Cache warmed with {len(prompts)} prompts",
        "mode": "full" if full else "embed_only"
    }

@app.get("/cache/debug/embeddings")
def debug_embeddings(session_id: str):
    prefix = f"chat:v1:{session_id}:"
    keys = list(cache.scan_iter(f"{prefix}*:vec"))
    result = {}
    for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else str(key)
        val = cache.get(key)
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        result[key_str] = val
    return result


@app.post("/process_prompt")
def chat(req: PromptRequest):
    final_message = req.message or req.prompt
    if not final_message:
        raise ValueError("Either 'message' or 'prompt' must be provided.")
    req.message = final_message
    return process_prompt(req)
