from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from cache.redis_backend import RedisCache
from cache.config import CacheConfig
from chat.models import PromptRequest
from chat.logic import process_prompt

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

@app.post("/process_prompt")
def chat(req: PromptRequest):
    return process_prompt(req)
