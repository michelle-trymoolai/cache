import os

class CacheConfig:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    CACHE_TTL = int(os.getenv("CACHE_TTL", 600))
    ENABLED = True

    # Semantic caching config
    USE_SEMANTIC_CACHE = True
    SIMILARITY_THRESHOLD = 0.92



