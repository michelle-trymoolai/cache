import redis, zlib, json, time
from cache.config import CacheConfig
import numpy as np

class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=CacheConfig.REDIS_HOST,
            port=CacheConfig.REDIS_PORT,
            decode_responses=False  # store raw bytes
        )

    def _versioned(self, key):

        if isinstance(key, bytes):
            key = key.decode("utf-8")

        if key.startswith("chat:v1:"):
            return key

        if key.startswith("chat:"):
            return "chat:v1:" + key[len("chat:"):]

        return key


    def _compress(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        return zlib.compress(value.encode())

    def _decompress(self, value):
        decompressed = zlib.decompress(value).decode()
        try:
            return json.loads(decompressed)
        except json.JSONDecodeError:
            return decompressed

    def set(self, key, value, ttl=None):
        if not CacheConfig.ENABLED:
            return
        ttl = ttl or CacheConfig.CACHE_TTL
        now = time.time()
        key = self._versioned(key)
        self.client.setex(key, ttl, self._compress(value))

        # Metadata
        meta_key = f"{key}:meta"
        meta = {
            "created_at": now,
            "last_accessed": now,
            "model": "text-embedding-3-small"
        }
        self.client.setex(meta_key, ttl, self._compress(meta))

    def get(self, key):
        key = self._versioned(key)
        val = self.client.get(key)
        if val:
            # Update last_accessed
            meta_key = f"{key}:meta"
            if self.client.exists(meta_key):
                try:
                    meta = self._decompress(self.client.get(meta_key))
                    meta["last_accessed"] = time.time()
                    self.client.setex(meta_key, CacheConfig.CACHE_TTL, self._compress(meta))
                except:
                    pass
            return self._decompress(val)
        return None

    def list_keys(self):
        keys = [k.decode() for k in self.client.keys("*") if b":meta" not in k]
        result = []
        for k in keys:
            meta_key = f"{k}:meta"
            meta_raw = self.client.get(meta_key)
            if meta_raw:
                try:
                    meta = self._decompress(meta_raw)
                    result.append({"key": k, **meta})
                except:
                    result.append({"key": k})
            else:
                result.append({"key": k})
        return result

    def clear(self):
        self.client.flushdb()

    def export_json(self):
        return {k: self.get(k) for k in self.list_keys()}

    def export_csv(self):
        import csv, io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Key", "Value"])
        for k in self.list_keys():
            writer.writerow([k, self.get(k)])
        return output.getvalue()

    def scan_iter(self, match=None):  
        return self.client.scan_iter(match=match)

    def stats(self):
        return {
            "enabled": CacheConfig.ENABLED,
            "ttl": CacheConfig.CACHE_TTL,
            "threshold": CacheConfig.SIMILARITY_THRESHOLD,
            "keys": self.list_keys()
        }
