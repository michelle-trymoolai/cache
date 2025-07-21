import redis, zlib, json
from cache.config import CacheConfig

class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=CacheConfig.REDIS_HOST,
            port=CacheConfig.REDIS_PORT,
            decode_responses=False  # we want raw bytes
        )

    def _compress(self, value):
        # Convert lists/dicts to string before compressing
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        return zlib.compress(value.encode())

    def _decompress(self, value):
        decompressed = zlib.decompress(value).decode()
        try:
            return json.loads(decompressed)  # if it's JSON
        except json.JSONDecodeError:
            return decompressed  # plain string

    def set(self, key, value, ttl=None):
        if not CacheConfig.ENABLED:
            return
        ttl = ttl or CacheConfig.CACHE_TTL
        self.client.setex(key, ttl, self._compress(value))

    def get(self, key):
        val = self.client.get(key)
        return self._decompress(val) if val else None

    def list_keys(self):
        return [k.decode() for k in self.client.keys("*")]

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

    def stats(self):
        return {
            "enabled": CacheConfig.ENABLED,
            "ttl": CacheConfig.CACHE_TTL,
            "threshold": CacheConfig.SIMILARITY_THRESHOLD,
            "keys": self.list_keys()
        }
