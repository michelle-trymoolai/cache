from pydantic import BaseModel
from typing import List, Literal, Optional

class PromptRequest(BaseModel):
    session_id: str
    message: str | None = None
    prompt: str | None = None

class PromptResponse(BaseModel):
    session_id: str
    response: str = ""
    from_cache: bool
    similarity: Optional[float] = None
    label: Optional[str] = None

class CacheWarmRequest(BaseModel):
    session_id: str
    prompts: List[str]
    mode: Optional[str] = "embed_only"
