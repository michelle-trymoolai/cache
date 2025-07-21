from pydantic import BaseModel

class PromptRequest(BaseModel):
    session_id: str
    message: str

class PromptResponse(BaseModel):
    session_id: str
    response: str
    from_cache: bool
