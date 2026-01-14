"""
Pydantic models for API requests and responses.
OpenAI API compatible schemas.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single message in the conversation."""

    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    messages: List[Message]
    model: str = Field(default="ragtime")
    stream: bool = Field(default=False)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    stop: Optional[List[str]] = Field(default=None)


class ChatChoice(BaseModel):
    """A single choice in the completion response."""

    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage = Field(default_factory=Usage)


class StreamChoice(BaseModel):
    """A single choice in a streaming response chunk."""

    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[dict] = Field(default_factory=list)
    root: str
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for listing available models."""

    object: str = "list"
    data: List[ModelInfo]


class MemoryStats(BaseModel):
    """Process memory statistics."""

    rss_mb: float  # Resident Set Size (actual RAM used)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of total system RAM
    available_mb: float  # Available system RAM
    total_mb: float  # Total system RAM


class IndexLoadingDetail(BaseModel):
    """Detail about a single index being loaded."""

    name: str
    status: str  # 'pending', 'loading', 'loaded', 'error'
    size_mb: Optional[float] = None
    chunk_count: Optional[int] = None
    load_time_seconds: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    indexes_loaded: List[str]
    model: str
    llm_provider: str
    # Progressive loading status
    indexes_ready: Optional[bool] = None
    indexes_loading: Optional[bool] = None
    indexes_total: Optional[int] = None
    indexes_loaded_count: Optional[int] = None
    # Real-time memory stats
    memory: Optional[MemoryStats] = None
    index_details: Optional[List[IndexLoadingDetail]] = None
    sequential_loading: Optional[bool] = None
    loading_index: Optional[str] = None  # Currently loading index name
