from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# 基础响应模型
class BaseResponse(BaseModel):
    code: int = Field(description="响应状态码")
    msg: str = Field(description="响应消息")
    data: Optional[Dict[str, Any]] = Field(description="响应数据", default=None)
