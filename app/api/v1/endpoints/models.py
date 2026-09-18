"""模型清单接口：/api/v1/models

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/v1/models/list` | 返回配置里所有可用模型（**不含任何 API Key**） |

这是前端"模型下拉框"的数据源。它回答三个问题：

1. 有哪些模型可以选（`id` / `label` / `provider` / 说明文字）；
2. 哪些模型是本地跑的（`is_local`，不需要填 Key）；
3. 后端是不是已经配好了 Key（`has_default_key`，**只给布尔值**）——
   配好了就不用逼着学生填，没配好前端就提示"请输入你自己的 Key"。

顺带还会带上当前会话的状态（`session`），前端刷新页面后据此判断
"我上次填的 Key 还在不在"。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header

from app.api.deps import ApiKeyManagerDep, AppSettingsDep
from app.core.api_key_manager import SESSION_HEADER, ApiKeyNotFound
from app.schemas.auth import ModelListResponse, ModelOptionRead, SessionStateRead

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/list",
    response_model=ModelListResponse,
    summary="可用模型清单（不含 API Key）",
    description=(
        "返回 `.env` 里配置的全部模型：ID、展示名、提供商、模型名、"
        "是否需要 API Key、后端是否已有默认 Key。\n\n"
        "**返回值里永远没有 API Key**：只在 `has_default_key` 里回答"
        "「后端配没配」，具体值不出服务端。"
    ),
)
async def list_models(
    settings: AppSettingsDep,
    manager: ApiKeyManagerDep,
    x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> ModelListResponse:
    # get_available_models 的返回值已经严格过滤过（见 config.LLMSettings.public_models），
    # 这里再过一遍 ModelOptionRead：万一以后有人往清单里塞了新字段，
    # 响应模型只认声明过的字段，多余的东西不会被带出去（多一道保险）。
    from app.core.config import get_available_models

    models = [ModelOptionRead(**item) for item in get_available_models(settings)]

    # 会话状态：查不到（没带会话号 / 已过期）就当"没有会话"，不算错误
    session_state: SessionStateRead | None = None
    session_id = (x_session_id or "").strip()
    if session_id:
        try:
            session_state = SessionStateRead(**manager.get(session_id).to_public_dict())
        except ApiKeyNotFound:
            session_state = None

    logger.debug(
        "返回模型清单：共 %s 个，其中默认 %s（会话=%s）",
        len(models),
        settings.llm.default_model_id,
        (session_id[:6] + "…") if session_id else "无",
    )

    return ModelListResponse(
        models=models,
        default_model_id=settings.llm.default_model_id,
        allow_client_key=settings.security.allow_client_key,
        session=session_state,
        total=len(models),
    )


__all__ = ["router"]
