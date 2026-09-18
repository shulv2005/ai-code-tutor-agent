"""API Key 会话接口：/api/v1/auth

三个能力：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| POST | `/api/v1/auth/set_key` | 把用户在网页上填的 Key 存进**内存会话**（不落库、不进日志） |
| POST | `/api/v1/auth/clear_key` | 清除会话里的 Key（"退出"或"换一把"时调） |
| GET | `/api/v1/auth/status` | 查当前会话状态（只回"有没有 Key"，不回 Key） |

安全约定（本模块的核心）：

1. **Key 只走请求体**，绝不走 URL 查询串——查询串会被访问日志原样打印。
2. **响应里没有 Key**：所有返回值都由 `SessionStateRead` 组装，它压根没有
   `api_key` 这个字段；直接返回记录对象也不可能，因为 `SessionRecord`
   的 `to_public_dict()` 只给布尔值。
3. **错误信息先脱敏**：任何可能带上 Key 的文本，写日志/返回前都过一遍
   `redact()`。
4. **会话号本身就是凭据**：它由后端用 `secrets.token_urlsafe` 生成，
   前端要把它存好（放 `X-Session-Id` 请求头带回来），泄露了等于泄露 Key。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.api.deps import ApiKeyManagerDep, AppSettingsDep
from app.core.api_key_manager import (
    SESSION_HEADER,
    ApiKeyNotFound,
    ApiKeyRejected,
    redact,
)
from app.schemas.auth import (
    ClearKeyRequest,
    ClearKeyResponse,
    SessionStateRead,
    SetKeyRequest,
    SetKeyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _pick_session_id(
    session_id: str | None, header_value: str | None
) -> str | None:
    """决定这次用哪个会话号：请求体优先，其次请求头。

    两种传法都支持，是因为它们各有各的合适场景：
      · 请求体 —— `POST` 接口（set_key / clear_key）写起来最直观；
      · 请求头 —— 以后 AI 接口（检测/改错/注释）会用它，
        这样表单里不用多带一个字段。
    """
    return (session_id or "").strip() or (header_value or "").strip() or None


def _session_payload(manager: ApiKeyManagerDep, session_id: str | None) -> SessionStateRead:
    """把会话记录转成"可以安全返回"的状态对象（不含 Key）。"""
    try:
        record = manager.get(session_id)
    except ApiKeyNotFound as exc:
        # 会话不存在/已过期：这不是服务端故障，告诉前端"重新填一次"即可
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=redact(str(exc)),
        ) from exc
    return SessionStateRead(**record.to_public_dict())


# ---------------------------------------------------------------------------
# 1. 存 Key
# ---------------------------------------------------------------------------
@router.post(
    "/set_key",
    response_model=SetKeyResponse,
    summary="保存用户填写的 API Key（仅内存，不落库）",
    description=(
        "把网页上填的 API Key 存进服务端**内存**里的会话，供后续调用模型使用。\n\n"
        "- 只存内存：进程重启即消失；数据库里不会留下任何 Key；\n"
        "- 只进不出：响应里只有「有没有 Key」，不会把 Key 回传；\n"
        "- 有有效期：默认 30 分钟无操作自动清除，每次使用会顺延；\n"
        "- 本地模型（如 Ollama）可以不填 Key。\n\n"
        "返回的 `session.session_id` 请前端保存好，后续请求放在 `X-Session-Id` 头里。"
    ),
)
async def set_api_key(
    request: Request,
    payload: SetKeyRequest,
    manager: ApiKeyManagerDep,
    settings: AppSettingsDep,
    x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> SetKeyResponse:
    # ---- 0. 后端可能整体关掉了"网页填 Key"这个能力 ----
    if not settings.security.allow_client_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "后端已关闭「网页填写 API Key」（SECURITY__ALLOW_CLIENT_KEY=false），"
                "请让管理员在 .env 里配置好 Key"
            ),
        )

    # ---- 1. 模型必须真实存在，否则后面调用必然失败 ----
    try:
        model = settings.llm.get_model(payload.model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc).strip("'\""),
        ) from exc

    # ---- 2. 云端模型必须有 Key；本地模型允许留空 ----
    api_key = (payload.api_key or "").strip()
    if not api_key and model.requires_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"模型「{model.display_name}」需要 API Key，请填写后再试；"
                "（本地模型如 Ollama 才允许留空）"
            ),
        )

    # ---- 3. 存进内存会话 ----
    session_id = _pick_session_id(payload.session_id, x_session_id)
    try:
        record = manager.set_key(model_id=payload.model_id, api_key=api_key, session_id=session_id)
    except ApiKeyRejected as exc:
        # 注意 redact：万一异常文本里拼进了用户输入，也先脱敏再返回
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=redact(str(exc), api_key)
        ) from exc

    # 日志里只有会话号前几位与模型名（manager 内部已经这样打了，这里不重复打 Key）
    logger.info(
        "收到网页提交的 Key：session=%s… model=%s 来源=%s",
        record.session_id[:6],
        payload.model_id,
        "新建会话" if not session_id else "已有会话",
    )

    message = (
        f"已保存「{model.display_name}」的 API Key（仅存在服务器内存里，"
        f"{settings.security.api_key_ttl_seconds // 60} 分钟不用会自动清除）"
        if api_key
        else f"「{model.display_name}」是本地模型，不需要 API Key"
    )
    return SetKeyResponse(
        session=SessionStateRead(**record.to_public_dict()),
        message=message,
        ttl_seconds=settings.security.api_key_ttl_seconds,
    )


# ---------------------------------------------------------------------------
# 2. 清除 Key
# ---------------------------------------------------------------------------
@router.post(
    "/clear_key",
    response_model=ClearKeyResponse,
    summary="清除会话里的 API Key（退出 / 换 Key 时调用）",
    description=(
        "不带 `model_id` 时清除整个会话（相当于「退出登录」）；"
        "带上 `model_id` 时只清掉这个模型的 Key，会话与其它模型保留。"
    ),
)
async def clear_api_key(
    payload: ClearKeyRequest,
    manager: ApiKeyManagerDep,
    x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> ClearKeyResponse:
    session_id = _pick_session_id(payload.session_id, x_session_id)
    if not session_id:
        # 没有会话号就没什么可清的；返回 200 + cleared=false，免得前端把它当报错
        return ClearKeyResponse(cleared=False, message="当前没有会话，无需清除")

    cleared = manager.clear_key(session_id, payload.model_id)
    message = (
        "已清除" if cleared else "没有找到对应的 Key（可能已经过期或清除过了）"
    )
    if cleared and payload.model_id:
        message = f"已清除「{payload.model_id}」的 API Key"
    elif cleared:
        message = "已清除本次会话的全部 API Key"
    return ClearKeyResponse(cleared=cleared, session_id=session_id, message=message)


# ---------------------------------------------------------------------------
# 3. 查会话状态（方便前端刷新页面后判断"还要不要再填一次"）
# ---------------------------------------------------------------------------
@router.get(
    "/status",
    response_model=SessionStateRead,
    summary="查看当前会话状态（不含 Key）",
    description=(
        "返回会话里存了哪些模型的 Key、什么时候过期。"
        "**不会返回 Key 本身**，只会告诉你「有没有」。"
    ),
)
async def session_status(
    manager: ApiKeyManagerDep,
    x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
    session_id: str | None = None,
) -> SessionStateRead:
    return _session_payload(manager, _pick_session_id(session_id, x_session_id))


__all__ = ["router"]
