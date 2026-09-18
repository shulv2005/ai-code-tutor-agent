"""API Key 的临时保管箱：只放在进程内存里，不落库、不进日志、不回传。

## 它是干什么的

原来 API Key 只能写在 `.env` 里，一个服务只能用一个模型。现在同学可以
**在网页上选模型 + 填自己的 Key**，那些 Key 就临时存在这里。

## 三条硬规矩（安全设计）

1. **不落库**：内存字典，进程重启即消失。数据库里永远查不到 Key。
2. **不进日志**：本模块所有日志只打 `session=xxxxxx…`（会话号前 6 位）、
   模型名、动作，**从不打印 Key**；记录对象用 `SecretStr` 包着 Key，
   就算有人手滑 `logger.info(record)`，打出来的也是 `**********`。
   另外提供 `redact()`：调用模型失败时，服务商可能在响应里回显 Key，
   把错误信息写进日志/响应前先过一遍它，把 Key 抠掉。
3. **不回传**：所有对外的方法只回答"有没有 Key"（`has_key`），
   绝不把 Key 本身或其片段返回给前端；需要给日志看时用 `mask()`，
   而 `mask()` 只保留极短前缀。

## 会话怎么来

前端第一次调 `POST /api/v1/auth/set_key` 时不用带会话号，后端会发一个
（`secrets.token_urlsafe`，猜不出来）。之后前端把会话号放在
`X-Session-Id` 请求头（或请求体的 `session_id` 字段）里带回来即可。

一个会话里可以同时存多个模型的 Key（形如 `{会话号: {模型: Key}}`），
所以"切模型"不用重新填 Key。

## 过期

**滑动过期**：默认 30 分钟没动静就自动清掉；每次用（`resolve`）都会把
倒计时重置。清理是"顺手做"的（每次读写时顺带清一遍 + 显式调 `purge_expired`），
不引入后台线程——省得在测试和进程退出时多一份要管的东西。
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 默认有效期：30 分钟无操作即失效（需求里点名的时长）
DEFAULT_TTL_SECONDS = 30 * 60
# 内存里最多留多少个会话：这是个**没有登录**的接口，不设上限的话
# 有人反复调 set_key 就能把内存撑爆。超过上限时淘汰最久没用的那个。
DEFAULT_MAX_SESSIONS = 500
# 单个 Key 的长度上限：正常 Key 都在 200 字符以内，超了必然是误传/恶意
MAX_KEY_LENGTH = 512
# 会话号放在这个请求头里（前端记住它，后续请求带着）
SESSION_HEADER = "X-Session-Id"
# 会话号格式：URL 安全的随机串，只允许字母数字和 - _
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
# 生成会话号用的随机字节数（32 字节 ≈ 43 个字符，暴力猜不中）
SESSION_ID_BYTES = 32


# ---------------------------------------------------------------------------
# 异常：让调用方能区分"没配 Key"和"Key 过期了"
# ---------------------------------------------------------------------------
class ApiKeyStoreError(RuntimeError):
    """保管箱相关错误的基类。"""


class ApiKeyNotFound(ApiKeyStoreError):
    """会话不存在，或这个会话里没有对应模型的 Key。"""


class ApiKeyExpired(ApiKeyNotFound):
    """Key 曾经存在，但已经超过有效期被清掉了。"""


class ApiKeyRejected(ApiKeyStoreError):
    """Key 本身不合法（太长、空值、含奇怪字符等）。"""


# ---------------------------------------------------------------------------
# 掩码与脱敏：唯一被允许"看见"Key 的地方
# ---------------------------------------------------------------------------
def mask(secret: str, keep: int = 3) -> str:
    """把 Key 变成可以写进日志的样子，例如 `sk-***`。

    刻意**只保留开头几个字符**、不保留结尾：结尾几位在有些平台的 Key 里
    也是有信息量的；而开头这段足够我们回答"学生填的是不是 sk- 开头的那把"。

    Args:
        secret: 原始 Key。
        keep: 保留前几个字符（默认 3 个）。

    Returns:
        形如 `sk-***` 的字符串；空值返回 `(空)`。
    """
    text = (secret or "").strip()
    if not text:
        return "(空)"
    return f"{text[:keep]}***"


def redact(text: str, *secrets_to_hide: str) -> str:
    """把文本里出现过的 Key 全部替换成 `***`。

    用途：调用大模型失败时，服务商返回的错误信息里**有可能回显 Key**
    （例如 `Incorrect API key provided: sk-abc...`）。这类信息我们要写进
    日志、也要返回给前端看，所以先过一遍这个函数。

    Args:
        text: 原始文本（错误信息）。
        *secrets_to_hide: 需要抹掉的敏感串（通常是当前用的 Key）。

    Returns:
        抹掉敏感串之后的文本；长度超过 8 个字符的才处理（太短的串误伤面积太大）。
    """
    result = str(text or "")
    for secret in secrets_to_hide:
        value = (secret or "").strip()
        if len(value) >= 8 and value in result:
            result = result.replace(value, "***")
    return result


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SessionRecord:
    """一个会话里存的东西：多个模型各自的 Key + 时间戳。

    注意 `api_keys` 存的是 `SecretStr`：这样即使有人不小心把整个对象
    打进日志（`logger.info(record)`），输出的也是 `**********` 而不是明文。
    """

    session_id: str
    api_keys: dict[str, SecretStr] = field(default_factory=dict)
    # 会话里"最后设置的模型"，前端不带 model_id 时用它
    default_model_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    use_count: int = 0

    def __repr__(self) -> str:
        """自定义 repr：**不含 Key**，只报"存了几个模型的 Key"。"""
        return (
            f"SessionRecord(session_id={self.session_id[:6]}…, "
            f"models={sorted(self.api_keys)}, default={self.default_model_id or '-'}, "
            f"expires_at={self.expires_at.isoformat()})"
        )

    # ---- 不含敏感信息的对外视图 ----
    def to_public_dict(self) -> dict[str, object]:
        """转成"可以安全返回给前端 / 写进日志"的字典。

        Returns:
            只包含会话号、存了哪些模型、有没有 Key、什么时候过期——
            **没有一个字段是 Key 本身**。
        """
        return {
            "session_id": self.session_id,
            "models": sorted(self.api_keys),
            "default_model_id": self.default_model_id,
            "has_key": self.has_key,
            "expires_at": self.expires_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "use_count": self.use_count,
        }

    @property
    def has_key(self) -> bool:
        """这个会话里有没有"非空"的 Key（本地模型存的空串不算）。"""
        return any(secret.get_secret_value().strip() for secret in self.api_keys.values())

    def key_for(self, model_id: str) -> str:
        """取某个模型的 Key；没存过就抛 ApiKeyNotFound。"""
        secret = self.api_keys.get(model_id)
        if secret is None:
            raise ApiKeyNotFound(f"这个会话里没有 {model_id} 的 API Key")
        return secret.get_secret_value()

    def store(self, model_id: str, api_key: str) -> None:
        """写入某个模型的 Key（空串表示"这个模型不需要 Key"）。"""
        self.api_keys[model_id] = SecretStr(api_key)
        self.default_model_id = model_id


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    """一次调用模型要用到的凭据（已经解析好，可以直接喂给 LLM 客户端）。

    Attributes:
        model_id: 用哪个模型。空串表示"交给配置里的默认模型决定"。
        api_key: 要用的 Key；本地模型可能是空串。
        source: Key 从哪来的，取值 `request`（本次请求带的）/ `session`（会话里存的）
                / `config`（后端 .env 里配的）。日志里只打这个，不打 Key。
    """

    model_id: str
    api_key: str
    source: str

    def __repr__(self) -> str:
        """repr 里只有掩码，避免日志/异常里带出明文 Key。"""
        return (
            f"ResolvedCredential(model_id={self.model_id or '-'}, "
            f"api_key={mask(self.api_key)}, source={self.source})"
        )


# ---------------------------------------------------------------------------
# 保管箱本体
# ---------------------------------------------------------------------------
class ApiKeyManager:
    """进程内的 API Key 临时保管箱（线程安全）。

    为什么用进程内存而不是数据库/文件：
    - 需求要求"用完即弃"，内存最符合；
    - Key 一旦落盘，就得考虑加密、备份、误提交等一堆问题，收益却很小
      （同学在网页上重填一次只要几秒）。

    代价（如实说明）：进程重启后所有 Key 都会失效；多进程部署
    （`uvicorn --workers 4`）时每个进程各存一份，需要把请求固定到同一进程
    （或改用 Redis 之类的共享存储）。当前项目是单进程启动，够用。
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """
        Args:
            ttl_seconds: 有效期（秒）。每次使用都会顺延，即"无操作 N 秒后失效"。
            max_sessions: 最多保留多少个会话，超出时淘汰最久未使用的。
            clock: 取当前时间的函数（测试里注入假时钟，方便验过期逻辑）。
        """
        self._records: dict[str, SessionRecord] = {}
        # 用锁保护：FastAPI 的异步端点都在事件循环里跑，但同步端点会走线程池，
        # 加锁的成本可以忽略，换来的是"不会被并发写坏"。
        self._lock = threading.Lock()
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._max_sessions = max(1, int(max_sessions))
        self._clock = clock or (lambda: datetime.now(UTC))
        logger.info(
            "API Key 保管箱已就绪：有效期 %s 秒，最多 %s 个会话（仅内存，重启即清空）",
            self._ttl_seconds,
            self._max_sessions,
        )

    # ------------------------------------------------------------------
    # 会话与 Key 的写入
    # ------------------------------------------------------------------
    def new_session_id(self) -> str:
        """生成一个猜不出来的会话号（前端拿它来"认领"自己存的 Key）。"""
        return secrets.token_urlsafe(SESSION_ID_BYTES)

    def set_key(
        self,
        model_id: str,
        api_key: str,
        session_id: str | None = None,
    ) -> SessionRecord:
        """把前端传来的 Key 存进会话（会话不存在就新建）。

        Args:
            model_id: 模型 ID，例如 `deepseek`（必须是配置里存在的模型）。
            api_key: 用户输入的 Key。本地模型（Ollama 等）可以传空串。
            session_id: 已有的会话号；为空则新建一个会话。

        Returns:
            更新后的会话记录（**不要直接返回给前端**，请用 `to_public_dict()`）。

        Raises:
            ApiKeyRejected: Key 太长／会话号格式不对。
        """
        key = (api_key or "").strip()
        if len(key) > MAX_KEY_LENGTH:
            # 超长必然不是正常 Key；直接拒绝，避免有人拿它当"内存填充器"
            raise ApiKeyRejected(
                f"API Key 太长了（{len(key)} 字符，上限 {MAX_KEY_LENGTH}），请检查是否粘贴错了"
            )

        model = (model_id or "").strip()
        if not model:
            raise ApiKeyRejected("必须指定 model_id（要用哪个模型）")

        with self._lock:
            self._purge_locked()
            record = self._get_or_create_locked(session_id)
            record.store(model, key)
            self._touch_locked(record)
            self._evict_if_needed_locked()

        # 日志只有会话号前几位 + 模型名 + Key 掩码，没有明文
        logger.info(
            "已保存会话 Key：session=%s… model=%s key=%s",
            record.session_id[:6],
            model,
            mask(key),
        )
        return record

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get(self, session_id: str | None) -> SessionRecord:
        """取会话记录（并顺延有效期）。

        Raises:
            ApiKeyNotFound: 会话号为空/格式不对/不存在。
            ApiKeyExpired: 会话存在过，但已经过期。
        """
        if not session_id or not SESSION_ID_PATTERN.match(session_id.strip()):
            raise ApiKeyNotFound("缺少有效的会话标识（请先调用 /api/v1/auth/set_key）")

        with self._lock:
            record = self._records.get(session_id.strip())
            if record is None:
                raise ApiKeyNotFound("会话不存在或已失效，请重新填写 API Key")
            if record.expires_at <= self._clock():
                # 顺手删掉，别让过期数据占着内存
                self._records.pop(record.session_id, None)
                logger.info("会话已过期并清除：session=%s…", record.session_id[:6])
                raise ApiKeyExpired("API Key 已超过有效期（默认 30 分钟无操作），请重新填写")
            self._touch_locked(record)
            return record

    def get_key(self, session_id: str | None, model_id: str | None = None) -> str:
        """取某个模型的 Key（给调用模型的代码用）。

        Args:
            session_id: 会话号。
            model_id: 模型 ID；为空时用会话里"最后一次设置的模型"；
                若会话里只存了一个模型，也直接用它。

        Raises:
            ApiKeyNotFound / ApiKeyExpired: 取不到。
        """
        record = self.get(session_id)
        return self._key_from_record(record, model_id)

    def resolve(
        self,
        session_id: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> ResolvedCredential:
        """**调用模型前统一走这里**：算出这次到底用哪个模型、哪把 Key。

        优先级：
          1. 本次请求直接带的 `api_key`（前端"这次就用这把"）；
          2. 会话里存的（用户之前 set_key 过）；
          3. 都不给 → 返回 `source="config"`，交给后端的 `.env` 配置兜底。

        这样"网页上填 Key"和"后端配 Key"两条路能共存，谁也没配时会由上层
        给出友好提示（而不是在这里抛一个看不懂的错）。

        Args:
            session_id: 会话号（可空）。
            model_id: 模型 ID（可空，空则用会话/配置里的默认）。
            api_key: 本次请求直接带来的 Key（可空）。

        Returns:
            ResolvedCredential（`repr` 里只有掩码）。
        """
        request_key = (api_key or "").strip()
        wanted = (model_id or "").strip()
        if request_key:
            return ResolvedCredential(model_id=wanted, api_key=request_key, source="request")

        if session_id:
            try:
                record = self.get(session_id)
            except ApiKeyNotFound:
                # 会话没了/过期了不算错误：往下走，用配置里的 Key 兜底
                record = None
            if record is not None:
                # 会话里存了哪些模型是"用户自己填过的"，优先用
                target = wanted or record.default_model_id
                secret = record.api_keys.get(target) if target else None
                if secret is None and not wanted and len(record.api_keys) == 1:
                    # 会话里只存了一个模型，而这次没指定模型 → 就用它
                    target = next(iter(record.api_keys))
                    secret = record.api_keys[target]
                if secret is not None:
                    return ResolvedCredential(
                        model_id=target,
                        api_key=secret.get_secret_value(),
                        source="session",
                    )

        # 会话里没有这个模型的 Key：交给配置兜底
        return ResolvedCredential(model_id=wanted, api_key="", source="config")

    # ------------------------------------------------------------------
    # 清除
    # ------------------------------------------------------------------
    def clear_key(self, session_id: str | None, model_id: str | None = None) -> bool:
        """清除会话里的 Key。

        两种用法：
          - 只传 `session_id`：清掉整个会话（用户"退出/换人用"时调这个）；
          - 再传 `model_id`：只清掉这个模型的 Key，会话和别的模型保留
            （用户"换一把 Key"时调这个）。

        Returns:
            True = 真的清掉了东西；False = 本来就没有（不用当成错误）。
        """
        if not session_id or not SESSION_ID_PATTERN.match(session_id.strip()):
            return False

        with self._lock:
            record = self._records.get(session_id.strip())
            if record is None:
                return False

            if model_id:
                removed = record.api_keys.pop(model_id.strip(), None) is not None
                if record.default_model_id == model_id.strip():
                    # 默认模型被删了，换一个还留着的
                    record.default_model_id = next(iter(record.api_keys), "")
                if not record.api_keys:
                    self._records.pop(record.session_id, None)
                logger.info(
                    "已清除会话中某个模型的 Key：session=%s… model=%s",
                    record.session_id[:6],
                    model_id,
                )
                return removed

            self._records.pop(record.session_id, None)
            logger.info("已清除整个会话：session=%s…", record.session_id[:6])
            return True

    def clear_all(self) -> int:
        """清空所有会话（应用关闭、或测试之间清理时用）。

        Returns:
            清掉了几个会话。
        """
        with self._lock:
            count = len(self._records)
            self._records.clear()
        if count:
            logger.info("已清空全部会话 Key：共 %s 个", count)
        return count

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    def purge_expired(self) -> int:
        """清掉所有过期会话，返回清掉的个数。

        正常请求里会顺手清（见 `_purge_locked`），这个方法是给
        "想主动清一次"的场景用的（例如定时任务、验收脚本）。
        """
        with self._lock:
            return self._purge_locked()

    def stats(self) -> dict[str, object]:
        """保管箱的体检数据（不含任何 Key）。"""
        with self._lock:
            now = self._clock()
            active = sum(1 for item in self._records.values() if item.expires_at > now)
            return {
                "sessions": len(self._records),
                "active_sessions": active,
                "with_key": sum(1 for item in self._records.values() if item.has_key),
                "ttl_seconds": self._ttl_seconds,
                "max_sessions": self._max_sessions,
            }

    # ------------------------------------------------------------------
    # 内部实现（都要在持锁状态下调用）
    # ------------------------------------------------------------------
    def _get_or_create_locked(self, session_id: str | None) -> SessionRecord:
        """取出会话；不存在（或传来的号码格式不对）就新建一个。"""
        wanted = (session_id or "").strip()
        if wanted and SESSION_ID_PATTERN.match(wanted):
            existing = self._records.get(wanted)
            if existing is not None:
                return existing
        # 没带会话号，或带的号码已经不在了 → 新建
        record = SessionRecord(
            session_id=self.new_session_id(),
            expires_at=self._clock() + timedelta(seconds=self._ttl_seconds),
        )
        self._records[record.session_id] = record
        return record

    def _touch_locked(self, record: SessionRecord) -> None:
        """记一次"被用过"，并把有效期往后顺延（滑动过期）。"""
        now = self._clock()
        record.last_used_at = now
        record.expires_at = now + timedelta(seconds=self._ttl_seconds)
        record.use_count += 1

    def _purge_locked(self) -> int:
        """清掉过期会话；返回清掉的个数。"""
        now = self._clock()
        expired = [sid for sid, item in self._records.items() if item.expires_at <= now]
        for sid in expired:
            self._records.pop(sid, None)
        return len(expired)

    def _evict_if_needed_locked(self) -> None:
        """会话数超过上限时，淘汰最久没用过的那些。

        为什么需要：`set_key` 是**无需登录**就能调的接口，没有上限的话
        有人循环调用就能把内存吃光。淘汰最久未使用的会话，
        对正常用户几乎没有影响（他重新填一次 Key 即可）。
        """
        overflow = len(self._records) - self._max_sessions
        if overflow <= 0:
            return
        ordered = sorted(self._records.items(), key=lambda item: item[1].last_used_at)
        for session_id, _ in ordered[:overflow]:
            self._records.pop(session_id, None)
        logger.warning("会话数超过上限 %s，已淘汰最久未使用的 %s 个", self._max_sessions, overflow)

    @staticmethod
    def _key_from_record(record: SessionRecord, model_id: str | None) -> str:
        """从会话里挑出要用的 Key（不含安全逻辑，纯挑选规则）。"""
        wanted = (model_id or "").strip()
        if wanted:
            return record.key_for(wanted)
        if record.default_model_id:
            return record.key_for(record.default_model_id)
        if len(record.api_keys) == 1:
            # 只存了一个模型，用户没指定模型时就用它，省得前端每次都要带 model_id
            return next(iter(record.api_keys.values())).get_secret_value()
        raise ApiKeyNotFound("请指定 model_id（这个会话里存了多个模型的 Key）")


# ---------------------------------------------------------------------------
# 进程内单例
# ---------------------------------------------------------------------------
_manager: ApiKeyManager | None = None
_manager_lock = threading.Lock()


def get_api_key_manager() -> ApiKeyManager:
    """获取全局保管箱单例（配置变化后调用 reset_api_key_manager 重置）。"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                # 延迟导入：避免 config 与 core 模块之间的循环引用
                from app.core.config import get_settings

                settings = get_settings().security
                _manager = ApiKeyManager(
                    ttl_seconds=settings.api_key_ttl_seconds,
                    max_sessions=settings.max_sessions,
                )
    return _manager


def reset_api_key_manager() -> None:
    """清空单例（测试、或改了配置之后调用）。"""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.clear_all()
        _manager = None


__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_TTL_SECONDS",
    "MAX_KEY_LENGTH",
    "SESSION_HEADER",
    "SESSION_ID_PATTERN",
    "ApiKeyExpired",
    "ApiKeyManager",
    "ApiKeyNotFound",
    "ApiKeyRejected",
    "ApiKeyStoreError",
    "ResolvedCredential",
    "SessionRecord",
    "get_api_key_manager",
    "mask",
    "redact",
    "reset_api_key_manager",
]
