"""多模型配置测试：.env 的解析、模型清单、API Key 的取值优先级。

这一批用例守的是三件事：

1. **`.env` 能写出多个模型来**：`LLM__MODELS__<ID>__*` 能被解析成模型表，
   而且能自动补出 API 地址、自动判断"要不要 Key"（本地模型不要）。
2. **Key 绝不外泄**：`get_available_models()` 是给前端下拉框用的，
   里面只能有"后端有没有 Key"这个布尔值，不能出现 Key 本身。
3. **Key 的取值优先级可预期**：网页输入的 > 模型自带的 > 指定的环境变量 > 全局兜底，
   并且本地模型不吃全局兜底（免得把云端密钥发给本机服务）。

两个测试习惯，都是为了"结果不受开发机影响"：
- 构造配置一律带 `_env_file=None`，把真实的 `.env` 排除掉；
- 配置项一律通过 `monkeypatch.setenv` 注入，走**和真实启动完全一样**的解析路径
  （pydantic-settings 只会把环境变量按 `__` 拆成嵌套字段，
  而 `Settings(LLM__MODELS__X__Y=...)` 这种初始化参数是不会被拆的，写了也不生效）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import (
    LOCAL_PROVIDERS,
    PROJECT_ROOT,
    PROVIDER_DEFAULT_BASE_URLS,
    LLMModelSettings,
    LLMSettings,
    Settings,
    get_available_models,
)


# ---------------------------------------------------------------------------
# 工具：按 .env 的写法注入配置，再构造 Settings
# ---------------------------------------------------------------------------
def build(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """注入若干环境变量并构造配置（不读真实 .env，保证可复现）。"""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


@pytest.fixture()
def multi_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """三个模型：DeepSeek（自带 Key）、Qwen（没 Key）、Ollama（本地、免 Key）。"""
    monkeypatch.setenv("LLM__API_KEY", "")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__LABEL", "DeepSeek 官方")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__PROVIDER", "deepseek")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__API_KEY", "sk-deepseek")
    monkeypatch.setenv("LLM__MODELS__QWEN__PROVIDER", "qwen")
    monkeypatch.setenv("LLM__MODELS__QWEN__MODEL_NAME", "qwen-plus")
    monkeypatch.setenv("LLM__MODELS__OLLAMA__PROVIDER", "ollama")
    monkeypatch.setenv("LLM__MODELS__OLLAMA__MODEL_NAME", "qwen2.5-coder:7b")
    monkeypatch.setenv("LLM__DEFAULT_MODEL", "deepseek")


# ---------------------------------------------------------------------------
# 1. 解析：一个 .env 里能写多个模型
# ---------------------------------------------------------------------------
def test_multiple_models_are_parsed(multi_model_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """三个模型都要出现在模型表里，顺序与 .env 里写的一致。"""
    assert list(build(monkeypatch).llm.model_table) == ["deepseek", "qwen", "ollama"]


def test_model_fields_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID 与 provider 统一小写、首尾空格去掉——否则前端选了模型会对不上。"""
    settings = build(
        monkeypatch,
        LLM__MODELS__X__PROVIDER="  DeepSeek  ",
        LLM__MODELS__X__MODEL_NAME="  deepseek-chat  ",
    )
    model = settings.llm.get_model("x")
    assert model.provider == "deepseek"
    assert model.model_name == "deepseek-chat"


def test_base_url_is_filled_from_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """只写 provider 时自动补出官方地址，同学少填一项就少错一次。"""
    settings = build(
        monkeypatch,
        LLM__MODELS__Q__PROVIDER="qwen",
        LLM__MODELS__Q__MODEL_NAME="qwen-plus",
    )
    assert settings.llm.get_model("q").base_url == PROVIDER_DEFAULT_BASE_URLS["qwen"]


def test_explicit_base_url_wins_and_trailing_slash_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自己填的地址优先；末尾斜杠要去掉，否则会拼出 `//chat/completions`。"""
    settings = build(
        monkeypatch,
        LLM__MODELS__CUSTOM__PROVIDER="deepseek",
        LLM__MODELS__CUSTOM__MODEL_NAME="m",
        LLM__MODELS__CUSTOM__BASE_URL="http://192.168.1.10:8000/v1/",
    )
    assert settings.llm.get_model("custom").base_url == "http://192.168.1.10:8000/v1"


def test_label_defaults_to_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """没写 LABEL 时网页上显示模型名，不至于出现一个空白选项。"""
    settings = build(
        monkeypatch,
        LLM__MODELS__Q__PROVIDER="deepseek",
        LLM__MODELS__Q__MODEL_NAME="deepseek-chat",
    )
    assert settings.llm.get_model("q").display_name == "deepseek-chat"


@pytest.mark.parametrize("provider", sorted(LOCAL_PROVIDERS))
def test_local_providers_do_not_require_api_key(provider: str) -> None:
    """本地模型（Ollama / vLLM / LM Studio）自动判定为"不需要 Key"。"""
    model = LLMModelSettings(
        provider=provider, model_name="local-model",
        base_url="http://127.0.0.1:11434/v1",
    )
    assert model.is_local is True
    assert model.requires_api_key is False


def test_loopback_base_url_counts_as_local() -> None:
    """provider 写的是自定义名字，但地址指向本机时也算本地模型。"""
    model = LLMModelSettings(
        provider="my-gateway", model_name="m", base_url="http://127.0.0.1:9000/v1"
    )
    assert model.is_local is True
    assert model.requires_api_key is False


def test_cloud_model_requires_api_key_by_default() -> None:
    model = LLMModelSettings(provider="deepseek", model_name="deepseek-chat")
    assert model.is_local is False
    assert model.requires_api_key is True


def test_requires_api_key_can_be_forced() -> None:
    """有些自建网关也要校验 token，就手动把 REQUIRES_API_KEY 打开。"""
    model = LLMModelSettings(provider="ollama", model_name="m", requires_api_key=True)
    assert model.requires_api_key is True


# ---------------------------------------------------------------------------
# 2. 配置写错时要当场报错，而不是等到"点了没反应"
# ---------------------------------------------------------------------------
def test_missing_model_name_is_rejected() -> None:
    """MODEL_NAME 是必填项：没有它根本不知道该请求哪个模型。"""
    with pytest.raises(ValidationError):
        LLMModelSettings(provider="deepseek", model_name="")


def test_unknown_provider_without_base_url_is_rejected() -> None:
    """provider 不认识又没填 BASE_URL 时，明确报错而不是猜一个地址。"""
    with pytest.raises(ValidationError):
        LLMModelSettings(provider="some-new-vendor", model_name="m")


@pytest.mark.parametrize("bad_id", ["", "  ", "my model", "模型", "a/b", "a.b"])
def test_invalid_model_id_is_rejected(bad_id: str) -> None:
    """模型 ID 只允许 ASCII 字母数字下划线连字符——它会长在环境变量名里。"""
    with pytest.raises(ValidationError):
        LLMSettings(models={bad_id: LLMModelSettings(provider="deepseek", model_name="m")})


def test_uppercase_model_id_is_accepted_and_lowered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.env` 里写成大写（LLM__MODELS__DeepSeek__*）也要能用，内部统一转小写。"""
    settings = build(
        monkeypatch,
        LLM__MODELS__DEEPSEEK__PROVIDER="deepseek",
        LLM__MODELS__DEEPSEEK__MODEL_NAME="deepseek-chat",
        LLM__DEFAULT_MODEL="DEEPSEEK",
    )
    assert list(settings.llm.model_table) == ["deepseek"]
    assert settings.llm.default_model_id == "deepseek"


def test_default_model_must_exist(multi_model_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM__DEFAULT_MODEL 写错时启动就失败，比"学生点了没用"好排查得多。"""
    with pytest.raises(ValidationError):
        build(monkeypatch, LLM__DEFAULT_MODEL="gpt-5")


# ---------------------------------------------------------------------------
# 3. 没配多模型时，退回单模型写法（老配置不能坏）
# ---------------------------------------------------------------------------
def test_falls_back_to_single_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个多模型都没配时，用 LLM__MODEL / BASE_URL / API_KEY 拼出 default 模型。"""
    settings = build(
        monkeypatch,
        LLM__MODEL="deepseek-chat",
        LLM__BASE_URL="https://api.deepseek.com/v1",
        LLM__API_KEY="sk-single",
    )
    llm = settings.llm
    assert list(llm.model_table) == ["default"]
    assert llm.default_model_id == "default"
    model = llm.get_model()
    assert model.model_name == "deepseek-chat"
    assert model.api_key.get_secret_value() == "sk-single"


def test_default_model_id_uses_first_when_unspecified(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM__DEFAULT_MODEL 留空时用第一个模型，顺序就是 .env 里的书写顺序。"""
    settings = build(
        monkeypatch,
        LLM__MODELS__B__PROVIDER="deepseek",
        LLM__MODELS__B__MODEL_NAME="m-b",
        LLM__MODELS__A__PROVIDER="deepseek",
        LLM__MODELS__A__MODEL_NAME="m-a",
    )
    assert settings.llm.default_model_id == "b"


def test_get_model_raises_with_helpful_message(
    multi_model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """取一个不存在的模型时，报错里要列出可选值，方便直接改对。"""
    with pytest.raises(KeyError) as excinfo:
        build(monkeypatch).llm.get_model("gpt-4")
    assert "deepseek" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. get_available_models()：给前端用，绝不能带 Key
# ---------------------------------------------------------------------------
def test_available_models_contain_no_api_key(
    multi_model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型清单里不能出现 Key 本身，也不能出现 api_key 这种字段名。"""
    options = get_available_models(build(monkeypatch, LLM__API_KEY="sk-global-secret"))

    assert len(options) == 3
    for item in options:
        for value in item.values():
            assert "sk-" not in str(value)
        # 字段名里除了两个布尔量（requires_api_key / has_default_key）不该再出现 api_key
        assert not [key for key in item if "api_key" in key and key not in {
            "requires_api_key", "has_default_key",
        }]


def test_available_models_describe_each_model(
    multi_model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """前端要靠这些字段渲染下拉框与提示，字段一个都不能少。"""
    options = {item["id"]: item for item in get_available_models(build(monkeypatch))}

    deepseek = options["deepseek"]
    assert deepseek["label"] == "DeepSeek 官方"
    assert deepseek["provider"] == "deepseek"
    assert deepseek["model_name"] == "deepseek-chat"
    assert deepseek["base_url"] == "https://api.deepseek.com/v1"
    assert deepseek["requires_api_key"] is True
    assert deepseek["has_default_key"] is True      # 它自己配了 Key
    assert deepseek["is_default"] is True

    qwen = options["qwen"]
    assert qwen["has_default_key"] is False         # 没配 Key，全局兜底也是空的
    assert qwen["is_default"] is False

    ollama = options["ollama"]
    assert ollama["is_local"] is True
    assert ollama["requires_api_key"] is False


def test_global_key_shows_up_as_default_key_for_cloud_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全局 Key 会作为云端模型的兜底，因此 has_default_key 应当是 true。"""
    options = {
        item["id"]: item
        for item in get_available_models(
            build(
                monkeypatch,
                LLM__API_KEY="sk-global",
                LLM__MODELS__DEEPSEEK__PROVIDER="deepseek",
                LLM__MODELS__DEEPSEEK__MODEL_NAME="deepseek-chat",
                LLM__MODELS__OLLAMA__PROVIDER="ollama",
                LLM__MODELS__OLLAMA__MODEL_NAME="qwen2.5-coder:7b",
            )
        )
    }
    assert options["deepseek"]["has_default_key"] is True
    # 本地模型不吃兜底 Key：不然前端会误以为"这个模型已经配好了"
    assert options["ollama"]["has_default_key"] is False


def test_disabled_model_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENABLED=false 的模型不出现在清单里，等价于"暂时下架"。"""
    options = get_available_models(
        build(
            monkeypatch,
            LLM__MODELS__A__PROVIDER="deepseek",
            LLM__MODELS__A__MODEL_NAME="m-a",
            LLM__MODELS__B__PROVIDER="deepseek",
            LLM__MODELS__B__MODEL_NAME="m-b",
            LLM__MODELS__B__ENABLED="false",
        )
    )
    assert [item["id"] for item in options] == ["a"]


def test_get_available_models_uses_global_settings_by_default() -> None:
    """不传参数时用全局单例（接口层就是这么调的），至少要返回一个模型。"""
    assert get_available_models()


# ---------------------------------------------------------------------------
# 5. API Key 的取值优先级
# ---------------------------------------------------------------------------
def test_api_key_priority(multi_model_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """优先级：网页输入 > 模型自带 > 指定的环境变量 > 全局兜底。"""
    llm = build(monkeypatch, LLM__API_KEY="sk-global").llm
    deepseek = llm.get_model("deepseek")          # 它自己配了 sk-deepseek

    assert llm.effective_api_key(deepseek, "sk-from-web") == "sk-from-web"
    assert llm.effective_api_key(deepseek) == "sk-deepseek"
    assert llm.effective_api_key(llm.get_model("qwen")) == "sk-global"


def test_api_key_env_variable_is_supported(
    multi_model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API_KEY_ENV 可以从别的环境变量取 Key（"我机器上已经有 DASHSCOPE_API_KEY 了"）。"""
    llm = build(
        monkeypatch,
        LLM__API_KEY="sk-global",
        LLM__MODELS__QWEN__API_KEY_ENV="DASHSCOPE_API_KEY",
        DASHSCOPE_API_KEY="sk-from-env",
    ).llm
    assert llm.effective_api_key(llm.get_model("qwen")) == "sk-from-env"


def test_local_model_never_uses_global_key(
    multi_model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本地模型不吃全局兜底 Key：免得把云端密钥发给本机服务。"""
    llm = build(monkeypatch, LLM__API_KEY="sk-global").llm
    assert llm.effective_api_key(llm.get_model("ollama")) == ""


# ---------------------------------------------------------------------------
# 6. resolved_settings()：给 LLM 客户端用的适配层
# ---------------------------------------------------------------------------
def test_resolved_settings_switches_model(
    multi_model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """换个模型 ID，就得到一份指向该模型的单模型配置（客户端不用改代码）。"""
    resolved = build(monkeypatch, LLM__API_KEY="sk-global").llm.resolved_settings(
        "qwen", api_key="sk-from-web"
    )

    assert resolved.base_url == PROVIDER_DEFAULT_BASE_URLS["qwen"]
    assert resolved.model == "qwen-plus"
    assert resolved.api_key.get_secret_value() == "sk-from-web"


def test_resolved_settings_inherits_global_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型没写超时 / max_tokens 时继承全局；写了就以模型自己的为准。"""
    llm = build(
        monkeypatch,
        LLM__TIMEOUT_SECONDS="99",
        LLM__MAX_TOKENS="1234",
        LLM__MODELS__A__PROVIDER="deepseek",
        LLM__MODELS__A__MODEL_NAME="m-a",
        LLM__MODELS__B__PROVIDER="deepseek",
        LLM__MODELS__B__MODEL_NAME="m-b",
        LLM__MODELS__B__MAX_TOKENS="77",
    ).llm
    option_a = llm.resolved_settings("a")
    option_b = llm.resolved_settings("b")

    assert (option_a.timeout_seconds, option_a.max_tokens) == (99, 1234)
    assert option_b.max_tokens == 77
    assert option_b.timeout_seconds == 99          # 没写的项仍然继承全局


# ---------------------------------------------------------------------------
# 7. 模板本身要能解析（防止 .env.example 被改坏）
# ---------------------------------------------------------------------------
def test_env_example_parses_and_has_three_models() -> None:
    """.env.example 是发给别人的模板，必须能直接解析，且含需求里的三个模型。

    这里特意读真实文件：模板一旦被改坏（少个等号、模型 ID 写错），
    下一个人复制成 .env 就会启动失败——这条断言就是为了当场发现它。
    """
    settings = Settings(_env_file=str(PROJECT_ROOT / ".env.example"))
    assert {"deepseek", "qwen", "ollama"} <= set(settings.llm.model_table)


def test_env_example_contains_no_real_key() -> None:
    """模板里绝不能有真实 Key：分发出去等于把密钥公开。"""
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("LLM__") or "API_KEY" not in stripped or "=" not in stripped:
            continue
        value = stripped.split("=", 1)[1].strip()
        assert value in {"", "not-needed", "sk-replace-me", "DASHSCOPE_API_KEY"}, line


def test_env_example_models_are_documented() -> None:
    """云端模型要写清去哪申请 Key（网页上要显示给学生看）。"""
    settings = Settings(_env_file=str(PROJECT_ROOT / ".env.example"))
    for model_id, model in settings.llm.model_table.items():
        if model.is_local:
            continue
        assert model.description, f"模型 {model_id} 缺少 DESCRIPTION 说明"


def test_base_url_table_covers_common_providers() -> None:
    """常用服务商都要有内置默认地址，省得每个人自己查文档。"""
    for provider in ("deepseek", "qwen", "openai", "ollama"):
        assert provider in PROVIDER_DEFAULT_BASE_URLS
        assert PROVIDER_DEFAULT_BASE_URLS[provider].startswith("http")


def test_env_example_is_readable_utf8() -> None:
    """模板是中文写的，必须能被 UTF-8 正常读出（避免编辑器存成 GBK）。"""
    example = PROJECT_ROOT / ".env.example"
    assert "多模型配置" in example.read_text(encoding="utf-8")
    assert example.stat().st_size > 2000
