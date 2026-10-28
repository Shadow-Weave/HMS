"""Provider factories: registry-driven construction of LLMs, embedders, indexes, rerankers and entity extractors.
提供者工厂：由注册表驱动构造 LLM、嵌入器、索引、重排器与实体抽取器，并提供共享的配置规整。
"""

# Purpose (EN): Turn (provider name, config) pairs into provider instances using configs.registry as the single truth.
# 目的（中）：以 configs.registry 为唯一真相，把（提供者名, 配置）对变成提供者实例。

# Input (EN): Provider names, config dicts or pydantic config instances, constructor kwargs.
# 输入（中）：提供者名、配置字典或 pydantic 配置实例、构造参数。

# Output (EN): Provider instances; ConfigError (CONFIG_001/002) or DependencyError (DEPS_001) on failure.
# 输出（中）：提供者实例；失败时抛出 ConfigError（CONFIG_001/002）或 DependencyError（DEPS_001）。

# State effect (EN): register_provider mutates configs.registry.REGISTRY in place; create() has no side effects.
# 状态影响（中）：register_provider 原地修改 configs.registry.REGISTRY；create() 无副作用。

from __future__ import annotations

import difflib
import importlib
import inspect
from typing import Any, ClassVar

import pydantic
from pydantic import BaseModel

from hms_elite.configs.registry import EXTRA_OF_PROVIDER, REGISTRY, supported
from hms_elite.exceptions import ConfigError, DependencyError, unwrap_hms_error


def check_extra_fields(model_cls: type[BaseModel], data: Any) -> Any:
    """Before-validator body shared by every provider config: unknown keys → ConfigError(CONFIG_001) with a hint.
    所有提供者配置共享的前置校验体：未知键 → 带提示的 ConfigError(CONFIG_001)。

    Input / 输入：
        model_cls: The pydantic config class being constructed.
        model_cls：正在构造的 pydantic 配置类。

        data: The raw input (only dicts are inspected).
        data：原始输入（仅检查字典）。

    Output / 输出：
        The unchanged data; raises ConfigError "Extra fields not allowed: X. Did you mean 'Y'?" on unknown keys.
        原样返回数据；遇到未知键抛出 ConfigError "Extra fields not allowed: X. Did you mean 'Y'?"。

    State effect / 状态影响：
        None.
        无。
    """
    if not isinstance(data, dict):
        return data
    allowed = list(model_cls.model_fields)
    unknown = [str(key) for key in data if key not in model_cls.model_fields]
    if not unknown:
        return data
    hint = difflib.get_close_matches(unknown[0], allowed, n=1)
    message = f"Extra fields not allowed: {', '.join(unknown)}."
    if hint:
        message += f" Did you mean '{hint[0]}'?"
    raise ConfigError(
        message,
        "CONFIG_001",
        details={"model": model_cls.__name__, "fields": unknown, "allowed": allowed},
        suggestion=f"allowed fields: {', '.join(allowed)}",
    )


def coerce_config(config_cls: type[BaseModel], config: BaseModel | dict | None, extra: dict | None = None) -> BaseModel:
    """Coerce None / dict / model into an instance of config_cls (caller dict never mutated).
    将 None / 字典 / 模型规整为 config_cls 实例（调用方字典从不被修改）。

    Input / 输入：
        config_cls: Target pydantic config class.
        config_cls：目标 pydantic 配置类。

        config: None (defaults), a dict of fields, or an existing model instance (passthrough when already the class).
        config：None（默认值）、字段字典，或已有模型实例（已是目标类时直接透传）。

        extra: Extra keyword fields merged over a dict/None config.
        extra：合并到字典/None 配置之上的额外关键字字段。

    Output / 输出：
        A config_cls instance; pydantic errors become ConfigError(CONFIG_001) or the embedded HMSError.
        config_cls 实例；pydantic 错误转为 ConfigError(CONFIG_001) 或其中包裹的 HMSError。

    State effect / 状态影响：
        None.
        无。
    """
    extra = dict(extra or {})
    if isinstance(config, config_cls) and not extra:
        return config
    if isinstance(config, BaseModel):
        data = {**config.model_dump(), **extra}
    elif config is None:
        data = extra
    elif isinstance(config, dict):
        data = {**config, **extra}
    else:
        raise ConfigError(
            f"config must be a dict, {config_cls.__name__} or None, got {type(config).__name__}",
            "CONFIG_001",
            details={"type": type(config).__name__},
        )
    try:
        return config_cls(**data)
    except pydantic.ValidationError as exc:
        inner = unwrap_hms_error(exc)
        if inner is not None:
            raise inner from exc
        errors = exc.errors()
        first = errors[0] if errors else {}
        where = ".".join(str(part) for part in (first.get("loc") or ()))
        raise ConfigError(
            f"invalid {config_cls.__name__} at '{where}': {first.get('msg', 'validation error')}",
            "CONFIG_001",
            details={"errors": errors, "model": config_cls.__name__},
        ) from exc


def load_class(path: str) -> type:
    """Import a dotted `module.Class` path.
    导入点分的 `module.Class` 路径。

    Input / 输入：
        path: Dotted path, e.g. 'hms_elite.llms.rule.RuleLLM'.
        path：点分路径，如 'hms_elite.llms.rule.RuleLLM'。

    Output / 输出：
        The class object; ImportError / AttributeError propagate to the caller.
        类对象；ImportError / AttributeError 直接传播给调用方。

    State effect / 状态影响：
        Imports the module (module-level side effects of that module apply).
        导入该模块（该模块的模块级副作用生效）。
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def register_provider(kind: str, name: str, class_path: str, config_class_path: str) -> None:
    """Register (or replace) a provider for a kind in configs.registry.REGISTRY.
    在 configs.registry.REGISTRY 中登记（或替换）某种类下的提供者。

    Input / 输入：
        kind: 'llm' | 'embedder' | 'index' | 'reranker' | 'entity_extractor'.
        kind：'llm' | 'embedder' | 'index' | 'reranker' | 'entity_extractor'。

        name: Provider name used in config envelopes.
        name：配置信封中使用的提供者名。

        class_path: Dotted path of the provider class.
        class_path：提供者类的点分路径。

        config_class_path: Dotted path of its config class.
        config_class_path：其配置类的点分路径。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        Mutates REGISTRY[kind] in place so envelope validation and factories share one truth.
        原地修改 REGISTRY[kind]，使信封校验与工厂共享同一真相。
    """
    if kind not in REGISTRY:
        raise ConfigError(
            f"unknown provider kind {kind!r}; expected one of {sorted(REGISTRY)}",
            "CONFIG_002",
            details={"kind": kind, "supported": sorted(REGISTRY)},
        )
    REGISTRY[kind][name] = (class_path, config_class_path)


class _Factory:
    """Base factory: resolve provider → (class, config class) through REGISTRY[kind].
    工厂基类：通过 REGISTRY[kind] 把提供者名解析为（实现类, 配置类）。
    """

    kind: ClassVar[str] = ""

    @classmethod
    def _resolve(cls, provider: str) -> tuple[type, type]:
        """Load the provider and config classes, mapping ImportError to DependencyError.
        加载提供者类与配置类，并把 ImportError 映射为 DependencyError。
        """
        table = REGISTRY.get(cls.kind, {})
        if provider not in table:
            names = supported(cls.kind)
            raise ConfigError(
                f"unknown {cls.kind} provider {provider!r}; supported: {', '.join(names)}",
                "CONFIG_002",
                details={"kind": cls.kind, "provider": provider, "supported": names},
                suggestion=f"use one of: {', '.join(names)}",
            )
        class_path, config_path = table[provider]
        try:
            impl = load_class(class_path)
            config_cls = load_class(config_path)
        except DependencyError:
            raise
        except ImportError as exc:
            package = getattr(exc, "name", None) or provider
            raise DependencyError(
                f"provider {provider!r} needs the {package!r} package: {exc}",
                "DEPS_001",
                details={"package": package, "extra": EXTRA_OF_PROVIDER.get(provider, provider)},
            ) from exc
        return impl, config_cls

    @classmethod
    def create(cls, provider: str, config: dict | BaseModel | None = None, **kwargs: Any) -> Any:
        """Instantiate a registered provider with a coerced config.
        用规整后的配置实例化已注册的提供者。

        Input / 输入：
            provider: Registered provider name.
            provider：已注册的提供者名。

            config: None → defaults; dict → merged with kwargs (never mutated); model instance → passthrough.
            config：None → 默认值；字典 → 与 kwargs 合并（不修改原字典）；模型实例 → 透传。

            kwargs: Extra config fields merged over the dict.
            kwargs：合并到字典之上的额外配置字段。

        Output / 输出：
            The provider instance.
            提供者实例。

        State effect / 状态影响：
            Imports the provider module; unknown provider → ConfigError(CONFIG_002); missing SDK → DependencyError.
            导入提供者模块；未知提供者 → ConfigError(CONFIG_002)；缺少 SDK → DependencyError。
        """
        impl, config_cls = cls._resolve(provider)
        cfg = coerce_config(config_cls, config, kwargs)
        return impl(cfg)

    @classmethod
    def register_provider(cls, name: str, class_path: str, config_class_path: str) -> None:
        """Register a provider for this factory's kind (delegates to the module-level function).
        为本工厂的种类登记提供者（委托给模块级函数）。

        Input / 输入：
            name: Provider name.
            name：提供者名。

            class_path: Dotted path of the provider class.
            class_path：提供者类的点分路径。

            config_class_path: Dotted path of its config class.
            config_class_path：其配置类的点分路径。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Mutates configs.registry.REGISTRY[kind].
            修改 configs.registry.REGISTRY[kind]。
        """
        register_provider(cls.kind, name, class_path, config_class_path)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Sorted provider names registered for this kind.
        本种类下已注册提供者名（排序）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of names.
            名称列表。

        State effect / 状态影响：
            None.
            无。
        """
        return supported(cls.kind)


def _accepts_param(impl: type, name: str) -> bool:
    """True when the provider's constructor declares a parameter called `name` (rerankers: `llm`; RuleLLM: `entity_extractor`).
    提供者构造函数声明了名为 `name` 的参数时为真（重排器：`llm`；RuleLLM：`entity_extractor`）。
    """
    try:
        return name in inspect.signature(impl).parameters
    except (TypeError, ValueError):
        return False


class LlmFactory(_Factory):
    """Factory for LLM providers (kind 'llm'); create() forwards an optional `entity_extractor` to the constructor.
    LLM 提供者工厂（种类 'llm'）；create() 把可选的 `entity_extractor` 转交给构造函数。
    """

    kind: ClassVar[str] = "llm"

    @classmethod
    def create(cls, provider: str, config: dict | BaseModel | None = None, **kwargs: Any) -> Any:
        """Instantiate an LLM: `entity_extractor` goes to constructors that declare it (RuleLLM), other kwargs to the config.
        实例化 LLM：`entity_extractor` 传给声明了该参数的构造函数（RuleLLM），其余 kwargs 并入配置。

        Input / 输入：
            provider: 'rule' | 'scripted' | 'openai' | 'anthropic' | a registered name.
            provider：'rule' | 'scripted' | 'openai' | 'anthropic' | 已注册名称。

            config: LLM config dict / model / None.
            config：LLM 配置字典 / 模型 / None。

            kwargs: `entity_extractor` (an EntityExtractorBase; dropped for providers without that parameter) plus config fields.
            kwargs：`entity_extractor`（EntityExtractorBase；提供者无此参数时丢弃）及配置字段。

        Output / 输出：
            The LLM instance.
            LLM 实例。

        State effect / 状态影响：
            Imports the provider module.
            导入提供者模块。
        """
        extractor = kwargs.pop("entity_extractor", None)
        impl, config_cls = cls._resolve(provider)
        cfg = coerce_config(config_cls, config, kwargs)
        if extractor is not None and _accepts_param(impl, "entity_extractor"):
            return impl(cfg, entity_extractor=extractor)
        return impl(cfg)


class EmbedderFactory(_Factory):
    """Factory for embedding providers (kind 'embedder').
    嵌入提供者工厂（种类 'embedder'）。
    """

    kind: ClassVar[str] = "embedder"


class IndexFactory(_Factory):
    """Factory for vector index providers (kind 'index'); create() takes store and dims as constructor kwargs.
    向量索引提供者工厂（种类 'index'）；create() 以构造参数形式接收 store 与 dims。
    """

    kind: ClassVar[str] = "index"

    @classmethod
    def create(cls, provider: str, config: dict | BaseModel | None = None, **kwargs: Any) -> Any:
        """Instantiate an index provider: `store` and `dims` go to the constructor, other kwargs into the config.
        实例化索引提供者：`store` 与 `dims` 传给构造函数，其余 kwargs 并入配置。

        Input / 输入：
            provider: 'sqlite' | 'memory' | a registered name.
            provider：'sqlite' | 'memory' | 已注册名称。

            config: Index config dict / model / None.
            config：索引配置字典 / 模型 / None。

            kwargs: `store` (StateStore, required by sqlite), `dims` (default 64), plus config fields.
            kwargs：`store`（StateStore，sqlite 必需）、`dims`（默认 64）及配置字段。

        Output / 输出：
            The index instance.
            索引实例。

        State effect / 状态影响：
            Imports the provider module.
            导入提供者模块。
        """
        store = kwargs.pop("store", None)
        dims = kwargs.pop("dims", 64)
        impl, config_cls = cls._resolve(provider)
        cfg = coerce_config(config_cls, config, kwargs)
        return impl(cfg, store=store, dims=dims)


def _accepts_llm(impl: type) -> bool:
    """True when the provider's constructor declares an `llm` parameter (LLMReranker does, OverlapReranker does not).
    提供者构造函数声明了 `llm` 参数时为真（LLMReranker 有，OverlapReranker 没有）。
    """
    return _accepts_param(impl, "llm")


class EntityExtractorFactory(_Factory):
    """Factory for entity extractor providers (kind 'entity_extractor'; selected by IngestConfig.entity_extractor).
    实体抽取器提供者工厂（种类 'entity_extractor'；由 IngestConfig.entity_extractor 选择）。
    """

    kind: ClassVar[str] = "entity_extractor"


class RerankerFactory(_Factory):
    """Factory for reranker providers (kind 'reranker'); create() forwards an optional `llm` to the constructor.
    重排器提供者工厂（种类 'reranker'）；create() 把可选的 `llm` 转交给构造函数。
    """

    kind: ClassVar[str] = "reranker"

    @classmethod
    def create(cls, provider: str, config: dict | BaseModel | None = None, **kwargs: Any) -> Any:
        """Instantiate a reranker: `llm` (an LLMBase) goes to the constructor when it accepts one, other kwargs to the config.
        实例化重排器：`llm`（LLMBase）在构造函数接受时传入，其余 kwargs 并入配置。

        Input / 输入：
            provider: 'overlap' | 'llm' | a registered name.
            provider：'overlap' | 'llm' | 已注册名称。

            config: Reranker config dict / model / None.
            config：重排器配置字典 / 模型 / None。

            kwargs: `llm` (LLMBase to bind; dropped for providers without an `llm` parameter) plus config fields.
            kwargs：`llm`（要绑定的 LLMBase；提供者无 `llm` 参数时丢弃）及配置字段。

        Output / 输出：
            The reranker instance.
            重排器实例。

        State effect / 状态影响：
            Imports the provider module.
            导入提供者模块。
        """
        llm = kwargs.pop("llm", None)
        impl, config_cls = cls._resolve(provider)
        cfg = coerce_config(config_cls, config, kwargs)
        # OverlapReranker has no `llm` parameter; a bound LLM is only meaningful to LLM-backed providers.
        # OverlapReranker 没有 `llm` 参数；绑定的 LLM 只对 LLM 驱动的提供者有意义。
        if llm is not None and _accepts_llm(impl):
            return impl(cfg, llm=llm)
        return impl(cfg)


__all__ = [
    "EmbedderFactory",
    "EntityExtractorFactory",
    "IndexFactory",
    "LlmFactory",
    "RerankerFactory",
    "check_extra_fields",
    "coerce_config",
    "load_class",
    "register_provider",
]
