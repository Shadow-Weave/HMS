"""
Test script for modular search strategies.
Validates all registered strategies and displays their interfaces.
"""

import asyncio
import inspect
from typing import get_type_hints, get_origin, get_args

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_subsection(title):
    print(f"\n--- {title} ---")

async def test_module_imports():
    """Test that all modules can be imported successfully."""
    print_section("模块导入测试")

    try:
        from hms_api.engine.search import (
            retrieval_registry,
            graph_retrieval_registry,
            fusion_registry,
            reranking_registry,
            get_module_manager,
            SearchConfig,
            SearchModuleManager,
            ModularRetrievalPipeline,
        )
        print("✓ 所有核心模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def analyze_class_interface(cls):
    """分析类的接口（方法、属性、参数类型）。"""
    methods = []
    properties = []
    async_methods = []

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('_'):
            continue
        sig = inspect.signature(method)
        hints = get_type_hints(method)

        # 获取参数信息
        params = []
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            param_type = hints.get(param_name, param.annotation)
            default = param.default if param.default != inspect.Parameter.empty else None
            params.append((param_name, param_type, default))

        method_info = {
            'name': name,
            'params': params,
            'return_type': hints.get('return', sig.return_annotation),
            'is_async': inspect.iscoroutinefunction(method)
        }

        if method_info['is_async']:
            async_methods.append(method_info)
        else:
            methods.append(method_info)

    # 获取属性
    for name in dir(cls):
        if name.startswith('_'):
            continue
        attr = getattr(cls, name, None)
        if isinstance(attr, property):
            properties.append({
                'name': name,
                'fget': attr.fget is not None,
                'fset': attr.fset is not None
            })

    return {
        'async_methods': async_methods,
        'methods': methods,
        'properties': properties
    }

def analyze_strategy_interface(registry, category_name):
    """分析注册表中的所有策略。"""
    print_section(f"{category_name} 策略分析")

    strategies = registry.list()
    print(f"共 {len(strategies)} 个策略: {strategies}\n")

    for strategy_name in strategies:
        strategy_cls = registry.get(strategy_name)
        if strategy_cls is None:
            continue

        print_subsection(f"策略: {strategy_name}")
        print(f"类名: {strategy_cls.__name__}")
        print(f"模块: {strategy_cls.__module__}")

        interface = analyze_class_interface(strategy_cls)

        # 打印属性
        if interface['properties']:
            print(f"\n属性:")
            for prop in interface['properties']:
                access = []
                if prop['fget']: access.append('get')
                if prop['fset']: access.append('set')
                print(f"  - {prop['name']} ({'/'.join(access)})")

        # 打印异步方法
        if interface['async_methods']:
            print(f"\n异步方法:")
            for method in interface['async_methods']:
                params_str = ', '.join([
                    f"{p[0]}: {format_type(p[1])}" + (f" = {p[2]}" if p[2] is not None else "")
                    for p in method['params']
                ])
                return_str = format_type(method['return_type']) if method['return_type'] != inspect.Signature.empty else 'None'
                print(f"  async {method['name']}({params_str}) -> {return_str}")

        # 打印同步方法
        if interface['methods']:
            print(f"\n方法:")
            for method in interface['methods']:
                if method['name'] in ['__init__', '__new__']:
                    continue
                params_str = ', '.join([
                    f"{p[0]}: {format_type(p[1])}" + (f" = {p[2]}" if p[2] is not None else "")
                    for p in method['params']
                ])
                return_str = format_type(method['return_type']) if method['return_type'] != inspect.Signature.empty else 'None'
                print(f"  {method['name']}({params_str}) -> {return_str}")

        print()

def format_type(typ):
    """格式化类型提示。"""
    if typ is None or typ == inspect.Signature.empty:
        return 'Any'

    origin = get_origin(typ)
    args = get_args(typ)

    if origin is None:
        if hasattr(typ, '__name__'):
            return typ.__name__
        return str(typ)

    # 处理泛型
    base_name = getattr(origin, '__name__', str(origin))
    if args:
        args_str = ', '.join([format_type(arg) for arg in args])
        return f"{base_name}[{args_str}]"
    return base_name

def test_strategy_creation():
    """测试策略创建。"""
    print_section("策略创建测试")

    from hms_api.engine.search import (
        retrieval_registry,
        graph_retrieval_registry,
        fusion_registry,
        reranking_registry,
    )

    test_cases = [
        ("检索策略", retrieval_registry, "semantic_bm25"),
        ("图检索策略", graph_retrieval_registry, "link_expansion"),
        ("融合策略", fusion_registry, "rrf"),
        ("重排序策略", reranking_registry, "passthrough"),
    ]

    for name, registry, default_name in test_cases:
        try:
            strategy = registry.create(default_name)
            print(f"✓ {name} '{default_name}' 创建成功")
            print(f"  实例类型: {type(strategy).__name__}")
            print(f"  策略名称: {strategy.name}")
            print()
        except Exception as e:
            print(f"✗ {name} '{default_name}' 创建失败: {e}")
            print()

def test_module_manager():
    """测试模块管理器。"""
    print_section("模块管理器测试")

    from hms_api.engine.search import SearchModuleManager, SearchConfig

    # 测试默认配置
    try:
        manager = SearchModuleManager()
        print("✓ 模块管理器创建成功")

        # 测试获取策略
        retrieval = manager.get_retrieval_strategy()
        print(f"  检索策略: {retrieval.name}")

        graph = manager.get_graph_retrieval_strategy()
        print(f"  图检索策略: {graph.name}")

        fusion = manager.get_fusion_strategy()
        print(f"  融合策略: {fusion.name}")

        rerank = manager.get_reranking_strategy()
        print(f"  重排序策略: {rerank.name}")

        # 测试列出所有可用策略
        available = manager.list_available_strategies()
        print(f"\n  所有可用策略: {available}")

        print()

    except Exception as e:
        print(f"✗ 模块管理器测试失败: {e}")
        print()

    # 测试自定义配置
    try:
        config = SearchConfig(
            retrieval_strategy="semantic_bm25",
            fusion_strategy="weighted",
            reranking_strategy="passthrough"
        )
        manager = SearchModuleManager(config)

        fusion = manager.get_fusion_strategy()
        print(f"✓ 自定义配置生效: 融合策略 = {fusion.name}")
        print()

    except Exception as e:
        print(f"✗ 自定义配置测试失败: {e}")
        print()

def test_strategy_override():
    """测试策略覆盖。"""
    print_section("策略覆盖测试")

    from hms_api.engine.search import get_module_manager, override_strategy

    try:
        manager = get_module_manager()

        # 保存原始策略
        original_fusion = manager.get_fusion_strategy().name
        print(f"原始融合策略: {original_fusion}")

        # 覆盖为加权融合
        override_strategy("fusion", "weighted", weights={"semantic": 0.6, "bm25": 0.4, "graph": 0, "temporal": 0})
        new_fusion = manager.get_fusion_strategy().name
        print(f"覆盖后融合策略: {new_fusion}")

        print("✓ 策略覆盖功能正常")
        print()

    except Exception as e:
        print(f"✗ 策略覆盖测试失败: {e}")
        print()

def test_config_dataclass():
    """测试配置数据类。"""
    print_section("SearchConfig 数据类测试")

    from hms_api.engine.search import SearchConfig

    try:
        config = SearchConfig(
            retrieval_strategy="semantic_bm25",
            graph_retrieval_strategy="link_expansion",
            fusion_strategy="weighted",
            reranking_strategy="cross_encoder",
            retrieval_params={"semantic_bm25": {"limit": 100}},
            fusion_params={"weighted": {"weights": {"semantic": 0.5, "bm25": 0.3}}}
        )

        print(f"✓ SearchConfig 创建成功")
        print(f"  检索策略: {config.retrieval_strategy}")
        print(f"  图检索策略: {config.graph_retrieval_strategy}")
        print(f"  融合策略: {config.fusion_strategy}")
        print(f"  重排序策略: {config.reranking_strategy}")
        print(f"  检索参数: {config.retrieval_params}")
        print(f"  融合参数: {config.fusion_params}")
        print()

    except Exception as e:
        print(f"✗ SearchConfig 测试失败: {e}")
        print()

async def main():
    """主测试函数。"""
    print("\n" + "="*60)
    print(" 模块化搜索策略测试套件")
    print("="*60)

    # 1. 测试模块导入
    if not await test_module_imports():
        print("\n模块导入失败，终止后续测试")
        return

    # 2. 分析各类策略的接口
    from hms_api.engine.search import (
        retrieval_registry,
        graph_retrieval_registry,
        fusion_registry,
        reranking_registry,
    )

    analyze_strategy_interface(retrieval_registry, "检索策略 (RetrievalStrategy)")
    analyze_strategy_interface(graph_retrieval_registry, "图检索策略 (GraphRetrievalStrategy)")
    analyze_strategy_interface(fusion_registry, "融合策略 (FusionStrategy)")
    analyze_strategy_interface(reranking_registry, "重排序策略 (RerankingStrategy)")

    # 3. 测试策略创建
    test_strategy_creation()

    # 4. 测试模块管理器
    test_module_manager()

    # 5. 测试策略覆盖
    test_strategy_override()

    # 6. 测试配置数据类
    test_config_dataclass()

    print_section("测试完成")
    print("所有测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
