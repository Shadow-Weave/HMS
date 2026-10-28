"""HTTP service face: a FastAPI application exposing the Mem0-shaped endpoints plus the HMS-native surface.
HTTP 服务面：一个 FastAPI 应用，暴露 Mem0 形状的端点以及 HMS 原生接口。
"""

# Purpose (EN): Sub-package root of the optional `server` extra; exposes create_app and the console-script main.
# 目的（中）：可选 `server` extra 的子包根模块；暴露 create_app 与控制台脚本入口 main。

# Input (EN): Nothing at import time (fastapi/uvicorn are only required when create_app/main run).
# 输入（中）：导入时不接收任何输入（fastapi/uvicorn 只在 create_app/main 运行时才需要）。

# Output (EN): create_app, main.
# 输出（中）：create_app、main。

# State effect (EN): None; importing has no side effects (no network, no home-directory writes).
# 状态影响（中）：无；导入没有副作用（不联网、不写用户目录）。

from hms_elite.server.app import create_app, main

__all__ = ["create_app", "main"]
