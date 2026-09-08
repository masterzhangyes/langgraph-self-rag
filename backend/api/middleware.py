"""
中间件模块 - 请求日志 / API 认证 / 异常处理
"""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings

logger = logging.getLogger("qa.middleware")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """请求日志中间件 - 记录每个请求的方法、路径、耗时和状态码"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"({duration:.0f}ms)"
        )
        return response


class APIAuthMiddleware(BaseHTTPMiddleware):
    """API 认证中间件 - 验证 X-API-Key 请求头（可通过配置开关控制）"""

    # 不需要认证的路径前缀
    PUBLIC_PATHS = [
        "/api/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 如果未启用认证，直接放行
        if not settings.API_AUTH_ENABLED:
            return await call_next(request)

        # 健康检查和文档路径跳过认证
        path = request.url.path
        # CORS 预检请求（OPTIONS）不含自定义头，必须放行
        if request.method == "OPTIONS":
            return await call_next(request)
        for prefix in self.PUBLIC_PATHS:
            if path.startswith(prefix):
                return await call_next(request)

        # 认证校验
        api_key = request.headers.get(settings.API_AUTH_HEADER, "")
        if api_key != settings.SECRET_KEY:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "未授权访问，请提供有效的 API Key",
                    "error_code": "UNAUTHORIZED",
                },
            )

        return await call_next(request)


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """全局异常捕获中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except ValueError as e:
            logger.warning(f"ValueError: {e} for {request.url.path}")
            return JSONResponse(
                status_code=400,
                content={"detail": str(e), "error_code": "INVALID_INPUT"},
            )
        except Exception as e:
            logger.exception(f"Unhandled error on {request.url.path}: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "服务器内部错误，请稍后重试", "error_code": "INTERNAL_ERROR"},
            )


def setup_logging(log_level: str = "INFO", log_path: str = "./logs/app.log"):
    """配置日志"""
    import os

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger.info("Logging initialized")
