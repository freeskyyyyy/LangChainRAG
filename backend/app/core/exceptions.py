"""
统一异常处理模块

【为什么需要"统一"】
如果不做统一处理，同一个系统出错时会冒出五花八门的响应：
  · 有的返回 {"detail": "..."}
  · 有的返回 {"error": "...", "code": 500}
  · 有的直接吐一大段 Python 报错堆栈（还把服务器文件路径暴露给用户）
前端就得写一堆 if-else 去猜，非常痛苦。

统一之后，所有错误都是同一个格式：
    {
      "code": 40401,
      "message": "会话不存在",
      "detail": null
    }
前端只要认这一个格式就行，大大简化。

【还有两个安全好处】
1. 不把内部报错细节暴露给用户。数据库语句、文件路径、堆栈信息，
   这些泄露出去对攻击者很有用。统一处理时把它们记进日志，只给用户看人话。
2. 区分"业务错误"和"系统错误"。业务错误（比如密码错、没权限）应该是
   可预期的、给用户看的；系统错误（比如数据库连不上）要记日志报警。
"""

from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import logger


# ==================================================================
# 一、自定义业务异常
# ==================================================================
class AppException(Exception):
    """
    所有业务异常的父类。

    属性说明：
        code       —— 业务错误码（5 位数：前 3 位是 HTTP 状态码，后 2 位是细分序号）
        message    —— 给用户看的人话
        detail     —— 调试用的补充信息（可选，生产环境不返回给用户）
        status_code —— HTTP 状态码
    """

    def __init__(
        self,
        message: str,
        code: int = 40000,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        detail: Optional[Any] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class BadRequestError(AppException):
    """请求参数有问题（比如用户名格式不对）"""

    def __init__(self, message: str = "请求参数有误", code: int = 40000, detail=None):
        super().__init__(message, code, status.HTTP_400_BAD_REQUEST, detail)


class AuthError(AppException):
    """认证失败（没登录、令牌过期、密码错误）"""

    def __init__(self, message: str = "身份验证失败", code: int = 40100, detail=None):
        super().__init__(message, code, status.HTTP_401_UNAUTHORIZED, detail)


class PermissionDeniedError(AppException):
    """
    已登录但权限不够（比如普通用户想访问知识库管理接口）。
    注意和 AuthError 的区别：401 是"你是谁不知道"，403 是"知道你是谁，但你不够格"。
    """

    def __init__(self, message: str = "没有权限执行此操作", code: int = 40300, detail=None):
        super().__init__(message, code, status.HTTP_403_FORBIDDEN, detail)


class NotFoundError(AppException):
    """资源不存在（会话被删了、文档找不到）"""

    def __init__(self, message: str = "资源不存在", code: int = 40400, detail=None):
        super().__init__(message, code, status.HTTP_404_NOT_FOUND, detail)


class ConflictError(AppException):
    """冲突（注册时用户名已被占用）"""

    def __init__(self, message: str = "资源冲突", code: int = 40900, detail=None):
        super().__init__(message, code, status.HTTP_409_CONFLICT, detail)


class RateLimitError(AppException):
    """请求太频繁，被限流了"""

    def __init__(self, message: str = "操作太频繁，请稍后再试", code: int = 42900, detail=None):
        super().__init__(message, code, status.HTTP_429_TOO_MANY_REQUESTS, detail)


class LLMServiceError(AppException):
    """
    调用大模型服务出错（网络超时、额度用尽、模型名不对）。
    这类错误要单独归类，因为它们往往需要不同的处理（比如重试、降级提示）。
    """

    def __init__(self, message: str = "AI 服务暂时不可用，请稍后重试", code: int = 50300, detail=None):
        super().__init__(message, code, status.HTTP_503_SERVICE_UNAVAILABLE, detail)


# ==================================================================
# 二、注册到 FastAPI 的异常处理器
# ==================================================================
def _error_response(code: int, message: str, detail: Any = None, status_code: int = 400) -> JSONResponse:
    """生成统一格式的错误响应"""
    body: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """把所有异常处理器挂到 FastAPI 应用上。在 main.py 里调用一次。"""

    # ---------- 1. 我们自己定义的业务异常 ----------
    @app.exception_handler(AppException)
    async def _handle_app_exception(request: Request, exc: AppException):
        # 业务异常是"预期内"的，用 warning 级别记，不用 ERROR（否则日志会太吵）
        logger.warning(f"[业务异常] {request.method} {request.url.path} → {exc.code} {exc.message}")
        return _error_response(
            code=exc.code,
            message=exc.message,
            detail=exc.detail if exc.status_code < 500 else None,
            status_code=exc.status_code,
        )

    # ---------- 2. 请求参数校验失败（Pydantic 自动抛出）----------
    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        # 把 Pydantic 那串又长又难懂的英文报错，整理成人话
        problems = []
        for err in exc.errors():
            loc = " → ".join(str(x) for x in err.get("loc", []) if x != "body")
            problems.append(f"{loc}: {err.get('msg', '')}")

        logger.warning(f"[参数校验失败] {request.method} {request.url.path} → {problems}")
        return _error_response(
            code=40001,
            message="请求参数格式不正确",
            detail=problems,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    # ---------- 3. FastAPI/Starlette 自带的 HTTPException ----------
    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException):
        logger.warning(f"[HTTP异常] {request.method} {request.url.path} → {exc.status_code} {exc.detail}")
        return _error_response(
            code=exc.status_code * 100,
            message=str(exc.detail),
            status_code=exc.status_code,
        )

    # ---------- 4. 兜底：所有没被抓到的意外错误 ----------
    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception):
        # 这种是"预期外"的错误，必须用 ERROR 级别 + 完整堆栈记进日志，
        # 因为这意味着程序有 bug 或者环境出了问题，需要人去修。
        logger.exception(
            f"[未预期的异常] {request.method} {request.url.path} → {type(exc).__name__}: {exc}"
        )
        # 注意：这里**绝对不能**把 exc 的内容返回给用户，
        # 否则可能泄露数据库结构、文件路径、甚至 API Key。
        return _error_response(
            code=50000,
            message="服务器内部错误，请稍后重试",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
