"""
认证接口 —— 注册、登录、登出、改密码、个人信息

【接口层的职责边界】
接口层只做三件事：
  1. 接收 HTTP 请求，把参数交给服务层
  2. 把服务层的结果包装成 HTTP 响应
  3. 处理 HTTP 层面的东西（状态码、响应头、限流）

**不写业务逻辑**——那不是它该管的事。这样的好处是：
业务规则改了，只需要改服务层；接口协议改了，只需要改接口层。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import CurrentUser
from app.core.logging import logger
from app.core.rate_limit import check_rate_limit
from app.schemas.auth import (
    ChangePasswordRequest,
    TokenResponse,
    UserLogin,
    UserOut,
    UserRegister,
    UserUpdate,
)
from app.schemas.common import SuccessResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="注册新用户",
    description="创建账号。用户名 3~50 位，只能用字母、数字、下划线、连字符。",
)
async def register(
    data: UserRegister,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserOut:
    """
    注册。

    注意返回的是 UserOut 而不是完整的 User 对象——
    UserOut 里**没有 password_hash 字段**，所以密码哈希不可能从这里泄露。
    """
    user = await AuthService.register(db, data)
    return UserOut.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="登录",
    description="登录成功返回访问令牌，前端需保存并在后续请求的 Authorization 头里携带。",
)
async def login(
    data: UserLogin,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    登录。

    【限流说明】
    登录接口是"撞库攻击"的重点目标——攻击者拿一批泄露的账号密码挨个试。
    所以我们在这里单独加一道更严的限流：按 **IP + 用户名** 限流，
    而不是按用户 ID（因为此时还没登录，没有用户 ID）。

    用 IP 限流能挡住"从同一个 IP 大量尝试"；
    用 用户名 限流能挡住"针对某个账号反复试密码"。
    两者组合，成本很低但效果显著。
    """
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"login:{client_ip}:{data.username}")

    user, token, expires_in = await AuthService.login(db, data.username, data.password)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserOut.model_validate(user),
    )


@router.post(
    "/logout",
    response_model=SuccessResponse,
    summary="登出",
)
async def logout(user: CurrentUser) -> SuccessResponse:
    """
    登出。

    【这里为什么什么也不做？】
    因为 JWT 是"无状态"的——令牌签发后在有效期内一直有效，
    服务器这边没有 session 可以销毁。

    那怎么真正登出？两个做法：
      ① 前端把本地存的令牌删掉（最简单，立即生效于该浏览器）
      ② 服务端维护一个"失效令牌黑名单"（能强制让令牌作废，但要额外存储）

    本项目采用 ①，因为：
      · 令牌有效期只有 24 小时，风险窗口有限
      · ② 需要引入 Redis 或数据库表，复杂度不值得

    这个接口保留着是为了：前端调用它时能统一走一遍"清理本地状态"的流程，
    并且服务端可以借机记录一条登出日志（审计用）。
    """
    logger.info(f"用户登出 | id={user.id} username={user.username}")
    return SuccessResponse(message="已退出登录")


@router.get(
    "/me",
    response_model=UserOut,
    summary="获取当前登录用户信息",
    description="前端刷新页面后，用它来恢复登录状态。",
)
async def get_me(user: CurrentUser) -> UserOut:
    """
    获取当前用户信息。

    前端典型用法：用户按 F5 刷新页面，本地还存着令牌，
    但内存里的用户信息没了。这时调这个接口，就知道"我是谁、我是什么角色"，
    进而决定侧边栏该显示哪些菜单。
    """
    return UserOut.model_validate(user)


@router.patch(
    "/me",
    response_model=UserOut,
    summary="修改个人资料",
)
async def update_me(
    data: UserUpdate,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserOut:
    """修改昵称和头像"""
    updated = await AuthService.update_profile(
        db, user, nickname=data.nickname, avatar=data.avatar
    )
    return UserOut.model_validate(updated)


@router.post(
    "/change-password",
    response_model=SuccessResponse,
    summary="修改密码",
    description="必须提供当前密码。修改成功后建议前端引导用户重新登录。",
)
async def change_password(
    data: ChangePasswordRequest,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SuccessResponse:
    """
    修改密码。

    前端在此之后应该清除本地令牌并跳转到登录页。
    因为旧令牌虽然技术上还有效（JWT 无状态的代价），
    但让用户重新登录一次，能确保"密码改了就一定是本人在用"。
    """
    await AuthService.change_password(db, user, data.old_password, data.new_password)
    return SuccessResponse(message="密码修改成功，请重新登录")
