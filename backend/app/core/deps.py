"""
依赖注入模块 —— 身份认证与权限校验

================================================================
【什么是"依赖注入"（Dependency Injection）】
================================================================
假设有 10 个接口都需要"先确认用户登录了"。笨办法是每个接口开头都写一遍：

    def api_1(token):
        user = 验令牌(token)     # 重复代码
        if not user: 报错        # 重复代码
        ...
    def api_2(token):
        user = 验令牌(token)     # 又重复一遍
        ...

依赖注入的做法是：把"验令牌"写成一个函数，然后在接口参数里"声明"我需要它：

    def api_1(user: User = Depends(get_current_user)):
        ...   # user 自动就有了，验令牌失败了根本进不来

FastAPI 会在调用你的接口之前，自动先执行 get_current_user，
把结果作为参数传进来；如果验证失败，接口压根不会被调用。

【好处】
  · 零重复代码
  · 不会漏——想给接口加权限，加一个参数就行，不可能忘记
  · 依赖关系一目了然，看函数签名就知道这个接口需要什么权限

================================================================
【认证 vs 鉴权 —— 这两个词容易混】
================================================================
  认证（Authentication）: 你是谁？   → get_current_user
  鉴权（Authorization） : 你能干什么？→ require_admin

  403 和 401 的区别：
    401 Unauthorized = "我不知道你是谁"（没登录 / 令牌过期）
    403 Forbidden    = "我知道你是谁，但你没这个权限"（普通用户想管知识库）
"""

from typing import Annotated, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import AuthError, PermissionDeniedError
from app.core.logging import logger
from app.core.security import decode_access_token
from app.models import User, UserRole

# ------------------------------------------------------------------
# HTTPBearer：告诉 FastAPI 和 Swagger 文档页"本接口需要一个 Bearer 令牌"。
# auto_error=False 表示：如果请求头里没带令牌，不要立刻报错，
# 而是把 None 传给我们自己的函数，由我们决定给出怎样友好的错误提示。
# ------------------------------------------------------------------
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    【认证】从请求里解析出"当前登录的是谁"。

    流程：
      1. 从请求头 Authorization: Bearer <令牌> 里取出令牌
      2. 验证令牌的真伪和有效期
      3. 用令牌里的用户 ID 去数据库查这个人的最新信息
      4. 检查账号有没有被禁用

    【第 3 步为什么必须查数据库，令牌里不是已经有用户信息了吗】
    因为令牌一旦签发就无法撤销，有效期 24 小时内一直有效。
    如果管理员在这期间禁用了某个账号，光看令牌是不知道的。
    每请求查一次数据库，才能保证"禁用立即生效"。
    （代价是每个请求多一次数据库查询，但 SQLite 本地查询是微秒级，完全值得。）
    """
    if credentials is None or not credentials.credentials:
        raise AuthError("请先登录", code=40101)

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        # 令牌无效或已过期。这两种情况对用户来说处理方式一样：重新登录
        raise AuthError("登录已过期，请重新登录", code=40102)

    user_id_raw = payload.get("sub")
    if not user_id_raw:
        # 令牌结构不对（可能是伪造的，或者签发时出了问题）
        raise AuthError("登录凭证无效，请重新登录", code=40103)

    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError):
        raise AuthError("登录凭证无效，请重新登录", code=40103) from None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        # 令牌是真的，但这个用户已经被删了
        raise AuthError("账号不存在，请重新登录", code=40104)

    if not user.is_active:
        # 管理员禁用了这个账号
        raise AuthError("账号已被禁用，请联系管理员", code=40105)

    # 把用户信息挂到 request 上，这样日志中间件也能拿到（记录"谁在访问什么"）
    request.state.user_id = user.id
    request.state.username = user.username

    return user


async def get_current_admin(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    【鉴权】要求必须是管理员。

    这是需求 6「只有管理员能打开知识库管理页面」的**后端强制实现**。

    ⚠️ 重点理解：
      前端隐藏菜单只是"体验优化"，懂技术的人用 Postman 直接调接口就绕过去了。
      真正拦住他的是这里——所有知识库管理接口都挂上这个依赖，
      普通用户调用时直接返回 403。

    用法：
        @router.post("/documents/upload")
        async def upload(admin: User = Depends(get_current_admin)):
            ...
    """
    if user.role != UserRole.ADMIN:
        logger.warning(
            f"越权访问被拦截 | 用户={user.username}(id={user.id}) "
            f"角色={user.role} 试图访问管理员接口"
        )
        raise PermissionDeniedError("此操作仅管理员可用", code=40301)

    return user


# ------------------------------------------------------------------
# 类型别名
#
# 用法对比：
#   老写法： user: User = Depends(get_current_user)
#   新写法： user: CurrentUser          ← 更简洁，而且编辑器提示更好
#
# Annotated 是 Python 的类型标注增强语法，可以给类型附加"元信息"，
# 这里的元信息就是"这个参数要通过 get_current_user 来获取"。
# ------------------------------------------------------------------
CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(get_current_admin)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
