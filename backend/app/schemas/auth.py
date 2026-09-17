"""
认证相关的数据结构 —— 注册、登录、改密码时前端发来的数据长什么样

【为什么密码要限制长度和复杂度】
  · 太短（少于 6 位）：容易被暴力猜解
  · 太长（超过 72 字节）：bcrypt 只处理前 72 字节，超出部分会被悄悄忽略，
    造成"我明明设了很长的密码，结果只认前 72 字节"的困惑，所以直接限制住
  · 太简单（纯数字）：用常见密码字典一撞就中

这里的校验只是"最低门槛"，是用户体验和安全的折中。
真正决定安全的是 bcrypt 加盐哈希——即使密码是 "123456"，
数据库里存的也是一串无法反推的乱码。
"""

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMBase

# ------------------------------------------------------------------
# 用户名规则：3~50 位，只能是字母、数字、下划线、连字符
#
# 为什么不允许中文和空格？
#   · 中文用户名在某些场景（URL、日志）容易出现编码问题
#   · 空格容易被"看起来像"的用户名混淆（"ad min" 和 "admin"）
#   · 限制字符集也能顺带防住一些注入攻击
# ------------------------------------------------------------------
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")


class UserRegister(BaseModel):
    """注册请求"""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="用户名：3~50 位，只能用字母、数字、下划线、连字符",
        examples=["zhangsan", "user_001"],
    )

    password: str = Field(
        ...,
        min_length=6,
        max_length=64,
        description="密码：6~64 位",
        examples=["myPassword123"],
    )

    nickname: Optional[str] = Field(
        default=None,
        max_length=50,
        description="昵称（可选，不填就用用户名）",
    )

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        v = v.strip()
        if not USERNAME_PATTERN.match(v):
            raise ValueError("用户名只能包含字母、数字、下划线、连字符，长度 3~50 位")
        return v

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str) -> str:
        # 注意：这里不做"必须含大小写+数字+符号"的强制要求。
        # 那种规则反而会逼用户写出 "Password1!" 这种好猜的密码，
        # 或者写在便利贴上。业界趋势是"长度优先，不强制复杂度"。
        if len(v.encode("utf-8")) > 72:
            raise ValueError("密码过长（超过 72 字节），请缩短一些")
        if v.strip() != v:
            raise ValueError("密码开头和结尾不能有空格")
        return v

    @field_validator("nickname")
    @classmethod
    def _check_nickname(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None  # 全是空格就当作没填


class UserLogin(BaseModel):
    """登录请求"""

    username: str = Field(..., min_length=1, max_length=50, description="用户名")
    password: str = Field(..., min_length=1, max_length=64, description="密码")


class TokenResponse(BaseModel):
    """登录成功后返回的令牌信息"""

    access_token: str = Field(..., description="访问令牌，前端要保存起来，之后每次请求都带上")
    token_type: str = Field(default="bearer", description="令牌类型，固定是 bearer")
    expires_in: int = Field(..., description="有效期（秒）")
    user: "UserOut" = Field(..., description="当前登录用户的信息")


class UserOut(ORMBase):
    """
    返回给前端的用户信息。

    ⚠️ 注意这里**故意没有** password_hash 字段。
    这是刻意的设计：响应模型决定了"哪些字段能被发出去"。
    只要不定义它，密码哈希就绝不可能通过这个接口泄露出去，
    哪怕以后有人不小心在代码里给对象赋了值也一样。
    这叫"白名单式输出"，比"黑名单式隐藏"安全得多。
    """

    id: int
    username: str
    nickname: Optional[str] = None
    role: str = Field(..., description="角色：admin 管理员 / user 普通用户")
    avatar: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class UserUpdate(BaseModel):
    """修改个人资料请求"""

    nickname: Optional[str] = Field(default=None, max_length=50, description="新昵称")
    avatar: Optional[str] = Field(default=None, max_length=255, description="头像地址")

    @field_validator("nickname")
    @classmethod
    def _strip_nickname(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ChangePasswordRequest(BaseModel):
    """
    修改密码请求。

    【为什么必须验证旧密码】
    因为令牌可能被盗用（比如用户在网吧忘了退出登录）。
    如果改密码不需要旧密码，别人拿到令牌就能直接改掉密码、
    把真正的用户锁在外面。要求输入旧密码，就堵住了这条路。
    """

    old_password: str = Field(..., min_length=1, max_length=64, description="当前密码")
    new_password: str = Field(..., min_length=6, max_length=64, description="新密码")

    @field_validator("new_password")
    @classmethod
    def _check_new_password(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("密码过长（超过 72 字节），请缩短一些")
        if v.strip() != v:
            raise ValueError("密码开头和结尾不能有空格")
        return v


# 解决 TokenResponse 里前向引用 "UserOut" 的问题。
# 因为 UserOut 定义在 TokenResponse 之后，Python 在解析时还不知道它是什么。
# 这一行告诉 Pydantic："去把 UserOut 找出来，补全类型信息"。
TokenResponse.model_rebuild()
