"""
安全模块 —— 密码加密 + 登录令牌（JWT）

================================================================
一、密码为什么要「加密」？为什么不能存明文？
================================================================
如果数据库里直接存 "123456"，那么：
  · 数据库管理员能看到所有用户的密码
  · 数据库一旦泄露（拖库），所有用户密码直接暴露
  · 很多人在不同网站用同一个密码，会连累他在别处的账号

所以我们要存的是「密码的指纹」，而不是密码本身。

【哈希函数是什么】
一种单向运算：密码 → 一长串乱码，容易；
但拿这串乱码 → 反推出密码，在数学上做不到。

【为什么要「加盐」（salt）】
如果不加盐，密码 "123456" 算出来的哈希永远是同一串。
攻击者可以预先算好几十亿个常见密码的哈希值做成表（叫彩虹表），
一对比就知道原始密码是什么。

加盐 = 给每个密码随机掺一段独有的"盐"再算哈希。
这样同一个 "123456"，两个用户算出来的哈希也完全不同，
彩虹表就失效了。

【bcrypt 的特殊之处】
它还故意算得很慢（约 0.1 秒），这让"暴力猜测"变得极其昂贵——
攻击者每秒只能试十几次，而不是几百万次。
用于登录验证时，0.1 秒你根本感觉不到；
但对攻击者来说，破解成本上升了几万倍。

================================================================
二、登录令牌（JWT）是什么？
================================================================
JWT（JSON Web Token）可以理解为一张「防伪门票」：

  登录成功 → 服务器发一张票（一串字符串）
           → 票上写着"用户ID=5，角色=admin，有效期到明天"
           → 并且盖了一个只有服务器知道的「防伪章」（用 SECRET_KEY 签名）

之后用户每次请求都带着这张票。服务器不需要查数据库，
只要验证防伪章是不是自己盖的、票有没有过期，就知道"你是谁"。

【好处】服务器不用记住谁登录了（叫"无状态"），
       所以哪怕以后开 10 台服务器，任何一台都能验证这张票。
【关键】SECRET_KEY 就是那个公章。它一旦泄露，别人就能伪造任意身份的票。
       所以它必须放在 .env 里，绝不能提交到代码库。
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
import jwt

from app.core.config import settings

# ==================================================================
# 一、密码哈希（bcrypt）
# ==================================================================


def hash_password(plain_password: str) -> str:
    """
    把明文密码变成哈希值（存入数据库的就是这个）。

    bcrypt 有一个限制：只处理前 72 个字节。
    超长密码会被截断，可能造成"两个不同密码算出相同哈希"。
    所以这里先自己截断到 72 字节，避免意外。

    返回示例：'$2b$12$eImiTXuWVxfM37uY4JANjQ...'（60 个字符）
    注意：同一个密码每次调用得到的哈希都不同（因为盐是随机的），这是正常的。
    """
    pwd_bytes = plain_password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)  # rounds 越高越安全也越慢，12 是业界常用值
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """
    校验密码是否正确。

    做法：把用户输入的密码用「同一个盐」重新算一遍哈希，
         看结果是否和数据库里存的完全一致。
         因为盐本身就存在哈希串里面，bcrypt 能自动提取出来。

    返回 True = 密码正确，False = 密码错误。
    """
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(pwd_bytes, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # 哈希串格式不对（比如数据库里的数据被破坏了）
        return False


# ==================================================================
# 二、登录令牌（JWT）
# ==================================================================


def create_access_token(
    subject: str | int,
    role: str,
    expires_minutes: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """
    签发一张登录令牌。

    参数：
        subject   —— 用户 ID（写进令牌的 "sub" 字段，标识"这是谁"）
        role      —— 角色（admin / user），前端据此显示菜单，后端据此鉴权
        expires_minutes —— 有效期（分钟），不传就用配置里的默认值
        extra     —— 想额外塞进令牌的信息

    返回：一串形如 "eyJhbGciOiJIUzI1NiIs..." 的字符串，交给前端保存。

    【令牌里到底装了什么】
        主要是三段：头部（用什么算法签名）+ 载荷（用户信息）+ 签名（防伪章）
    注意：载荷部分只是 Base64 编码，不是加密！任何人都能解开看内容。
    所以绝对不能往里放密码、手机号之类的敏感信息。
    它的安全性来自"改不了"（一改签名就对不上），而不是"看不见"。
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload: dict[str, Any] = {
        "sub": str(subject),   # 用户 ID（JWT 规范要求 sub 必须是字符串）
        "role": role,          # 角色
        "iat": now,            # 签发时间（issued at）
        "exp": expire,         # 过期时间（expiration）—— 服务器会自动校验
    }
    if extra:
        payload.update(extra)

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[dict[str, Any]]:
    """
    验证并解析令牌。

    返回值：
        验证通过 → 令牌里的信息（字典），例如 {"sub": "5", "role": "admin", ...}
        验证失败 → None

    什么情况下会失败？
        · 令牌被篡改过（签名对不上）
        · 令牌已过期
        · 令牌是用别的密钥签的（比如伪造的）
    """
    try:
        return jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        # 令牌过期了 —— 前端收到这个应该跳转登录页
        return None
    except jwt.PyJWTError:
        # 签名错误、格式错误等一切其他问题
        return None
