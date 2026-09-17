"""
认证业务逻辑层

【为什么要有"服务层"，直接写在接口里不行吗】
直接写在接口里也能跑，但会有这些问题：

1. **重复代码**：注册功能可能被网页、小程序、后台脚本同时使用，
   每个入口都重写一遍"检查用户名是否存在→哈希密码→存库"，容易改漏。

2. **难测试**：接口函数绑定了 FastAPI 的请求对象，想单独测这段逻辑
   还得伪造一个 HTTP 请求。而服务层函数只接收普通参数，测试起来很轻松。

3. **职责清晰**：
     接口层  负责"接收 HTTP 请求、返回 HTTP 响应"
     服务层  负责"业务规则"（用户名不能重复、密码要加密）
   分开之后，换掉 Web 框架（比如从 FastAPI 换成别的）时，
   服务层的代码一行都不用改。

这就是"分层"的实际价值：让变化只影响一小块地方。
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AuthError, BadRequestError, ConflictError, NotFoundError
from app.core.logging import logger
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User, UserRole
from app.schemas.auth import UserRegister


class AuthService:
    """
    认证服务。

    说明：这里的方法都是 @staticmethod（静态方法），因为服务层本身不需要
    保存状态——它只接收参数、操作数据库、返回结果。
    调用的方式是 AuthService.register(db, data)，不需要先创建实例。
    """

    @staticmethod
    async def get_by_username(db: AsyncSession, username: str) -> Optional[User]:
        """按用户名查用户。查不到返回 None。"""
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """按 ID 查用户"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    # ==============================================================
    # 注册
    # ==============================================================
    @staticmethod
    async def register(db: AsyncSession, data: UserRegister) -> User:
        """
        注册新用户。

        业务规则：
          1. 用户名不能重复
          2. 密码用 bcrypt 哈希后再存
          3. 第一个注册的用户自动成为管理员？
             —— 不，本项目按需求固定由 admin 账号担任管理员，
                普通注册一律是 user 角色。这样权限模型清晰可控。

        ⚠️ 关于"竞态条件"：
        我们先查"用户名是否存在"，再插入。理论上存在极小概率：
        两个请求同时查到"不存在"，然后都去插入，导致重复。
        防住它的正确做法是**数据库层面的唯一约束**（users.username 上的
        unique=True），插入时数据库会直接拒绝第二个。
        所以即使这段检查代码有并发漏洞，数据库也能兜底。
        这就是"纵深防御"：不依赖单点保证。
        """
        # 用户名统一转小写，避免 "Admin" 和 "admin" 被当成两个账号
        username = data.username.strip()

        existing = await AuthService.get_by_username(db, username)
        if existing:
            raise ConflictError(f"用户名「{username}」已被占用，请换一个", code=40901)

        user = User(
            username=username,
            password_hash=hash_password(data.password),
            nickname=data.nickname or username,
            role=UserRole.USER,  # 新注册的一律是普通用户
            is_active=True,
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)  # 刷新一下，把数据库生成的 id、created_at 取回来

        logger.info(f"新用户注册成功 | id={user.id} username={user.username}")
        return user

    # ==============================================================
    # 登录
    # ==============================================================
    @staticmethod
    async def login(db: AsyncSession, username: str, password: str) -> tuple[User, str, int]:
        """
        登录验证。

        返回：(用户对象, 访问令牌, 令牌有效期秒数)

        【安全要点：错误提示要模糊】
        用户名不存在 → "用户名或密码错误"
        密码不对     → "用户名或密码错误"

        为什么不区分开？因为如果提示"该用户名不存在"，攻击者就能用它来
        **枚举系统里有哪些账号**（这叫用户名枚举攻击）。
        统一说"用户名或密码错误"，攻击者就无法判断是哪一项错了。
        这个小小的用词差异，是安全设计的重要细节。
        """
        user = await AuthService.get_by_username(db, username.strip())

        if user is None:
            # 注意：这里不提示"用户不存在"
            logger.warning(f"登录失败（用户名不存在）| username={username}")
            raise AuthError("用户名或密码错误", code=40110)

        if not verify_password(password, user.password_hash):
            logger.warning(f"登录失败（密码错误）| username={username}")
            raise AuthError("用户名或密码错误", code=40110)

        if not user.is_active:
            # 这个可以明确提示，因为此时用户身份已经确认了
            raise AuthError("账号已被禁用，请联系管理员", code=40111)

        # 更新最后登录时间
        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)

        # 签发令牌
        expires_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        token = create_access_token(
            subject=user.id,
            role=user.role,
            expires_minutes=expires_minutes,
        )

        logger.info(f"登录成功 | id={user.id} username={user.username} role={user.role}")
        return user, token, expires_minutes * 60

    # ==============================================================
    # 修改密码
    # ==============================================================
    @staticmethod
    async def change_password(
        db: AsyncSession, user: User, old_password: str, new_password: str
    ) -> None:
        """
        修改密码。

        业务规则：
          1. 必须验证旧密码（防止令牌被盗后直接改密码把主人锁在外面）
          2. 新密码不能和旧密码相同（否则等于没改）
        """
        if not verify_password(old_password, user.password_hash):
            logger.warning(f"修改密码失败（旧密码错误）| username={user.username}")
            raise AuthError("当前密码不正确", code=40120)

        if old_password == new_password:
            raise BadRequestError("新密码不能与当前密码相同", code=40010)

        user.password_hash = hash_password(new_password)
        await db.commit()

        logger.info(f"密码修改成功 | id={user.id} username={user.username}")

    # ==============================================================
    # 更新个人资料
    # ==============================================================
    @staticmethod
    async def update_profile(
        db: AsyncSession,
        user: User,
        nickname: Optional[str] = None,
        avatar: Optional[str] = None,
    ) -> User:
        """更新昵称和头像（只更新传了值的字段）"""
        if nickname is not None:
            user.nickname = nickname
        if avatar is not None:
            user.avatar = avatar

        await db.commit()
        await db.refresh(user)
        return user

    # ==============================================================
    # 管理员：用户管理
    # ==============================================================
    @staticmethod
    async def list_users(
        db: AsyncSession, page: int = 1, page_size: int = 20, keyword: Optional[str] = None
    ) -> tuple[list[User], int]:
        """
        用户列表（管理员用）。

        返回：(用户列表, 总条数)
        """
        stmt = select(User)
        count_stmt = select(func.count()).select_from(User)

        if keyword:
            # ilike 是"不区分大小写"的模糊匹配
            pattern = f"%{keyword}%"
            stmt = stmt.where(User.username.ilike(pattern))
            count_stmt = count_stmt.where(User.username.ilike(pattern))

        total = await db.scalar(count_stmt) or 0

        stmt = stmt.order_by(User.id.desc()).offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        users = list(result.scalars().all())

        return users, total

    @staticmethod
    async def set_active(db: AsyncSession, target_user_id: int, is_active: bool) -> User:
        """启用/禁用某个账号（管理员用）"""
        user = await AuthService.get_by_id(db, target_user_id)
        if user is None:
            raise NotFoundError("用户不存在", code=40410)

        if user.username == settings.ADMIN_USERNAME and not is_active:
            # 防止管理员把自己锁在门外，导致没人能管理系统
            raise BadRequestError("不能禁用初始管理员账号", code=40020)

        user.is_active = is_active
        await db.commit()
        await db.refresh(user)

        logger.info(f"账号状态变更 | target={user.username} is_active={is_active}")
        return user
