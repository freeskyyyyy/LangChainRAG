"""
限流模块 —— 防止有人狂刷接口，把你的 API 额度烧光

================================================================
【为什么需要限流】
每次提问都要调用云端 API，那是要花钱的。
如果：
  · 有人手抖连点几十次发送按钮
  · 有人写了脚本循环调你的接口
  · 前端某个 bug 导致请求重发
后果就是额度被瞬间烧光，而且服务被拖慢。

限流就是给每个用户配一个"配额"：每分钟最多 N 次，超了就拒绝，让他等一会儿。

================================================================
【算法选择：令牌桶（Token Bucket）】
================================================================
想象每个用户手里有一个桶，桶里放着"令牌"（相当于门票）：

  · 桶的容量是 20 个令牌（= 每分钟最多 20 次）
  · 系统以固定速度往桶里放令牌：每 3 秒放 1 个
  · 每次请求要消耗 1 个令牌，没有令牌就拒绝（返回 429）
  · 桶满了就不再放（防止攒着一次性用掉）

【为什么用它，而不是"每分钟清零计数器"那种简单做法】
令牌桶允许"突发"：
  · 用户安静了 1 分钟 → 桶里攒满 20 个令牌
  · 突然连问 5 个问题 → 5 个令牌被消耗，全部放行，用户毫无感知
  · 而"计数器"方案会在这时直接拒绝，体验生硬

这更符合真实使用习惯：没人会匀速提问，都是"一阵一阵"的。

【对比其他方案】
| 方案 | 特点 | 本项目是否采用 |
|---|---|---|
| 固定窗口计数 | 每分钟清零。简单，但在窗口交界处可能瞬间放行 2 倍流量 | 否 |
| 滑动窗口 | 精确，但要记录每次请求的时间戳，内存开销大 | 否 |
| **令牌桶** | 允许突发、内存开销小、业界标准 | ✅ 采用 |
| 漏桶 | 强行匀速，不允许突发，适合保护下游 | 否 |

================================================================
【当前实现的局限（诚实说明）】
================================================================
令牌桶存在**程序内存**里。这意味着：
  · 程序重启后限流状态清空（影响不大，最多让某个人多问几次）
  · 如果将来开了多个进程/多台服务器，每个进程各算各的，
    实际放行量会变成 N 倍

将来上线时，把存储换成 Redis（用 Redis 的原子自增和过期时间实现），
就能多机共享。为此我把存储逻辑单独抽成了一个类，替换时不用改限流算法。
"""

import threading
import time
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.exceptions import RateLimitError
from app.core.logging import logger


@dataclass
class TokenBucket:
    """
    一个用户的令牌桶

    capacity   桶容量（最多能攒多少令牌）
    tokens     当前剩余令牌数
    refill_rate 每秒补充多少个令牌
    last_refill 上次补充令牌的时间戳
    """

    capacity: int
    refill_rate: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.monotonic)

    def _refill(self) -> None:
        """
        按流逝的时间补充令牌。

        【为什么用 time.monotonic() 而不是 time.time()】
        time.time() 是"墙上时钟"，可以被手动调整（同步网络时间、用户改系统时间）。
        如果有人把系统时间往回拨，令牌桶的计算就会出错。
        time.monotonic() 是"单调时钟"，只增不减，不受时间调整影响。
        凡是用来计算"经过了多久"，都应该用 monotonic。
        """
        now = time.monotonic()
        elapsed = now - self.last_refill

        if elapsed > 0:
            # 这段时间应该补多少令牌
            refill = elapsed * self.refill_rate
            # min 保证不会超过桶容量（攒不住多余的门票）
            self.tokens = min(self.capacity, self.tokens + refill)
            self.last_refill = now

    def consume(self, amount: float = 1.0) -> bool:
        """
        尝试消耗令牌。成功返回 True，不够返回 False。
        """
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    @property
    def wait_seconds(self) -> float:
        """还要等多久才能攒够 1 个令牌（用于告诉用户"请 X 秒后重试"）"""
        self._refill()
        if self.tokens >= 1:
            return 0.0
        return round((1 - self.tokens) / self.refill_rate, 1)


class RateLimiter:
    """
    限流器 —— 管理所有用户的令牌桶

    除了令牌桶本身，它还负责一件容易被忽略但很重要的事：
    **清理过期数据**。如果每个来访者都建一个桶且永不删除，
    用户越来越多时内存会被撑爆（这叫内存泄漏）。
    """

    def __init__(self, per_minute: int = 20, burst_multiplier: float = 1.0):
        self.per_minute = per_minute
        self.capacity = max(1, int(per_minute * burst_multiplier))
        self.refill_rate = per_minute / 60.0  # 每秒补充多少个

        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def _cleanup(self, now: float, interval: float = 300.0, idle: float = 600.0) -> None:
        """
        清理闲置超过 10 分钟的桶。

        为什么要做这一步：设想有人用脚本不断换 IP 来刷，
        每个 IP 都会产生一个桶。不清理的话内存会一直涨。
        桶的容量很小，重建成本几乎为零，所以大胆删。
        """
        if now - self._last_cleanup < interval:
            return  # 距离上次清理不到 5 分钟，先不折腾

        # 桶满 = 用户已经很久没来了（因为你没消耗它，它自己补满了）
        expired = [
            key
            for key, bucket in self._buckets.items()
            if bucket.tokens >= bucket.capacity
            and now - bucket.last_refill > idle
        ]
        for key in expired:
            self._buckets.pop(key, None)

        self._last_cleanup = now
        if expired:
            logger.debug(f"限流器清理了 {len(expired)} 个闲置令牌桶，当前剩余 {len(self._buckets)} 个")

    def check(self, key: str) -> tuple[bool, float]:
        """
        检查某个 key（通常用"用户ID"或"IP地址"）是否还有配额。

        返回：(是否放行, 需要等待的秒数)
        """
        now = time.monotonic()

        with self._lock:
            self._cleanup(now)

            bucket = self._buckets.get(key)
            if bucket is None:
                # 新用户：第一天就给他满桶，体验更好
                bucket = TokenBucket(
                    capacity=self.capacity,
                    refill_rate=self.refill_rate,
                    tokens=float(self.capacity),
                )
                self._buckets[key] = bucket

            allowed = bucket.consume(1.0)

            if not allowed:
                logger.warning(f"触发限流 | key={key} | 需等待 {bucket.wait_seconds} 秒")

            return allowed, bucket.wait_seconds

    def reset(self, key: str) -> None:
        """手动清除某个 key 的限流状态（管理接口测试用）"""
        with self._lock:
            self._buckets.pop(key, None)

    def stats(self) -> dict:
        """返回限流器状态，供监控接口查看"""
        with self._lock:
            return {
                "per_minute": self.per_minute,
                "bucket_capacity": self.capacity,
                "active_keys": len(self._buckets),
            }


# ==================================================================
# 全局限流器实例
# ==================================================================
_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    """获取全局限流器（单例）"""
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter(per_minute=settings.RATE_LIMIT_PER_MINUTE)
                logger.info(f"限流器已初始化 | 每用户每分钟 {settings.RATE_LIMIT_PER_MINUTE} 次")
    return _limiter


def check_rate_limit(key: str) -> None:
    """
    便捷函数：检查限流，超限直接抛异常。

    用法（在接口里调用）：
        check_rate_limit(f"user:{user.id}")
    """
    if not settings.RATE_LIMIT_ENABLED:
        return  # 配置里关了限流（本地调试时可以关掉，免得测起来碍手碍脚）

    allowed, wait = get_limiter().check(key)
    if not allowed:
        raise RateLimitError(
            message=f"操作太频繁了，请 {wait} 秒后再试",
            code=42901,
            detail={"retry_after": wait},
        )
