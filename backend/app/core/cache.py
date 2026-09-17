"""
缓存模块 —— 把算过的东西记在小本子上，下次直接翻本子

【为什么需要缓存】
每次提问都要调用云端 API，这些都是要花钱、要花时间的：
  · 向量化一次问题       → 约 0.2 秒 + 少量费用
  · 检索一次知识库       → 约 0.3 秒
  · 让模型生成一次回答   → 约 2~5 秒 + 大头费用

如果 10 个人问"你们支持退货吗"，同样的活儿干 10 遍就是纯浪费。
缓存就是把第一次的结果存起来，后面 9 次直接返回。

【本模块的设计目标：可替换】
我们定义了一个「缓存接口」（CacheBackend），规定了缓存必须支持哪些操作
（读、写、删、清空）。然后提供两套实现：
  · MemoryCacheBackend —— 存在程序内存里，零安装（当前使用）
  · RedisCacheBackend  —— 存在 Redis 服务里，多机共享（预留，将来上服务器时用）

上层业务代码只跟「接口」打交道，不关心底层用的是哪套。
将来从内存切到 Redis，只需要改 .env 里的一行配置，业务代码一行都不用动。
这就是"面向接口编程"的价值。
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from cachetools import TTLCache

from app.core.config import settings
from app.core.logging import logger


# ==================================================================
# 一、缓存接口（抽象基类）
# ==================================================================
class CacheBackend(ABC):
    """
    缓存后端的「合同」。任何缓存实现都必须提供这四个方法。

    为什么用 ABC（抽象基类）而不是直接写个普通类？
    因为 ABC 会强制子类必须实现所有抽象方法——如果哪天你写了新实现
    却忘了实现 delete()，程序启动时就会直接报错，而不是等到运行时才发现。
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """读缓存。命中返回存的值，没命中返回 None。"""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """写缓存。ttl 是存活秒数，不传则用默认值。"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """删掉某一条缓存。"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """清空所有缓存。"""
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """返回缓存的统计信息（条数、命中率等），用于监控和调试。"""
        raise NotImplementedError


# ==================================================================
# 二、内存缓存实现（当前使用）
# ==================================================================
class MemoryCacheBackend(CacheBackend):
    """
    把缓存存在程序自己的内存里。

    【优点】
      · 零安装，程序自带
      · 速度极快（内存读取，微秒级）
    【缺点】
      · 程序重启后缓存全丢（对本项目影响不大，重新算一遍就行）
      · 多进程/多台服务器之间不共享

    【关于 maxsize】
      我们设置了最大条数上限。超过之后会自动淘汰最久没被用到的那些（LRU 策略）。
      为什么必须设上限？因为如果不设，有人恶意发大量不同的问题，
      内存会被撑爆，程序崩溃。这是"防止内存泄漏"的基本功。
    """

    def __init__(self, maxsize: int = 1000, default_ttl: int = 3600):
        # TTLCache = 带"过期自动失效"的缓存。第二个参数是默认存活秒数。
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=default_ttl)
        self._default_ttl = default_ttl

        # 【为什么要加锁】
        # FastAPI 是异步的，多个请求可能同时读写缓存。
        # 如果两个请求同时往字典里塞东西，可能导致数据错乱（竞态条件）。
        # 加一把锁，保证同一时刻只有一个请求在操作缓存。
        # 用 RLock（可重入锁）是因为 set 内部可能调用其他方法，避免自己把自己锁死。
        self._lock = threading.RLock()

        # 统计命中率用（能直观看出"缓存到底省了多少活"）
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            try:
                raw = self._cache[key]
            except KeyError:
                # 没命中是正常情况，不是错误
                self._misses += 1
                return None

            # 取出来的可能是带过期时间的包装对象，需要解包
            value = self._unwrap(raw)

            if value is None:
                # 已经过期了。顺手清掉，避免它继续占内存。
                self._cache.pop(key, None)
                self._misses += 1
                return None

            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        写缓存。

        注意：TTLCache 的过期时间是在创建时统一设定的，
        单条记录没法单独指定 TTL。所以当传入的 ttl 和默认值不同时，
        我们用一个「时间戳包裹」的技巧来实现单条过期：
        存的时候连带记下"什么时候过期"，读的时候检查一下。
        """
        with self._lock:
            if ttl is None or ttl == self._default_ttl:
                self._cache[key] = value
            else:
                # 包一层，记下过期时间点
                self._cache[key] = _ExpiringValue(value=value, expires_at=time.time() + ttl)

    def _unwrap(self, raw: Any) -> Optional[Any]:
        """如果取出来的是带过期时间的包装对象，检查是否已过期。"""
        if isinstance(raw, _ExpiringValue):
            if time.time() > raw.expires_at:
                return None  # 已过期，当作没命中
            return raw.value
        return raw

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total else 0.0
            return {
                "backend": "memory",
                "items": len(self._cache),
                "maxsize": self._cache.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
            }


class _ExpiringValue:
    """
    内部用的「带过期时间的数据包裹」。

    为什么要包一层？因为 cachetools 的 TTLCache 只能给整个缓存设统一的过期时间，
    不能给单条记录设不同的过期时间。而我们确实有这个需求：
    比如"语义缓存"要存 1 小时，"检索结果缓存"只存 10 分钟。

    包一层记下各自的过期时间点，就实现了「单条独立过期」。
    """

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float):
        self.value = value
        self.expires_at = expires_at


# ==================================================================
# 三、Redis 缓存实现（预留，将来部署到服务器时启用）
# ==================================================================
class RedisCacheBackend(CacheBackend):
    """
    Redis 缓存实现。

    【什么时候用它】
    将来把项目部署到云服务器、并且需要多台服务器共享缓存时。

    【为什么现在不启用】
    Redis 官方不出 Windows 版，本地没有 Docker 装起来很折腾。
    但代码先写好，将来改 .env 里一行配置就能启用，不用重写业务代码。

    【启用方法】
      1. 安装：pip install redis
      2. 在 .env 里把 CACHE_BACKEND 改成 redis，并配上 REDIS_URL
    """

    def __init__(self, url: str = "redis://localhost:6379/0"):
        try:
            import redis  # 延迟导入：没装这个包也不影响程序启动
        except ImportError as e:
            raise RuntimeError(
                "启用了 Redis 缓存，但没装 redis 包。请执行：pip install redis"
            ) from e

        self._client = redis.Redis.from_url(url, decode_responses=False)
        self._client.ping()  # 连不上就立刻报错，不要等到用的时候才发现
        logger.info(f"Redis 缓存已连接: {url}")

    def get(self, key: str) -> Optional[Any]:
        import pickle

        raw = self._client.get(key)
        return pickle.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        import pickle

        self._client.set(key, pickle.dumps(value), ex=ttl or 3600)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def clear(self) -> None:
        self._client.flushdb()

    def stats(self) -> dict[str, Any]:
        info = self._client.info()
        return {
            "backend": "redis",
            "items": self._client.dbsize(),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
        }


# ==================================================================
# 四、缓存工厂：按配置选择实现
# ==================================================================
_cache_instance: Optional[CacheBackend] = None
_cache_lock = threading.Lock()


def get_cache() -> CacheBackend:
    """
    获取全局缓存实例（单例模式）。

    【什么是单例】整个程序只创建一个缓存对象，大家共用同一个。
    不能每次调用都 new 一个，否则每个请求都有自己独立的缓存，等于没缓存。
    """
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            # 双重检查：防止多个请求同时进来时创建出多个实例
            if _cache_instance is None:
                backend = settings.CACHE_BACKEND.lower()
                if backend == "redis":
                    _cache_instance = RedisCacheBackend()
                else:
                    _cache_instance = MemoryCacheBackend(
                        maxsize=1000,
                        default_ttl=settings.SEMANTIC_CACHE_TTL,
                    )
                logger.info(f"缓存后端已初始化: {backend}")
    return _cache_instance


# ==================================================================
# 五、提供给业务层使用的便捷函数
# ==================================================================
def cache_key(*parts: Any) -> str:
    """
    拼装缓存键。

    【为什么要统一拼装】
    直接用一个字符串当键容易撞车。比如用户 ID 是 "1"，会话 ID 是 "23"，
    拼成 "123" 就可能和"用户 12、会话 3"撞上。
    用分隔符隔开就能避免： "1:23" 和 "12:3" 就不会混。

    用法：cache_key("retrieval", user_id, question)  →  "retrieval:5:退货吗"
    """
    return ":".join(str(p) for p in parts)


def get_cached(key: str) -> Optional[Any]:
    """读缓存（便捷函数，省得每次都 get_cache().get()）"""
    return get_cache().get(key)


def set_cached(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """写缓存（便捷函数）"""
    get_cache().set(key, value, ttl)
