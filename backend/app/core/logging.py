"""
日志模块 —— 记录程序运行情况，出问题时靠它排查

【为什么要有日志，print 不够用吗】
print 只能往屏幕上打，程序一关就没了，而且没办法按级别筛选。
日志能：
  · 写到文件里，程序关掉了记录还在
  · 按级别区分（调试信息 / 普通信息 / 警告 / 错误），出问题只看"错误"那一档
  · 自动按天切分文件，不会越来越臃肿
  · 记录时间、位置、耗时，便于定位

【为什么选 loguru 而不是 Python 自带的 logging】
Python 自带的 logging 配置繁琐（要写十几行 Handler、Formatter），
loguru 一行就能用好，而且彩色输出、自动切分、异常回溯都开箱即用。

【本项目的日志设计】
  · 控制台：只显示 INFO 及以上，彩色，方便开发时看
  · 文件 - 全量：所有日志按天切分，保留 30 天，压缩存档
  · 文件 - 错误：只记 ERROR 及以上，单独一个文件，出问题时直奔这里
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings

# 日志格式说明：
#   {time:...}   时间
#   {level:...}  级别（INFO / ERROR 等），左对齐补齐 8 位以保证对齐
#   {name}       哪个模块打的日志
#   {line}       哪一行
#   {message}    日志内容
CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{line} | {message}"
)


def setup_logging() -> None:
    """
    初始化日志系统。在程序启动时调用一次即可。

    注意：loguru 默认已经有一个输出到控制台的 handler，
    所以第一步要先把默认的删掉，换上我们自己配置的。
    """
    # 清空 loguru 的默认配置（否则会重复输出）
    logger.remove()

    log_dir: Path = settings.log_path
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 控制台输出 ----------
    logger.add(
        sys.stderr,
        format=CONSOLE_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
        # 过滤掉 uvicorn 那些每次请求都刷的噪音日志
        filter=lambda record: "uvicorn.access" not in record["name"],
    )

    # ---------- 2. 全量日志文件（按天切分）----------
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format=FILE_FORMAT,
        level=settings.LOG_LEVEL,
        rotation="00:00",       # 每天零点切一个新文件
        retention="30 days",    # 只保留最近 30 天
        compression="zip",      # 旧文件自动压缩，省空间
        encoding="utf-8",       # 必须指定，否则中文会乱码
        enqueue=True,           # 多进程/异步环境下安全写入
        backtrace=True,         # 出错时显示完整的调用链
        diagnose=False,         # 不显示变量值（防止密钥等敏感信息被写进日志）
    )

    # ---------- 3. 只记错误（排查问题直奔这个文件）----------
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        format=FILE_FORMAT,
        level="ERROR",
        rotation="00:00",
        retention="90 days",    # 错误日志留久一点
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.info(f"日志系统已启动 | 级别={settings.LOG_LEVEL} | 目录={log_dir}")


__all__ = ["logger", "setup_logging"]
