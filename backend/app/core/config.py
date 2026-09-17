"""
配置加载模块 —— 整个程序的「总控台」

【这个文件干什么的】
把 .env 文件里那些"可调参数"读进程序，供所有模块使用。

【为什么要单独搞一个文件】
如果参数散落在各个代码文件里，想改个"检索返回几条"就得满项目找。
集中在这里之后，所有可调项一览无余，改配置不用碰业务代码。

【怎么用】
    from app.core.config import settings
    print(settings.LLM_MODEL)      # 直接点出来

【关于 @lru_cache】
它会把这个配置对象缓存起来。整个程序运行期间只读一次 .env 文件，
不会每用一次就读一遍磁盘。
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# ------------------------------------------------------------------
# 路径基准
#   __file__          = .../backend/app/core/config.py
#   .parent           = .../backend/app/core/
#   .parent.parent    = .../backend/app/
#   .parent.parent.parent = .../backend/
# ------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = DATA_DIR / "logs"
SAMPLES_DIR = DATA_DIR / "samples"


class Settings(BaseSettings):
    """所有配置项的定义。字段名必须和 .env 里的键名一致（大小写不敏感）。"""

    # pydantic-settings 的配置：告诉它去哪个文件读配置
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env 里有多余的键也不报错
    )

    # ---------------- 应用基础 ----------------
    APP_NAME: str = "企业级RAG知识库问答系统"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # ------------------------------------------------------------------
    # 热重载开关 —— 默认关闭，这是踩了三次坑之后的决定
    # ------------------------------------------------------------------
    # 【它是什么】
    # 打开后，改一行后端代码、保存，服务器自己重启，不用手动关了再开。
    #
    # 【为什么默认关掉】
    # uvicorn 的"热重载"不是简单重启，而是**开一棵三层进程树**：
    #
    #     监工进程（看文件变没变）
    #        └─ 派生进程
    #             └─ 干活的进程 ← 真正占着 8000 端口的这个
    #
    # 在 Windows 上，杀掉上面两层之后，最底下那个**经常杀不干净**。
    # 它会带着 8000 端口继续活着，于是出现最坑的局面：
    #
    #     你以为重启了 → 实际还是旧代码在应答 → 改了半天的东西"没生效"
    #
    # 而且这个故障**完全不报错**：接口正常返回 200，只有结果是旧的。
    # 本项目已经因此栽过三次（问题记录 #17、#20，以及一次"重新生成"功能
    # 怎么测都不生效）。每次都要花十几分钟去查，代价远大于省下的那点重启时间。
    #
    # 【而且它省下的时间本来就是假的】
    # 实测中 WatchFiles 还出现过"静静地看着文件变了、却不重启"的情况，
    # 所以本项目的既定规矩是"改完后端代码一律手动重启"——
    # 既然如此，热重载就是个既没用又危险的摆设。
    #
    # 【如果你还是想要它】
    # 在 backend/.env 里加一行：RELOAD=true
    # 但请记住：重启时一定要确认 8000 端口真的空了（netstat -ano | findstr :8000）
    RELOAD: bool = False

    # ---------------- 安全 ----------------
    # JWT 签名密钥：登录令牌的「防伪印章」，泄露了别人就能伪造登录态
    SECRET_KEY: str = "change-me-in-dotenv"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 默认 24 小时

    # ---------------- 阿里云百炼 ----------------
    DASHSCOPE_API_KEY: str = ""
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.3  # 越低越严谨，越高越发散；问答场景要低
    EMBEDDING_MODEL: str = "text-embedding-v4"
    RERANK_MODEL: str = "gte-rerank-v2"

    # ---------------- 数据库 ----------------
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/app.db"

    # ---------------- 向量库 ----------------
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "ec_product_kb"

    # ---------------- RAG 检索参数 ----------------
    # ------------------------------------------------------------------
    # 切片长度：为什么从 500 改成了 250
    # ------------------------------------------------------------------
    # 【实测发现的问题】
    # 用真实资料测试时发现：问「真丝连衣裙怎么洗」，系统找到了对的**文件**，
    # 但引用卡片上显示的是讲**冰丝**的那一段，不是真丝。
    #
    # 【原因】
    # 那份洗涤养护资料里，每种面料只占 100~200 字：
    #     1. 纯棉材质（约 150 字）
    #     2. 薄牛仔材质（约 150 字）
    #     ...
    # 但切片长度设的是 500 字，一刀下去，一片里塞进了两三个面料。
    # 向量化之后，这一片的语义是"各种面料混在一起"，
    # 跟"雪纺"的匹配度自然就低了。
    #
    # 【打个比方】
    # 你要在书里找"红烧肉怎么做"，但书是按每 5 页撕成一张纸的，
    # 你拿到的那张上既印着红烧肉、又印着糖醋排骨和清蒸鱼 ——
    # 找是能找到，但得翻半天才知道哪一行才是你要的。
    #
    # 【为什么改小是安全的】
    # 因为这是**父子文档**结构：
    #     检索用**子块**（薄的，250 字）→ 定位准
    #     喂给模型的是**父块**（厚的，1400 字）→ 上下文全
    # 子块改小**不会让大模型看到的上下文变少**，只会让定位更准。
    #
    # 【重叠为什么跟着一起改】
    # 重叠的用意是"防止关键信息正好被切断"，它应该跟切片长度成比例。
    # 原来是 80/500 = 16%，所以现在取 40/250 = 16%，比例不变。
    # 单独改一个容易说不清到底是哪个参数起了作用。
    CHUNK_SIZE: int = 250
    CHUNK_OVERLAP: int = 40

    # ------------------------------------------------------------------
    # 结构化切分：按文档自己的编号切，而不是按字数硬切
    # ------------------------------------------------------------------
    # 上面的 250 字只是"兜底" —— 真正的主力是这条。
    #
    # 【它做什么】
    # 识别文档里的编号小标题（"一、" "1." "（1）" "#"），
    # 在编号前面**强制断一刀**，保证一个小节不会跟别的小节混在一起。
    #
    # 【改造前后的区别，用你的洗涤养护资料举例】
    #   改造前：一片 470 字 = 冰丝 + 雪纺 + 麂皮绒（三种面料混在一起）
    #   改造后：一片 150 字 = 雪纺（干干净净就是它）
    #
    # 【什么样的文档它管不了】
    # 写成一大段、没有任何编号的散文式文档（比如小说、新闻稿）。
    # 那种文档识别不出结构，会自动退回按字数切 —— 也就是 250 字那档兜底。
    STRUCTURE_AWARE_SPLIT: bool = True

    # 小节合并阈值：小于这个字数的小节会被并进下一节。
    # 【为什么需要它】
    # 上面那份资料里有一行是"一、春季服装（纯棉、薄牛仔、针织棉、轻薄化纤）"，
    # 它自己只有 25 个字，如果单独成片，检索时毫无用处（没有内容可匹配）。
    # 所以让它跟后面第一节合并。
    MIN_SECTION_CHARS: int = 80

    PARENT_CHUNK_SIZE: int = 1400
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_N: int = 5
    MAX_CONTEXT_CHUNKS: int = 5

    # ------------------------------------------------------------------
    # 相关性阈值 —— 为什么需要两个？
    #
    # 【踩过的坑】
    # 我一开始只设了一个阈值 0.3，凭"向量相似度的直觉"定的。
    # 结果实测发现：重排序模型的分数量纲完全不同！
    #
    # 实测数据（12 个有答案的问题 + 6 个无答案的问题）：
    #   重排序分数：有答案 0.167~0.320  |  无答案 0.006~0.160  ← 分界清晰
    #   向量相似度：有答案 0.582~2.407  |  无答案 0.265~1.360  ← 严重重叠
    #
    # 如果拿 0.3 去卡重排序分数，正确答案（0.167）会被当成"不相关"拦掉，
    # 系统就会一直在知识库明明有答案的时候回答"没有找到"。
    #
    # 所以两个阈值必须分开设，各按各自的量纲来定。
    # ------------------------------------------------------------------
    RERANK_SCORE_THRESHOLD: float = 0.15   # 重排序分数阈值（实测分界线在 0.163）
    SCORE_THRESHOLD: float = 0.35          # 向量相似度阈值（用于重排序关闭时的降级路径）

    # ---------------- 缓存 ----------------
    CACHE_BACKEND: str = "memory"  # memory | redis
    SEMANTIC_CACHE_THRESHOLD: float = 0.95
    SEMANTIC_CACHE_TTL: int = 3600
    RETRIEVAL_CACHE_TTL: int = 600

    # ---------------- 限流 ----------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 20

    # ---------------- 文件上传 ----------------
    MAX_UPLOAD_SIZE_MB: int = 20

    # 【注意 Annotated[..., NoDecode] 这个写法】
    # pydantic-settings 默认会把「列表类型」的配置项当成 JSON 来解析。
    # 但我们在 .env 里写的是 .pdf,.docx,.md 这种逗号分隔的字符串，不是合法 JSON，
    # 它就会直接报错，导致下面那个加工函数根本没机会执行。
    # NoDecode 标记的作用就是告诉它：「这个字段你别管，交给我的加工函数处理」。
    ALLOWED_EXTENSIONS: Annotated[List[str], NoDecode] = [
        ".pdf", ".docx", ".doc", ".xlsx", ".xls",
        ".csv", ".md", ".txt", ".html", ".htm",
    ]

    # ---------------- 初始管理员 ----------------
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "123456"

    # ---------------- 日志 ----------------
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./data/logs"

    # ==============================================================
    # 校验器：把 .env 里读进来的字符串加工成程序需要的格式
    # ==============================================================

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def _parse_extensions(cls, v):
        """
        把 ".pdf,.docx,.md" 这样的字符串切成列表 [".pdf", ".docx", ".md"]。
        统一转成小写，并且保证每个都以点开头，防止配置时手滑写错。
        """
        if isinstance(v, str):
            items = [x.strip().lower() for x in v.split(",") if x.strip()]
            return [x if x.startswith(".") else f".{x}" for x in items]
        return v

    @field_validator("SECRET_KEY")
    @classmethod
    def _warn_default_secret(cls, v: str) -> str:
        """防止部署时忘了改默认密钥（这里只做提醒，不阻断启动）。"""
        if v in ("change-me-in-dotenv", "", None):
            raise ValueError(
                "SECRET_KEY 未配置！请在 backend/.env 中设置一个随机字符串。\n"
                '生成方法： .venv/Scripts/python.exe -c "import secrets;print(secrets.token_urlsafe(48))"'
            )
        return v

    # ==============================================================
    # 便捷属性：把路径统一转成绝对路径，避免"工作目录不同就找不到文件"
    # ==============================================================

    @property
    def upload_path(self) -> Path:
        """上传文档的存放目录（绝对路径）"""
        return UPLOAD_DIR

    @property
    def chroma_path(self) -> Path:
        """向量库的持久化目录（绝对路径）"""
        return CHROMA_DIR

    @property
    def log_path(self) -> Path:
        """日志目录（绝对路径）"""
        return LOG_DIR

    @property
    def samples_path(self) -> Path:
        """示例商品资料目录（绝对路径）"""
        return SAMPLES_DIR

    @property
    def max_upload_bytes(self) -> int:
        """上传大小上限（字节），供上传接口校验"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def ensure_dirs(self) -> None:
        """确保所有需要的数据目录都存在（不存在就创建）。程序启动时调用一次。"""
        for p in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR, LOG_DIR, SAMPLES_DIR):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """获取配置对象（带缓存，整个程序只创建一次）"""
    return Settings()


# 全局配置实例，其他模块直接 `from app.core.config import settings` 即可
settings = get_settings()
