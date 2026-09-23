"""
SQLite 数据库层 - 对话历史持久化
使用 aiosqlite + 信号量连接池实现真正的异步操作
"""
import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
import aiosqlite

from core.config import settings


def get_db_path() -> str:
    """获取数据库路径，确保目录存在"""
    db_dir = os.path.dirname(settings.DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    return settings.DB_PATH


# ──────── 改进版连接池 ────────

_POOL_SIZE = 10
_pool: List[aiosqlite.Connection] = []
_pool_semaphore = asyncio.Semaphore(_POOL_SIZE)  # 限制最大并发连接数
_pool_lock = asyncio.Lock()


async def _get_conn() -> aiosqlite.Connection:
    """从连接池获取连接（带超时和并发控制）"""
    await _pool_semaphore.acquire()
    try:
        return await _get_conn_inner()
    except Exception:
        _pool_semaphore.release()
        raise


async def _get_conn_inner() -> aiosqlite.Connection:
    """内部获取连接逻辑"""
    async with _pool_lock:
        # 尝试复用池中连接
        while _pool:
            conn = _pool.pop()
            try:
                await conn.execute("SELECT 1")
                conn.row_factory = aiosqlite.Row
                return conn
            except Exception:
                try:
                    await conn.close()
                except Exception:
                    pass

    # 创建新连接
    conn = await aiosqlite.connect(get_db_path())
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute("PRAGMA busy_timeout=5000")
    return conn


async def _release_conn(conn: aiosqlite.Connection):
    """归还连接到池"""
    try:
        async with _pool_lock:
            if len(_pool) < _POOL_SIZE:
                await conn.execute("PRAGMA optimize")
                _pool.append(conn)
            else:
                await conn.close()
    finally:
        _pool_semaphore.release()


async def _close_pool():
    """关闭所有连接"""
    async with _pool_lock:
        for conn in _pool:
            try:
                await conn.close()
            except Exception:
                pass
        _pool.clear()


# ──────── 数据库初始化 ────────

async def init_db():
    """初始化数据库表"""
    conn = await aiosqlite.connect(get_db_path())
    try:
        await conn.executescript("""
            -- 用户表（v2.2 新增）
            -- 密码只存 bcrypt 哈希，绝不存明文；
            -- role 实现两级 RBAC（user / admin）
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'admin')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            );

            -- 会话表（user_id 归属用户，NULL 表示匿名会话）
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '新对话',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
            );

            -- 消息表
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );

            -- 审计日志表（v2.2 新增）
            -- 记录登录 / 注册 / 管理员操作，供管理后台回溯
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                action TEXT NOT NULL,
                detail TEXT,
                ip TEXT,
                created_at TEXT NOT NULL
            );

            -- 知识库注册表（v2.3 新增）
            -- 实现按用户隔离的知识库: owner_id 为 NULL 表示公共库
            -- （所有人可读、仅管理员可写）
            -- physical_name 为磁盘目录名（chroma_{physical_name}），与逻辑名解耦:
            --   私有库 "u{owner_id}__{name}"、公共库 "pub__{name}"、旧库沿用原名
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                physical_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- 索引
            CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON chat_messages(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated 
                ON chat_sessions(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_created 
                ON audit_logs(created_at DESC);
            -- （owner, name) 唯一: SQLite 中 NULL 互不相等，用 ifnull 归一化后建表达式索引
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_owner_name
                ON knowledge_bases(ifnull(owner_id, -1), name);
        """)
        await conn.commit()

        # ── 幂等迁移：为旧库的 chat_sessions 补充 user_id 归属列 ──
        # 旧数据 user_id 为 NULL → 归入「匿名会话」桶，不破坏向后兼容
        cursor = await conn.execute("PRAGMA table_info(chat_sessions)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "user_id" not in columns:
            await conn.execute(
                "ALTER TABLE chat_sessions ADD COLUMN user_id INTEGER "
                "REFERENCES users(id) ON DELETE CASCADE"
            )
            await conn.commit()

        # 会话归属索引（迁移完成后创建，保证旧库也能建上）
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user "
            "ON chat_sessions(user_id, updated_at DESC)"
        )
        await conn.commit()

        # ── 幂等迁移：把磁盘上已存在的旧知识库目录注册为公共知识库 ──
        # v2.3 之前知识库无归属概念；升级后统一归入「公共库」桶:
        # 所有人可读（保持现有检索行为），仅管理员可写。
        # INSERT OR IGNORE 依赖唯一索引，重复执行无副作用。
        kb_base = os.path.dirname(settings.VECTOR_DB_PATH) or "."
        if os.path.isdir(kb_base):
            now = datetime.now().isoformat()
            for item in os.listdir(kb_base):
                kb_dir = os.path.join(kb_base, item)
                if not (item.startswith("chroma_") and os.path.isdir(kb_dir)):
                    continue
                physical = item[len("chroma_"):]
                if not physical or not os.listdir(kb_dir):
                    continue  # 跳过空目录与畸形名
                await conn.execute(
                    "INSERT OR IGNORE INTO knowledge_bases "
                    "(name, owner_id, physical_name, created_at, updated_at) "
                    "VALUES (?, NULL, ?, ?, ?)",
                    (physical, physical, now, now),
                )
            await conn.commit()
    finally:
        await conn.close()


# ──────── 会话操作 ────────

async def create_session(title: str = "新对话", user_id: Optional[int] = None) -> dict:
    """创建新会话（user_id 为 None 时归入匿名会话桶）"""
    session_id = uuid.uuid4().hex[:16]
    now = datetime.now().isoformat()

    conn = await _get_conn()
    try:
        await conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at, user_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, now, now, user_id),
        )
        await conn.commit()
    finally:
        await _release_conn(conn)
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


async def get_sessions(limit: int = 50, user_id: Optional[int] = None) -> List[dict]:
    """
    获取会话列表（按用户隔离）。

    · user_id 为具体值 → 只返回该用户的会话
    · user_id 为 None  → 只返回匿名会话（向后兼容未登录的旧客户端）
    · 管理端查看全部会话请使用 user_store.get_all_sessions_with_user()
    """
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            """SELECT s.*, 
               (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) as message_count
               FROM chat_sessions s 
               WHERE s.user_id IS ?
               ORDER BY s.updated_at DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await _release_conn(conn)


async def get_session(session_id: str) -> Optional[dict]:
    """获取单个会话"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await _release_conn(conn)


async def update_session_title(session_id: str, title: str):
    """更新会话标题"""
    conn = await _get_conn()
    try:
        await conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.now().isoformat(), session_id),
        )
        await conn.commit()
    finally:
        await _release_conn(conn)


async def delete_session(session_id: str):
    """删除会话及其消息"""
    conn = await _get_conn()
    try:
        await conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        await conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        await conn.commit()
    finally:
        await _release_conn(conn)


# ──────── 消息操作 ────────

async def save_message(session_id: str, role: str, content: str, sources: str = None):
    """保存一条消息"""
    conn = await _get_conn()
    try:
        now = datetime.now().isoformat()
        await conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, sources, now),
        )
        await conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await conn.commit()
    finally:
        await _release_conn(conn)


async def get_messages(session_id: str, limit: int = 100) -> List[dict]:
    """获取会话中的消息"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            "SELECT role, content, sources, created_at FROM chat_messages "
            "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await _release_conn(conn)


# ──────── 启动时初始化 ────────
# 注意: init_db() 为异步函数，需在事件循环中调用。
# 在 FastAPI lifespan 中会自动调用 init_db()。
# 如需在脚本中直接使用，请手动调用 await init_db()。
