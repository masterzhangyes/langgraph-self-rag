"""
SQLite 数据库层 - 对话历史持久化
使用 aiosqlite + 信号量连接池实现真正的异步操作
================================================

为什么 SQLite 也要"连接池":
    SQLite 是单文件嵌入式库，没有独立服务进程。但 aiosqlite 每个连接
    背后是一个独立线程 + WAL 模式下允许多读单写 —— 高并发下若随意
    开关连接，会频繁付线程创建/销毁与 PRAGMA 初始化的代价。
    连接池 = 复用已初始化的连接 + 用信号量限流并发写，一举两得。

设计概览（学习时按这个顺序读）:
    1. 连接池三件套: _get_conn / _release_conn / _close_pool
       （信号量限并发 + 互斥锁保池操作原子性 + 健康检查复用）
    2. init_db(): 建表 + 两个"幂等迁移"（列迁移 / 旧数据注册）
    3. CRUD: 会话与消息的增删改查（全部走池，try/finally 保证归还）

Schema 设计要点:
    · 外键全部 ON DELETE CASCADE —— 删会话自动清消息、删用户自动清会话，
      不需要应用层手动维护级联；
    · 密码只存 bcrypt 哈希（password_hash），绝不存明文；
    · role 用 CHECK 约束在数据库层兜底非法值。
"""
import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
import aiosqlite

# settings: 全局配置（DB_PATH 数据库文件路径 / VECTOR_DB_PATH 向量库路径）
from core.config import settings


def get_db_path() -> str:
    """
    获取数据库路径，确保目录存在。

    为什么不直接用 settings.DB_PATH:
        首次部署/新环境时 data/ 目录可能不存在，
        aiosqlite.connect 遇到不存在的父目录会直接报错。
        这里在返回前先补齐目录，调用方就永远无需关心这件事。
    """
    db_dir = os.path.dirname(settings.DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    return settings.DB_PATH


# ──────── 改进版连接池 ────────
# 三个模块级协作者（学习重点 —— 经典的"信号量 + 互斥锁"并发模型）:
#   _pool           空闲连接列表（共享可变状态）
#   _pool_semaphore 并发闸门: acquire 成功才有权拿到一个连接，
#                   release 归还连接时放行下一个等待者
#                   → 在任意时刻"借出连接数 ≤ 10"
#   _pool_lock      只保护 _pool 列表本身的 pop/append 操作。
#                   ⚠️ 与信号量的分工: 信号量管"数量"，锁管"列表操作的原子性"，
#                   二者不可互相替代 —— 没锁时两个协程可能同时 pop 同一个连接。

_POOL_SIZE = 10
_pool: List[aiosqlite.Connection] = []
_pool_semaphore = asyncio.Semaphore(_POOL_SIZE)  # 限制最大并发连接数
_pool_lock = asyncio.Lock()


async def _get_conn() -> aiosqlite.Connection:
    """
    从连接池获取连接（带超时和并发控制）。

    信号量的获取/释放跨越整个借还周期:
        acquire  → 拿到"配额"
        借出连接 → 配额随连接一起离开本函数
        _release_conn 里 release → 还配额

    失败补偿: 内部抛异常时必须手动 release，
    否则配额泄漏 → 池越用越少直到永久饿死（死锁）。
    """
    await _pool_semaphore.acquire()
    try:
        return await _get_conn_inner()
    except Exception:
        # 拿到了配额但没能拿到连接 —— 释放配额再抛，
        # 保证"配额数 == 借出连接数"这一不变式
        _pool_semaphore.release()
        raise


async def _get_conn_inner() -> aiosqlite.Connection:
    """内部获取连接逻辑（已持有信号量配额，此函数只管拿连接）"""
    async with _pool_lock:
        # 尝试复用池中连接
        # while 而非 if: 池顶连接可能已失效（被数据库关闭/超时），
        # 换下一个再试，直到池空或找到活的
        while _pool:
            conn = _pool.pop()
            try:
                # 健康检查: SELECT 1 是最轻量的探活语句。
                # 连接失效会抛异常 → 丢弃这条死连接继续尝试
                await conn.execute("SELECT 1")
                # row_factory=Row 让查询结果可以 dict(行) / 行["列名"] 访问，
                # 比裸 tuple 可读性强得多（上层全部依赖这个行为）
                conn.row_factory = aiosqlite.Row
                return conn
            except Exception:
                # 死连接关闭失败也无所谓（可能早已断开），吞掉继续
                try:
                    await conn.close()
                except Exception:
                    pass

    # 池里没有可用连接 → 开新连接。
    # 注意: 开新连接在锁外执行（慢操作不占锁，减少其他协程等待）；
    # 并发上限已由外层信号量保证，这里不会开爆
    conn = await aiosqlite.connect(get_db_path())
    conn.row_factory = aiosqlite.Row
    # 三条 PRAGMA 是 SQLite 生产环境的经典三件套:
    #   WAL             写不改读、读不阻塞写，并发性能大幅提升
    #                  （代价: 目录下多出 -wal/-shm 文件，属正常现象）
    #   foreign_keys=ON 必须显式开启！SQLite 默认"忽略"外键约束，
    #                  不开的话 ON DELETE CASCADE 完全不生效
    #   busy_timeout    遇到写锁冲突时先等 5 秒再报 SQLITE_BUSY，
    #                  而不是立刻失败 —— 给并发写留出排队窗口
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute("PRAGMA busy_timeout=5000")
    return conn


async def _release_conn(conn: aiosqlite.Connection):
    """
    归还连接到池。

    两条纪律:
        1. 池满（len == 10）就不强行塞，直接关掉该连接
           （借出的连接数可能 > 池容量？不 —— 信号量保证借出 ≤ 10，
            但归还时池里可能已有其他空闲连接，满了就关闭多余的）；
        2. release 信号量必须放在 finally —— 无论归还过程出什么错，
           配额都必须还，这是防止池饿死的不变式。
    """
    try:
        async with _pool_lock:
            if len(_pool) < _POOL_SIZE:
                # PRAGMA optimize: 顺手更新查询计划统计信息，
                # 对 SQLite 长期运行性能有微小收益
                await conn.execute("PRAGMA optimize")
                _pool.append(conn)
            else:
                await conn.close()
    finally:
        _pool_semaphore.release()


async def _close_pool():
    """
    关闭所有连接（应用退出时调用）。

    ⚠️ 局限: 只关了"池中空闲"连接，借出中的连接不在此列 ——
    严谨做法应等待信号量满额（所有连接都已归还）再关闭。
    当前作为关闭钩子已够用（进程退出 OS 会回收一切）。
    """
    async with _pool_lock:
        for conn in _pool:
            try:
                await conn.close()
            except Exception:
                pass  # 关闭失败只能放弃，不能影响其他连接的关闭
        _pool.clear()


# ──────── 数据库初始化 ────────

async def init_db():
    """
    初始化数据库表。

    幂等性设计（重点）: 全部 DDL 用 CREATE TABLE/INDEX IF NOT EXISTS，
    重复执行零副作用 —— 启动脚本可以放心每次都调。

    三段职责:
        1. executescript 建全量 schema（表 + 索引）
        2. 列级迁移: 为旧库补 user_id 列
        3. 数据迁移: 把磁盘上已存在的旧向量库目录注册进 knowledge_bases 表

    注意: 这里直接 connect 不走池 —— 初始化发生在服务就绪之前，
    池中的 PRAGMA（WAL 等）也依赖库文件先建好。
    """
    conn = await aiosqlite.connect(get_db_path())
    try:
        await conn.executescript("""
            -- 用户表（v2.2 新增）
            -- 密码只存 bcrypt 哈希，绝不存明文；
            -- role 实现两级 RBAC（user / admin）
            -- CHECK 约束在数据库层兜底: 即使应用代码有 bug
            -- 也插不进 role='superadmin' 之类的脏值
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
            -- 时间用 TEXT 存 ISO 字符串而非 TIMESTAMP:
            -- SQLite 没有真正的日期类型，ISO 格式字符串可按字典序比较，
            -- 恰好等价于时间先后排序（ORDER BY updated_at 直接可用）
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '新对话',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
            );

            -- 消息表
            -- sources 存 JSON 字符串（引用来源列表），应用层负责序列化
            -- (session_id, id) 复合索引支撑"按会话取消息按时间正序"的查询
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
            -- 不设外键: 用户被删后日志仍需保留（审计数据不能随人消失）
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
            -- 逻辑名/物理名分离的意义: 用户可以随时改展示名（name），
            -- 而磁盘目录与 Chroma 集合名（physical_name）保持稳定，
            -- 不会因改名导致向量库失联
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                physical_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- 索引
            -- 索引建在"查询会走的列"上，看索引就能反推查询模式:
            CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON chat_messages(session_id, id);          -- 取某会话的消息（按序）
            CREATE INDEX IF NOT EXISTS idx_sessions_updated 
                ON chat_sessions(updated_at DESC);          -- 会话列表按最近活跃排
            CREATE INDEX IF NOT EXISTS idx_audit_created 
                ON audit_logs(created_at DESC);             -- 审计日志按时间倒序
            -- （owner, name) 唯一: SQLite 中 NULL 互不相等，用 ifnull 归一化后建表达式索引
            -- 背景: SQL 标准里 NULL != NULL，两条 (NULL, '知识库') 都能插入，
            -- 唯一约束对"公共库同名"会失效。ifnull(owner_id,-1) 把 NULL 映射成 -1，
            -- 语义上仍唯一（用户 id 从 1 自增，不会撞 -1）
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_owner_name
                ON knowledge_bases(ifnull(owner_id, -1), name);
        """)
        await conn.commit()

        # ── 幂等迁移：为旧库的 chat_sessions 补充 user_id 归属列 ──
        # SQLite 的 ALTER TABLE 只支持 ADD COLUMN（不能改类型/加约束的完整版），
        # 所以迁移模式是: PRAGMA table_info 查现有列 → 缺则补列。
        # 旧数据 user_id 为 NULL → 归入「匿名会话」桶，不破坏向后兼容
        cursor = await conn.execute("PRAGMA table_info(chat_sessions)")
        columns = [row[1] for row in await cursor.fetchall()]  # row[1] 是列名
        if "user_id" not in columns:
            await conn.execute(
                "ALTER TABLE chat_sessions ADD COLUMN user_id INTEGER "
                "REFERENCES users(id) ON DELETE CASCADE"
            )
            await conn.commit()

        # 会话归属索引（迁移完成后创建，保证旧库也能建上）
        # 放在迁移之后而非 executescript 里: 若表还没有 user_id 列，
        # executescript 里这条会直接报错 —— 顺序即依赖
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user "
            "ON chat_sessions(user_id, updated_at DESC)"
        )
        await conn.commit()

        # ── 幂等迁移：把磁盘上已存在的旧知识库目录注册为公共知识库 ──
        # v2.3 之前知识库无归属概念；升级后统一归入「公共库」桶:
        # 所有人可读（保持现有检索行为），仅管理员可写。
        #
        # 为什么扫磁盘而不是扫表: "已存在的库"事实在磁盘上（chroma_* 目录），
        # 表里没有任何记录 —— 以磁盘为源做一次对账式导入。
        # INSERT OR IGNORE 依赖唯一索引，重复执行无副作用（幂等）。
        kb_base = os.path.dirname(settings.VECTOR_DB_PATH) or "."
        if os.path.isdir(kb_base):
            now = datetime.now().isoformat()
            for item in os.listdir(kb_base):
                kb_dir = os.path.join(kb_base, item)
                # 过滤条件逐层收紧: chroma_ 前缀 → 是目录 → 名字非空 → 目录非空
                if not (item.startswith("chroma_") and os.path.isdir(kb_dir)):
                    continue
                physical = item[len("chroma_"):]
                if not physical or not os.listdir(kb_dir):
                    continue  # 跳过空目录与畸形名
                # owner_id 传 NULL → 公共库；name 与 physical 同名（旧库无改名历史）
                await conn.execute(
                    "INSERT OR IGNORE INTO knowledge_bases "
                    "(name, owner_id, physical_name, created_at, updated_at) "
                    "VALUES (?, NULL, ?, ?, ?)",
                    (physical, physical, now, now),
                )
            await conn.commit()
    finally:
        # init 不走池，连接必须显式关闭（finally 保证异常路径也不泄漏）
        await conn.close()


# ──────── 会话操作 ────────
# CRUD 模板（五个函数同一个骨架，学一个等于学全部）:
#     conn = await _get_conn()        # 借连接
#     try:
#         ...业务 SQL（全部参数化 ? 占位，防 SQL 注入）...
#         await conn.commit()          # 写操作必须提交
#     finally:
#         await _release_conn(conn)    # 无论成败都归还 —— 连接不泄漏
# 读操作（SELECT）不需要 commit；写操作（INSERT/UPDATE/DELETE）必须。

async def create_session(title: str = "新对话", user_id: Optional[int] = None) -> dict:
    """创建新会话（user_id 为 None 时归入匿名会话桶）"""
    # uuid4 hex 截 16 位做主键: 全局唯一（跨机器可迁库）、
    # 不暴露自增 id 的"总共有多少会话"信息、URL 友好
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
    # 返回 dict 而非只给 id: 调用方常直接把它塞进响应/前端状态
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


async def get_sessions(limit: int = 50, user_id: Optional[int] = None) -> List[dict]:
    """
    获取会话列表（按用户隔离）。

    · user_id 为具体值 → 只返回该用户的会话
    · user_id 为 None  → 只返回匿名会话（向后兼容未登录的旧客户端）
    · 管理端查看全部会话请使用 user_store.get_all_sessions_with_user()

    隔离的实现精髓: WHERE s.user_id IS ?
    —— 普通的 "user_id = NULL" 在 SQL 里永远为假（NULL 比较返回 UNKNOWN），
    想匹配 NULL 必须用 IS。参数传 None 时即匹配匿名会话桶。
    """
    conn = await _get_conn()
    try:
        # 相关子查询内联算消息数: 一次查询带出列表页所需全部字段，
        # 避免 N+1（每行再查一次 COUNT）。有 idx_messages_session 索引支撑
        cursor = await conn.execute(
            """SELECT s.*, 
               (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) as message_count
               FROM chat_sessions s 
               WHERE s.user_id IS ?
               ORDER BY s.updated_at DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        # aiosqlite.Row → dict，方便上层 JSON 序列化
        return [dict(r) for r in rows]
    finally:
        await _release_conn(conn)


async def get_session(session_id: str) -> Optional[dict]:
    """获取单个会话（不存在返回 None，调用方据此判 404）"""
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
    """
    更新会话标题。

    顺手维护 updated_at —— "最近活跃"排序依赖它，
    改名也算一次交互，会话应浮到列表顶部（产品语义）。
    """
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
    """
    删除会话及其消息。

    为什么手动删两张表而不是靠外键 CASCADE:
        功能上 CASCADE 可覆盖（PRAGMA foreign_keys=ON 已开），
        但显式删除让语义一目了然，也不依赖读者记得"外键开关"这件事。
        两条 DELETE 在同一事务里 commit —— 要么都成，要么都不成。
    """
    conn = await _get_conn()
    try:
        # 先删子表（消息）再删父表（会话），顺序符合外键依赖方向
        await conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        await conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        await conn.commit()
    finally:
        await _release_conn(conn)


# ──────── 消息操作 ────────

async def save_message(session_id: str, role: str, content: str, sources: str = None):
    """
    保存一条消息。

    一并推进会话的 updated_at —— 会话列表的"最近活跃"排序
    就靠每次存消息时刷新，无需额外的定时任务。
    两次写同一事务提交，保证消息与时间戳的一致性。
    """
    conn = await _get_conn()
    try:
        now = datetime.now().isoformat()
        # sources 为 JSON 字符串（由上层序列化），无引用来源时为 NULL
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
    """
    获取会话中的消息。

    ORDER BY id ASC: 自增 id 即插入顺序，等价于时间正序 ——
    拉历史消息给 LLM 构造对话上下文时必须是"用户→助手→用户…"的正序。
    limit 兜底超长会话（默认 100 条），防止把整个上下文窗口塞爆。
    """
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
