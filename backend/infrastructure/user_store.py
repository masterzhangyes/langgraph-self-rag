"""
用户数据访问层 - 用户 / 审计日志 / 管理统计
==========================================

职责:
    · 用户 CRUD（密码 bcrypt 哈希、角色、激活状态）
    · 审计日志（登录 / 注册 / 管理操作）
    · 管理后台聚合统计与全量会话查询

设计说明:
    · 复用 database.py 的异步连接池（_get_conn / _release_conn），
      避免引入第二套数据库连接管理。
    · 密码只存 bcrypt 哈希（自带随机盐），明文密码不落库、不打日志。
    · 用户表的增删会级联清理其会话与消息（手动双删除，不依赖 PRAGMA foreign_keys）。
"""
import logging
from datetime import datetime
from typing import List, Optional

import bcrypt

from infrastructure.database import _get_conn, _release_conn

logger = logging.getLogger("qa.user_store")


# ──────── 密码哈希（bcrypt） ────────

def hash_password(plain: str) -> str:
    """
    bcrypt 哈希密码。

    bcrypt 内置随机盐（同一密码每次哈希结果不同，抗彩虹表），
    且 cost factor 可随硬件升级而提高，抗暴力破解。
    """
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码与 bcrypt 哈希是否匹配（常量时间比较）"""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


# ──────── 用户 CRUD ────────

async def create_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    role: str = "user",
) -> dict:
    """创建用户并返回脱敏后的用户信息（不含密码哈希）"""
    now = datetime.now().isoformat()
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            "INSERT INTO users (username, email, password_hash, role, is_active, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (username, email, hash_password(password), role, now),
        )
        await conn.commit()
        user_id = cursor.lastrowid
    finally:
        await _release_conn(conn)
    return {
        "id": user_id,
        "username": username,
        "email": email,
        "role": role,
        "is_active": True,
        "created_at": now,
        "last_login_at": None,
    }


async def get_user_by_username(username: str) -> Optional[dict]:
    """按用户名查询（含 password_hash，仅供认证流程使用）"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await _release_conn(conn)


async def get_user_by_id(user_id: int) -> Optional[dict]:
    """按 ID 查询（含 password_hash，仅供认证流程使用）"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await _release_conn(conn)


async def count_users() -> int:
    """用户总数（用于「首个注册用户自动成为管理员」的引导约定）"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute("SELECT COUNT(*) FROM users")
        row = await cursor.fetchone()
        return row[0]
    finally:
        await _release_conn(conn)


async def count_admins() -> int:
    """活跃管理员数量（防止误删 / 误降级最后一个管理员）"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1")
        row = await cursor.fetchone()
        return row[0]
    finally:
        await _release_conn(conn)


async def update_last_login(user_id: int):
    """登录成功后更新最近登录时间"""
    conn = await _get_conn()
    try:
        await conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id),
        )
        await conn.commit()
    finally:
        await _release_conn(conn)


async def list_users() -> List[dict]:
    """用户列表（管理后台用，脱敏 + 附带会话统计）"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            """SELECT u.id, u.username, u.email, u.role, u.is_active,
                      u.created_at, u.last_login_at,
                      (SELECT COUNT(*) FROM chat_sessions WHERE user_id = u.id) as session_count
               FROM users u ORDER BY u.created_at DESC"""
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await _release_conn(conn)


async def update_user(
    user_id: int,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    password: Optional[str] = None,
) -> bool:
    """更新用户角色 / 激活状态 / 重置密码。返回是否命中目标用户。"""
    updates, params = [], []
    if role is not None:
        updates.append("role = ?")
        params.append(role)
    if is_active is not None:
        updates.append("is_active = ?")
        params.append(1 if is_active else 0)
    if password is not None:
        updates.append("password_hash = ?")
        params.append(hash_password(password))
    if not updates:
        return False

    params.append(user_id)
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params
        )
        await conn.commit()
        return cursor.rowcount > 0
    finally:
        await _release_conn(conn)


async def delete_user(user_id: int) -> bool:
    """
    删除用户，并显式级联删除其全部会话与消息。

    不依赖 PRAGMA foreign_keys 开关（显式删除在任何配置下都正确），
    这是 SQLite 环境下更稳妥的级联策略。
    """
    conn = await _get_conn()
    try:
        await conn.execute(
            "DELETE FROM chat_messages WHERE session_id IN "
            "(SELECT id FROM chat_sessions WHERE user_id = ?)",
            (user_id,),
        )
        await conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
        cursor = await conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await conn.commit()
        return cursor.rowcount > 0
    finally:
        await _release_conn(conn)


# ──────── 审计日志 ────────

async def record_audit(
    user_id: Optional[int],
    username: Optional[str],
    action: str,
    detail: str = "",
    ip: str = "",
):
    """写入一条审计日志（尽力而为，失败不影响主流程）"""
    try:
        conn = await _get_conn()
        try:
            await conn.execute(
                "INSERT INTO audit_logs (user_id, username, action, detail, ip, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, username, action, detail, ip, datetime.now().isoformat()),
            )
            await conn.commit()
        finally:
            await _release_conn(conn)
    except Exception:
        logger.exception("写入审计日志失败 action=%s", action)


async def get_audit_logs(limit: int = 100) -> List[dict]:
    """查询最近的审计日志（管理后台用）"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            "SELECT id, user_id, username, action, detail, ip, created_at "
            "FROM audit_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await _release_conn(conn)


# ──────── 管理后台聚合查询 ────────

async def get_admin_stats() -> dict:
    """平台运营统计：用户 / 会话 / 消息 / 今日活跃"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM users)                                  AS total_users,
              (SELECT COUNT(*) FROM users WHERE is_active = 1)              AS active_users,
              (SELECT COUNT(*) FROM users WHERE role = 'admin')             AS admin_users,
              (SELECT COUNT(*) FROM users
                 WHERE date(created_at) = date('now', 'localtime'))         AS new_users_today,
              (SELECT COUNT(*) FROM chat_sessions)                          AS total_sessions,
              (SELECT COUNT(*) FROM chat_messages)                          AS total_messages,
              (SELECT COUNT(DISTINCT user_id) FROM chat_sessions
                 WHERE date(updated_at) = date('now', 'localtime')
                   AND user_id IS NOT NULL)                                 AS active_users_today
            """
        )
        row = dict(await cursor.fetchone())
        row["active_users"] = bool(row.get("active_users"))
        return row
    finally:
        await _release_conn(conn)


async def get_all_sessions_with_user(limit: int = 200) -> List[dict]:
    """全量会话列表（含归属用户名），供管理后台审计用户会话"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            """SELECT s.id, s.title, s.created_at, s.updated_at, s.user_id,
                      u.username,
                      (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) AS message_count
               FROM chat_sessions s
               LEFT JOIN users u ON u.id = s.user_id
               ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await _release_conn(conn)
