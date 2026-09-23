"""
知识库注册表存取层（v2.3 新增）
==============================

实现「按用户隔离」的知识库模型:

  · 私有库 — owner_id 指向具体用户，仅本人（及管理员）可见可写
  · 公共库 — owner_id 为 NULL，所有人可读，仅管理员可写
    （服务器上历史遗留的 chroma_* 目录会在启动时自动迁移注册为公共库）

核心设计 — 逻辑名与物理名解耦:
    用户眼中的知识库名（name）可以跨用户重复（类似网盘的文件夹名），
    而磁盘上的向量目录（chroma_{physical_name}）全局唯一:
        私有库 → "u{owner_id}__{name}"     如 u3__技术文档
        公共库 → "pub__{name}"             如 pub__团队 wiki
    从而在存储层彻底杜绝跨用户数据串读。

权限模型（两级 RBAC 之下的资源级 ACL）:
    · 读取/检索: 本人私有库 + 所有公共库（管理员: 任意库）
    · 写入/清空: 本人私有库（管理员: 任意库；公共库仅管理员）
"""
import re
from datetime import datetime
from typing import List, Optional

from infrastructure.database import _get_conn, _release_conn

# 用户可见的逻辑名白名单: 字母/数字/下划线/中文/连字符，其余替换为 _
_SAFE_NAME = re.compile(r"[^\w\u4e00-\u9fa5-]")


def sanitize_kb_name(name: str) -> str:
    """
    清洗知识库逻辑名，防止路径注入与超长目录名。

    · 去首尾空白后，把白名单以外字符替换为下划线
    · 截断到 64 字符（目录名安全上限）
    · 空名兜底为 "kb"
    """
    cleaned = _SAFE_NAME.sub("_", (name or "").strip())[:64]
    return cleaned or "kb"


def physical_name_for(name: str, owner_id: Optional[int]) -> str:
    """根据逻辑名与归属用户生成全局唯一的物理目录名"""
    if owner_id is None:
        return f"pub__{sanitize_kb_name(name)}"
    return f"u{owner_id}__{sanitize_kb_name(name)}"


def _row_to_dict(row) -> Optional[dict]:
    """把 aiosqlite.Row 转为普通 dict，并附加 is_public 派生字段"""
    if row is None:
        return None
    d = dict(row)
    d["is_public"] = d.get("owner_id") is None
    return d


async def get_kb(name: str, owner_id: Optional[int]) -> Optional[dict]:
    """精确获取（owner, name）对应的知识库记录；owner_id=None 查公共库"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM knowledge_bases WHERE name = ? AND owner_id IS ?",
            (name, owner_id),
        )
        return _row_to_dict(await cursor.fetchone())
    finally:
        await _release_conn(conn)


async def get_kb_any(name: str) -> Optional[dict]:
    """按名称获取任意归属的知识库（仅管理员场景使用；私有库优先）"""
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM knowledge_bases WHERE name = ? "
            "ORDER BY CASE WHEN owner_id IS NULL THEN 1 ELSE 0 END, id",
            (name,),
        )
        return _row_to_dict(await cursor.fetchone())
    finally:
        await _release_conn(conn)


async def register_kb(name: str, owner_id: Optional[int]) -> dict:
    """
    注册（或获取已存在的）知识库。

    · （owner, name）已存在 → 直接返回旧记录（追加文档场景）
    · 不存在 → 生成物理名并插入注册表
    """
    name = (name or "").strip()
    existing = await get_kb(name, owner_id)
    if existing:
        return existing

    now = datetime.now().isoformat()
    physical = physical_name_for(name, owner_id)
    conn = await _get_conn()
    try:
        cursor = await conn.execute(
            "INSERT INTO knowledge_bases (name, owner_id, physical_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, owner_id, physical, now, now),
        )
        await conn.commit()
        kb_id = cursor.lastrowid
    finally:
        await _release_conn(conn)

    row = await get_kb(name, owner_id)
    return row or {"id": kb_id, "name": name, "owner_id": owner_id,
                   "physical_name": physical, "is_public": owner_id is None}


async def list_kbs(user: Optional[dict]) -> List[dict]:
    """
    列出当前用户可见的知识库（带所有者用户名）。

    · 普通用户 → 本人私有库 + 所有公共库
    · 管理员   → 全部知识库（含其他用户的私有库）
    · 未登录   → 仅公共库
    """
    conn = await _get_conn()
    try:
        if user and user.get("role") == "admin":
            cursor = await conn.execute(
                "SELECT k.*, u.username AS owner_username "
                "FROM knowledge_bases k LEFT JOIN users u ON u.id = k.owner_id "
                "ORDER BY k.updated_at DESC"
            )
        else:
            cursor = await conn.execute(
                "SELECT k.*, u.username AS owner_username "
                "FROM knowledge_bases k LEFT JOIN users u ON u.id = k.owner_id "
                "WHERE k.owner_id IS ? OR k.owner_id IS NULL "
                "ORDER BY k.updated_at DESC",
                (user["id"] if user else None,),
            )
        rows = [_row_to_dict(r) for r in await cursor.fetchall()]
        return [r for r in rows if r]
    finally:
        await _release_conn(conn)


async def touch_kb(kb_id: int) -> None:
    """更新知识库的 updated_at（上传/写入文档后调用）"""
    conn = await _get_conn()
    try:
        await conn.execute(
            "UPDATE knowledge_bases SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), kb_id),
        )
        await conn.commit()
    finally:
        await _release_conn(conn)


async def resolve_readable(name: str, user: Optional[dict]) -> Optional[dict]:
    """
    将逻辑名解析为当前用户**可读**（检索/统计）的知识库记录。

    解析顺序（短路返回）:
        1. 本人同名的私有库
        2. 管理员 → 任意归属的同名库
        3. 同名公共库
        4. "default" 且注册表中无任何记录 → 合成遗留兼容记录
           （保证匿名对话 / 空库部署下默认知识库行为不变）
        5. 其余 → None（调用方返回 404，不泄露他人库的存在性）
    """
    if user:
        row = await get_kb(name, user["id"])
        if row:
            return row
        if user.get("role") == "admin":
            row = await get_kb_any(name)
            if row:
                return row
    row = await get_kb(name, None)
    if row:
        return row
    if name == "default":
        return {"id": None, "name": "default", "owner_id": None,
                "physical_name": "default", "is_public": True}
    return None


async def resolve_writable(name: str, user: dict) -> tuple:
    """
    将逻辑名解析为当前用户**可写**（上传/清空）的知识库记录。

    返回 (kb, need_create):
        · kb 为记录，need_create=False → 可直接写入
        · kb 为公共库记录且非管理员 → 调用方应返回 403
        · kb 为 None，need_create=True → 调用方应新建私有库

    解析顺序:
        1. 本人同名的私有库（可写）
        2. 管理员 → 任意归属的同名库（可写）
        3. 同名公共库 → 仅管理员可写（need_create=False，
           由调用方根据 user.role 决定放行或 403）
        4. 其余 → (None, True) 新建私有库
    """
    row = await get_kb(name, user["id"])
    if row:
        return row, False
    if user.get("role") == "admin":
        row = await get_kb_any(name)
        if row:
            return row, False
    row = await get_kb(name, None)
    if row:
        # 公共库: 管理员可直接写; 普通用户由调用方拦截（403）
        return row, False
    return None, True
