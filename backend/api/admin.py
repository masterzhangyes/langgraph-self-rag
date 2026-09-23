"""
管理后台模块 - 用户管理 / 运营统计 / 会话审计 / 操作日志
======================================================

路由前缀: /api/admin（全部经 require_admin 守卫）

功能:
    · GET    /stats              平台运营统计（用户/会话/消息/今日活跃）
    · GET    /users              用户列表
    · PATCH  /users/{id}         更新用户（角色 / 启停 / 重置密码）
    · DELETE /users/{id}         删除用户（级联删除其会话与消息）
    · GET    /sessions           全量会话列表（含归属用户）
    · DELETE /sessions/{id}      删除任意会话
    · GET    /audit-logs         审计日志（登录 / 注册 / 管理操作）

安全设计:
    · 路由级 RBAC 守卫（dependencies=[Depends(require_admin)]）
    · 防误操作: 不可删除自己 / 不可禁用自己 / 不可降级或删除最后一个管理员
    · 所有写操作写审计日志，含操作者与目标
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require_admin
from api.models import AdminUserUpdate
from infrastructure import user_store
from infrastructure.database import delete_session

router = APIRouter(
    prefix="/api/admin",
    tags=["管理后台"],
    dependencies=[Depends(require_admin)],
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")


# ═══════════════════════════════════════════════════════════════
# 运营统计
# ═══════════════════════════════════════════════════════════════

@router.get("/stats")
async def admin_stats():
    """
    平台运营统计。

    返回: 用户总数 / 活跃用户 / 管理员数 / 今日新增、
          会话总数 / 消息总数 / 今日活跃用户数。
    """
    stats = await user_store.get_admin_stats()
    # 附带知识库规模（来自 Chroma 层）
    try:
        from core.rag_chain import get_all_kb_stats
        stats["knowledge_bases"] = get_all_kb_stats()
    except Exception:
        stats["knowledge_bases"] = []
    return stats


# ═══════════════════════════════════════════════════════════════
# 用户管理
# ═══════════════════════════════════════════════════════════════

@router.get("/users")
async def admin_list_users():
    """用户列表（含会话数统计，脱敏不含密码哈希）"""
    return {"users": await user_store.list_users()}


@router.patch("/users/{user_id}")
async def admin_update_user(
    user_id: int,
    req: AdminUserUpdate,
    request: Request,
    admin: dict = Depends(require_admin),
):
    """
    更新用户：角色 / 启用状态 / 重置密码（字段均可选）。

    防误操作规则:
        · 不能对自己执行禁用或降级（防止管理员把自己锁死）
        · 不能禁用 / 降级最后一个活跃管理员（保证系统始终有人可管理）
    """
    target = await user_store.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")

    is_self = target["id"] == admin["id"]
    demoting_last_admin = (
        target["role"] == "admin"
        and (
            (req.role == "user")
            or (req.is_active is False)
        )
        and await user_store.count_admins() <= 1
    )

    if is_self and (req.is_active is False or req.role == "user"):
        raise HTTPException(status_code=400, detail="不能禁用或降级当前登录的管理员自己")
    if demoting_last_admin:
        raise HTTPException(status_code=400, detail="系统必须保留至少一个活跃管理员")

    ok = await user_store.update_user(
        user_id, role=req.role, is_active=req.is_active, password=req.password,
    )
    if not ok:
        raise HTTPException(status_code=400, detail="没有任何需要更新的字段")

    changes = []
    if req.role is not None:
        changes.append(f"角色→{req.role}")
    if req.is_active is not None:
        changes.append("启用" if req.is_active else "禁用")
    if req.password is not None:
        changes.append("重置密码")
    await user_store.record_audit(
        admin["id"], admin["username"], "admin_update_user",
        f"目标用户 #{user_id}({target['username']}): {', '.join(changes)}",
        _client_ip(request),
    )
    return {"message": "用户已更新"}


@router.delete("/users/{user_id}")
async def admin_delete_user(user_id: int, request: Request, admin: dict = Depends(require_admin)):
    """
    删除用户（级联删除其全部会话与消息，不可恢复）。

    规则: 不能删除自己；不能删除最后一个管理员。
    """
    target = await user_store.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")
    if target["id"] == admin["id"]:
        raise HTTPException(status_code=400, detail="不能删除当前登录的自己")
    if target["role"] == "admin" and await user_store.count_admins() <= 1:
        raise HTTPException(status_code=400, detail="系统必须保留至少一个活跃管理员")

    await user_store.delete_user(user_id)
    await user_store.record_audit(
        admin["id"], admin["username"], "admin_delete_user",
        f"删除用户 #{user_id}({target['username']}) 及其全部会话",
        _client_ip(request),
    )
    return {"message": "用户及其会话已删除"}


# ═══════════════════════════════════════════════════════════════
# 会话审计
# ═══════════════════════════════════════════════════════════════

@router.get("/sessions")
async def admin_list_sessions(limit: int = 200):
    """全量会话列表（含归属用户名，user_id 为空表示匿名会话）"""
    return {"sessions": await user_store.get_all_sessions_with_user(limit=limit)}


@router.delete("/sessions/{session_id}")
async def admin_delete_session(session_id: str, request: Request, admin: dict = Depends(require_admin)):
    """删除任意会话（内容治理）"""
    await delete_session(session_id)
    await user_store.record_audit(
        admin["id"], admin["username"], "admin_delete_session",
        f"删除会话 {session_id}", _client_ip(request),
    )
    return {"message": "会话已删除"}


# ═══════════════════════════════════════════════════════════════
# 审计日志
# ═══════════════════════════════════════════════════════════════

@router.get("/audit-logs")
async def admin_audit_logs(limit: int = 100):
    """最近审计日志（登录 / 注册 / 管理操作，含 IP）"""
    return {"logs": await user_store.get_audit_logs(limit=limit)}
