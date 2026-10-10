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
# ────────────────────────── 依赖导入 ──────────────────────────
# FastAPI 路由与依赖注入相关：
#   APIRouter    构建子路由（统一挂 /api/admin 前缀）
#   Depends      依赖注入：把守卫函数挂到路由上
#   HTTPException 以 RESTful 语义抛出 4xx 错误（自动转 JSON 响应）
#   Request      拿到原始请求对象（用于提取客户端 IP）
from fastapi import APIRouter, Depends, HTTPException, Request

# require_admin: 管理员守卫
#   1) 解析并校验 JWT（Authorization: Bearer <token>）
#   2) 查库确认用户存在、is_active=True
#   3) 校验 role == "admin"，否则抛 403
#   通过后返回当前管理员信息 dict（id / username / role）
from api.auth import require_admin

# AdminUserUpdate: PATCH /users/{id} 的请求体 Pydantic 模型
#   字段全部 Optional —— 只更新传入的字段（部分更新语义）
from api.models import AdminUserUpdate

# user_store: 用户/会话/审计日志的数据访问层（async 接口，底层 SQLite）
import infrastructure.user_store as user_store
# delete_session: 按 session_id 删会话（级联删消息），来自数据库基础设施层
from infrastructure.database import delete_session

# ────────────────────────── 路由器定义 ──────────────────────────
# dependencies 写在路由器级别 = 对本文件所有路由统一生效，
# 无需在每个函数上重复挂守卫（相当于全局中间件，但仅限本路由器）
router = APIRouter(
    prefix="/api/admin",
    tags=["管理后台"],                        # OpenAPI 文档分组（docs 已关闭，仅作语义标注）
    dependencies=[Depends(require_admin)],   # ⚠️ 全部接口仅管理员可用
)


def _client_ip(request: Request) -> str:
    """
    提取客户端真实 IP（用于审计日志留痕）。

    优先级:
        1. X-Forwarded-For 首段 —— 经 nginx 等反向代理转发时的真实客户端 IP
           （代理会追加链路: "客户端IP, 代理1, 代理2"，取第一个即原始来源）
        2. request.client.host —— 直连时的 TCP 对端地址
        3. "unknown"           —— 极端情况下 client 为空的兜底
    """
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
    # 数据库层聚合查询（一次 SQL 拿用户/会话/消息维度计数）
    stats = await user_store.get_admin_stats()
    # 附带知识库规模（来自 Chroma 层）
    try:
        # 局部导入避免循环依赖（rag_chain 反向引用了 API 层的情况）
        # get_all_kb_stats: 遍历所有 Chroma collection，返回各知识库的
        # 名称 / 文档数 / 向量数等元信息
        from core.rag_chain import get_all_kb_stats
        stats["knowledge_bases"] = get_all_kb_stats()
    except Exception:
        # 知识库统计属于"锦上添花"字段，任何异常（如 Chroma 未初始化）
        # 都不应导致整个统计接口 500，降级为空列表
        stats["knowledge_bases"] = []
    return stats


# ═══════════════════════════════════════════════════════════════
# 用户管理
# ═══════════════════════════════════════════════════════════════

@router.get("/users")
async def admin_list_users():
    """
    用户列表（含会话数统计，脱敏不含密码哈希）。

    说明:
        list_users() 在数据层即已剔除 password_hash 字段，
        避免敏感信息（即使已哈希）泄漏到前端。
    """
    return {"users": await user_store.list_users()}


@router.patch("/users/{user_id}")
async def admin_update_user(
    user_id: int,          # 路径参数：目标用户 ID
    req: AdminUserUpdate,  # 请求体：role / is_active / password 三者皆可选
    request: Request,      # 用于提取 IP 写审计日志
    admin: dict = Depends(require_admin),  # 当前操作者（路由器级已守卫一次，
                                            # 这里再注入一次以拿到 admin 身份详情）
):
    """
    更新用户：角色 / 启用状态 / 重置密码（字段均可选）。

    防误操作规则:
        · 不能对自己执行禁用或降级（防止管理员把自己锁死）
        · 不能禁用 / 降级最后一个活跃管理员（保证系统始终有人可管理）
    """
    # ── 1. 目标存在性校验 ──
    target = await user_store.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")

    # ── 2. 业务安全校验（防误操作） ──
    is_self = target["id"] == admin["id"]

    # "降级最后一个管理员"的判定：
    #   目标当前是 admin，且本次操作会使其失去管理能力
    #   （改角色为普通用户，或直接禁用），而系统里活跃管理员只剩 1 个
    demoting_last_admin = (
        target["role"] == "admin"
        and (
            (req.role == "user")          # 降级为普通用户
            or (req.is_active is False)   # 或保持角色但禁用账号
        )
        and await user_store.count_admins() <= 1
    )

    # 规则一：不能对自己禁用/降级 —— 否则自己的 JWT 下次刷新即失效，把自己锁死
    if is_self and (req.is_active is False or req.role == "user"):
        raise HTTPException(status_code=400, detail="不能禁用或降级当前登录的管理员自己")
    # 规则二：系统必须始终保留至少一个活跃管理员
    if demoting_last_admin:
        raise HTTPException(status_code=400, detail="系统必须保留至少一个活跃管理员")

    # ── 3. 执行部分更新 ──
    # update_user 内部只更新非 None 字段；返回 False 表示三个字段全没传
    ok = await user_store.update_user(
        user_id, role=req.role, is_active=req.is_active, password=req.password,
    )
    if not ok:
        raise HTTPException(status_code=400, detail="没有任何需要更新的字段")

    # ── 4. 审计留痕 ──
    # 拼接人类可读的变更摘要，如 "角色→user, 禁用, 重置密码"
    changes = []
    if req.role is not None:
        changes.append(f"角色→{req.role}")
    if req.is_active is not None:
        changes.append("启用" if req.is_active else "禁用")
    if req.password is not None:
        changes.append("重置密码")  # 只记录动作，绝不记录明文密码
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

    ⚠️ 危险操作: delete_user 在数据库层使用外键级联/事务，
       连同该用户的 sessions、messages 一并删除，无法回滚。
    """
    # ── 1. 存在性 + 安全校验 ──
    target = await user_store.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")
    # 不能删自己：一方面防误触，另一方面审计日志需要操作者存在
    if target["id"] == admin["id"]:
        raise HTTPException(status_code=400, detail="不能删除当前登录的自己")
    # 最后一个管理员不可删，否则平台将无人能进入管理后台
    if target["role"] == "admin" and await user_store.count_admins() <= 1:
        raise HTTPException(status_code=400, detail="系统必须保留至少一个活跃管理员")

    # ── 2. 执行级联删除 ──
    await user_store.delete_user(user_id)

    # ── 3. 审计留痕（删除是不可逆操作，必须记录操作者+目标+IP） ──
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
    """
    全量会话列表（含归属用户名，user_id 为空表示匿名会话）。

    参数:
        limit: 返回条数上限，默认 200（防止全表拖垮接口，
               供管理端分页/翻页加载）
    """
    # LEFT JOIN users：匿名会话（无登录用户）也能列出，用户名显示为空
    return {"sessions": await user_store.get_all_sessions_with_user(limit=limit)}


@router.delete("/sessions/{session_id}")
async def admin_delete_session(session_id: str, request: Request, admin: dict = Depends(require_admin)):
    """
    删除任意会话（内容治理）。

    与用户端"删除自己的会话"不同，此接口可删任何用户的会话，
    用于治理违规内容。删除后该 session_id 下的消息一并清除。
    """
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
    """
    最近审计日志（登录 / 注册 / 管理操作，含 IP）。

    参数:
        limit: 返回条数上限，默认 100，按时间倒序。

    日志来源:
        record_audit() 的全部调用点 —— 登录成功/失败、注册、
        以及本模块的所有写操作（更新/删除用户、删除会话）。
    """
    return {"logs": await user_store.get_audit_logs(limit=limit)}
