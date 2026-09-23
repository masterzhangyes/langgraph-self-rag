"""
用户认证模块 - JWT 双令牌认证体系
================================

路由前缀: /api/auth

设计要点（生产级实践）:
    · 密码存储    bcrypt 自适应哈希（随机盐 + cost=12，抗彩虹表 / 暴力破解）
    · 令牌体系    双 Token 无状态 JWT（access 30min + refresh 7d，HS256 签名）
    · 防爆破      基于 (IP, 用户名) 的滑动窗口失败计数，超限临时锁定
    · RBAC        user / admin 两级角色；管理端点经 require_admin 守卫
    · 审计        注册 / 登录 / 登录失败均写入 audit_logs，供管理后台回溯
    · 引导约定    首个注册用户自动成为管理员（避免部署后无管理员可用）

依赖注入（供其他路由复用）:
    get_current_user   要求登录，否则 401
    get_optional_user  未登录返回 None（会话按「匿名桶」隔离，向后兼容）
    require_admin      要求 admin 角色，否则 403
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Request

from core.config import settings
from infrastructure import user_store
from api.models import (
    RegisterRequest, LoginRequest, RefreshRequest,
    TokenResponse, UserResponse,
)

router = APIRouter(prefix="/api/auth", tags=["认证"])

# token type → 有效期（用于统一签发 / 校验）
_TOKEN_TTL = {
    "access": timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    "refresh": timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
}


# ═══════════════════════════════════════════════════════════════
# JWT 签发与校验
# ═══════════════════════════════════════════════════════════════

def _create_token(user_id: int, username: str, role: str, token_type: str) -> str:
    """
    签发 JWT。

    Claims:
        sub       用户 ID（JWT 标准 claim）
        username  冗余用户名，便于日志排查（不作为鉴权依据）
        role      登录时刻的角色快照（变更角色后需刷新令牌生效）
        type      access / refresh，防止两类令牌混用
        iat/exp   签发 / 过期时间（UTC 时间戳）
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "type": token_type,
        "iat": now,
        "exp": now + _TOKEN_TTL[token_type],
    }
    return pyjwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, expected_type: str) -> dict:
    """
    校验并解码 JWT。

    抛出:
        401 — 令牌无效 / 过期 / 类型不匹配
    """
    try:
        payload = pyjwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="令牌已过期，请刷新或重新登录")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效令牌")
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="令牌类型错误")
    return payload


def _issue_token_pair(user: dict) -> TokenResponse:
    """为用户签发 access + refresh 双令牌"""
    return TokenResponse(
        access_token=_create_token(user["id"], user["username"], user["role"], "access"),
        refresh_token=_create_token(user["id"], user["username"], user["role"], "refresh"),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(
            id=user["id"],
            username=user["username"],
            email=user.get("email"),
            role=user["role"],
            is_active=bool(user["is_active"]),
            created_at=user["created_at"],
            last_login_at=user.get("last_login_at"),
        ),
    )


# ═══════════════════════════════════════════════════════════════
# 依赖注入：当前用户 / 可选用户 / 管理员守卫
# ═══════════════════════════════════════════════════════════════

async def _load_user(request: Request) -> Optional[dict]:
    """
    从请求头解析 Bearer 令牌并加载用户。

    返回 None 的情况（不抛错，交由调用方决定语义）:
        · 未携带 Authorization 头
        · 令牌无效 / 过期
        · 用户已被删除或禁用（令牌签发后状态变更的场景）
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    try:
        payload = decode_token(auth_header[7:], "access")
    except HTTPException:
        return None

    user = await user_store.get_user_by_id(int(payload["sub"]))
    if not user or not user["is_active"]:
        return None
    return user


async def get_current_user(request: Request) -> dict:
    """要求已登录，否则 401（用于必须认证的端点）"""
    user = await _load_user(request)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="未登录或令牌已失效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_optional_user(request: Request) -> Optional[dict]:
    """
    可选认证：未登录返回 None。

    会话 / 对话等端点使用它 —
    登录用户的会话按 user_id 隔离，未登录请求落入匿名桶（向后兼容）。
    """
    return await _load_user(request)


async def require_admin(request: Request) -> dict:
    """要求管理员角色，否则 403（管理后台全部端点的守卫）"""
    user = await get_current_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


# ═══════════════════════════════════════════════════════════════
# 登录防爆破（滑动窗口失败计数）
# ═══════════════════════════════════════════════════════════════
#
# 以 (IP, 用户名) 为键，统计窗口期内的失败次数；
# 超过 LOGIN_MAX_ATTEMPTS 次后在 LOGIN_LOCKOUT_SECONDS 内直接拒绝。
# 说明: 进程内字典实现适用于单实例部署；多实例场景应换用 Redis 等共享存储。

_login_failures: dict[tuple, list] = {}


def _client_ip(request: Request) -> str:
    """获取客户端真实 IP（优先取反向代理透传头）"""
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")


def _is_locked_out(key: tuple) -> bool:
    """判断该键是否处于锁定状态，并顺手清理过期记录"""
    now = time.monotonic()
    failures = [t for t in _login_failures.get(key, []) if now - t < settings.LOGIN_LOCKOUT_SECONDS]
    _login_failures[key] = failures
    return len(failures) >= settings.LOGIN_MAX_ATTEMPTS


def _record_failure(key: tuple):
    _login_failures.setdefault(key, []).append(time.monotonic())


def _clear_failures(key: tuple):
    _login_failures.pop(key, None)


# ═══════════════════════════════════════════════════════════════
# 认证路由
# ═══════════════════════════════════════════════════════════════

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest, request: Request):
    """
    用户注册，成功后直接返回双令牌（免二次登录）。

    引导约定: 系统中不存在任何用户时，首个注册者自动成为 admin，
    之后注册的用户均为普通 user 角色。
    """
    if await user_store.get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="用户名已被占用")

    role = "admin" if await user_store.count_users() == 0 else "user"
    user = await user_store.create_user(req.username, req.password, req.email, role)

    await user_store.record_audit(
        user["id"], user["username"], "register",
        f"注册成为 {role}", _client_ip(request),
    )
    import logging
    logging.getLogger("qa.auth").info(
        "用户注册: %s (%s)%s", req.username, role,
        "，首个用户已自动授予管理员权限" if role == "admin" else "",
    )
    return _issue_token_pair(user)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request):
    """
    用户登录。

    安全策略:
        · (IP, 用户名) 维度防爆破限流
        · 用户不存在与密码错误返回统一提示（不泄露账号是否存在）
        · 被禁用账户返回 403
        · 成功 / 失败均写审计日志
    """
    ip = _client_ip(request)
    key = (ip, req.username)

    if _is_locked_out(key):
        await user_store.record_audit(
            None, req.username, "login_locked",
            f"失败次数过多，IP={ip} 已临时锁定", ip,
        )
        raise HTTPException(
            status_code=429,
            detail=f"尝试过于频繁，请 {settings.LOGIN_LOCKOUT_SECONDS // 60} 分钟后再试",
        )

    user = await user_store.get_user_by_username(req.username)
    if not user or not user_store.verify_password(req.password, user["password_hash"]):
        _record_failure(key)
        await user_store.record_audit(
            user["id"] if user else None, req.username, "login_failed",
            f"IP={ip}", ip,
        )
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not user["is_active"]:
        await user_store.record_audit(user["id"], user["username"], "login_blocked", "账户已被禁用", ip)
        raise HTTPException(status_code=403, detail="账户已被禁用，请联系管理员")

    _clear_failures(key)
    await user_store.update_last_login(user["id"])
    await user_store.record_audit(user["id"], user["username"], "login", f"IP={ip}", ip)
    return _issue_token_pair(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest):
    """
    刷新令牌 — 用 refresh_token 换取新的双令牌。

    说明: 无状态 JWT 下服务端不维护黑名单，刷新即「旧 refresh 作废、
    新对生效」由客户端替换存储实现；access token 短有效期控制泄露风险。
    """
    payload = decode_token(req.refresh_token, "refresh")
    user = await user_store.get_user_by_id(int(payload["sub"]))
    if not user or not user["is_active"]:
        raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
    return _issue_token_pair(user)


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    """获取当前登录用户信息（前端启动时用于恢复会话）"""
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user.get("email"),
        role=user["role"],
        is_active=bool(user["is_active"]),
        created_at=user["created_at"],
        last_login_at=user.get("last_login_at"),
    )


@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    """
    退出登录。

    无状态 JWT 下服务端不销毁令牌，客户端丢弃令牌即可；
    access token 最多存活 30 分钟，风险窗口有限。
    """
    return {"message": "已退出登录"}
