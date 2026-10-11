/**
 * 统一 API 请求封装
 * =================
 * · 自动附带 Authorization: Bearer <access_token>
 * · 收到 401 时清除本地令牌并广播事件（App 层监听后跳转登录页）
 * · 自动设置 JSON Content-Type（FormData 场景除外）
 */

const ACCESS_TOKEN_KEY = 'qa.access_token'
const REFRESH_TOKEN_KEY = 'qa.refresh_token'

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_TOKEN_KEY)
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_TOKEN_KEY)
}

export function setTokens(access: string, refresh: string) {
  localStorage.setItem(ACCESS_TOKEN_KEY, access)
  localStorage.setItem(REFRESH_TOKEN_KEY, refresh)
}

export function clearTokens() {
  localStorage.removeItem(ACCESS_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
}

export function isAuthenticated(): boolean {
  return !!getAccessToken()
}

/** 令牌失效事件名（App 层监听） */
export const UNAUTHORIZED_EVENT = 'qa:unauthorized'

/**
 * 🐛 修复: 静默续期单飞锁。
 * 此前任何 401 都直接清除双令牌 —— access token 到期（默认 30 分钟）后
 * 连仍有效的 refresh token 一并销毁，用户被强制登出，续期机制形同虚设。
 * 现在: 401 → 用 refresh token 续期（并发请求只发起一次）→ 重试原请求；
 *       仅当续期也失败（refresh 过期/无效）才清除令牌并广播登出。
 */
let refreshInFlight: Promise<boolean> | null = null

async function trySilentRefresh(): Promise<boolean> {
  const refresh = getRefreshToken()
  if (!refresh) return false
  try {
    // 用裸 fetch 而非 apiFetch，避免 401 重试递归
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refresh }),
    })
    if (!res.ok) return false
    const data = await res.json()
    if (data?.access_token && data?.refresh_token) {
      setTokens(data.access_token, data.refresh_token)
      return true
    }
    return false
  } catch {
    return false
  }
}

function buildInitWithAuth(path: string, init: RequestInit): RequestInit {
  const headers = new Headers(init.headers || {})
  const token = getAccessToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  return { ...init, headers }
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  let res = await fetch(path, buildInitWithAuth(path, init))

  if (res.status === 401) {
    // 单飞: 并发的多个 401 请求共享同一次续期
    refreshInFlight = refreshInFlight ?? trySilentRefresh()
    const refreshed = await refreshInFlight.finally(() => { refreshInFlight = null })
    if (refreshed) {
      // 续期成功 → 带新令牌重试原请求一次
      res = await fetch(path, buildInitWithAuth(path, init))
    } else {
      // 续期失败 → 真正的会话过期，清除并广播
      clearTokens()
      window.dispatchEvent(new Event(UNAUTHORIZED_EVENT))
    }
  }
  return res
}

/**
 * 解析 JSON 响应，失败时抛出可读的错误信息。
 *
 * 兼容三种后端 detail 形态:
 *   · 字符串          → 直接展示（业务错误，如「用户名已被占用」）
 *   · 校验错误数组(422) → 提取 pydantic 校验消息（去掉 "Value error," 前缀）
 *   · 其它             → 退化为「请求失败 (状态码)」
 */
export async function apiJson<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await apiFetch(path, init)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(extractErrorMessage(data, res.status))
  }
  return data as T
}

function extractErrorMessage(data: any, status: number): string {
  const detail = data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail.length > 0) {
    // pydantic 422: [{loc: [...], msg: "Value error, xxx"}, ...]
    const msgs = detail
      .map((d: any) => String(d?.msg || '').replace(/^Value error,\s*/, ''))
      .filter(Boolean)
    if (msgs.length > 0) return msgs[0]
  }
  return `请求失败 (${status})`
}
