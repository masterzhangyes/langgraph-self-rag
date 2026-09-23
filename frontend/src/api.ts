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

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {})
  const token = getAccessToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const res = await fetch(path, { ...init, headers })
  if (res.status === 401) {
    clearTokens()
    window.dispatchEvent(new Event(UNAUTHORIZED_EVENT))
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
