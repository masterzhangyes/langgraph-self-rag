/**
 * 认证 Hook — 登录态管理
 * =======================
 * · 启动时用本地 access token 调 /auth/me 恢复会话
 * · 登录 / 注册成功后保存双令牌（access + refresh）
 * · 提供静默刷新（refresh token 换新令牌对）
 */
import { useState, useCallback, useEffect, useRef } from 'react'
import { apiFetch, apiJson, setTokens, clearTokens, getAccessToken, getRefreshToken } from '../api'

export interface User {
  id: number
  username: string
  email: string | null
  role: 'user' | 'admin'
  is_active: boolean
  created_at: string
  last_login_at: string | null
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const userRef = useRef<User | null>(null)

  useEffect(() => { userRef.current = user }, [user])

  // 启动时恢复登录态
  useEffect(() => {
    let cancelled = false
    const restore = async () => {
      if (!getAccessToken()) {
        // 尝试用 refresh token 静默续期一次
        const refresh = getRefreshToken()
        if (refresh) {
          try {
            const data = await apiJson<{ access_token: string; refresh_token: string }>('/api/auth/refresh', {
              method: 'POST',
              body: JSON.stringify({ refresh_token: refresh }),
            })
            setTokens(data.access_token, data.refresh_token)
          } catch {
            clearTokens()
          }
        }
      }
      if (getAccessToken()) {
        try {
          const me = await apiJson<User>('/api/auth/me')
          if (!cancelled) setUser(me)
        } catch {
          if (!cancelled) clearTokens()
        }
      }
      if (!cancelled) setAuthLoading(false)
    }
    restore()
    return () => { cancelled = true }
  }, [])

  const login = useCallback(async (username: string, password: string): Promise<User> => {
    const data = await apiJson<{
      access_token: string
      refresh_token: string
      user: User
    }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    })
    setTokens(data.access_token, data.refresh_token)
    setUser(data.user)
    return data.user
  }, [])

  const register = useCallback(
    async (username: string, password: string, email?: string): Promise<User> => {
      const data = await apiJson<{
        access_token: string
        refresh_token: string
        user: User
      }>('/api/auth/register', {
        method: 'POST',
        body: JSON.stringify({ username, password, email: email || undefined }),
      })
      setTokens(data.access_token, data.refresh_token)
      setUser(data.user)
      return data.user
    },
    [],
  )

  const logout = useCallback(async () => {
    try {
      await apiFetch('/api/auth/logout', { method: 'POST' })
    } catch {
      /* ignore */
    }
    clearTokens()
    setUser(null)
  }, [])

  return { user, authLoading, login, register, logout }
}
