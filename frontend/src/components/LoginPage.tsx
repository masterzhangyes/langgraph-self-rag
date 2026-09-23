/**
 * 登录 / 注册页 — 双卡片切换 + Aurora 主题
 */
import { useState, useCallback, type FormEvent } from 'react'
import { Sparkles, User as UserIcon, Lock, Mail, LogIn, UserPlus, ShieldCheck } from 'lucide-react'

interface LoginPageProps {
  onLogin: (username: string, password: string) => Promise<unknown>
  onRegister: (username: string, password: string, email?: string) => Promise<unknown>
  darkMode: boolean
  onToggleDark: () => void
}

export default function LoginPage({ onLogin, onRegister, darkMode, onToggleDark }: LoginPageProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const isRegister = mode === 'register'

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault()
      if (busy) return
      setError('')
      if (isRegister) {
        if (username.trim().length < 3) return setError('用户名至少 3 个字符（字母/数字/下划线/中文）')
        if (password.length < 6) return setError('密码至少 6 位')
      }
      setBusy(true)
      try {
        if (isRegister) await onRegister(username.trim(), password, email.trim() || undefined)
        else await onLogin(username.trim(), password)
      } catch (err: any) {
        setError(err?.message || '操作失败，请重试')
      } finally {
        setBusy(false)
      }
    },
    [busy, isRegister, username, password, email, onLogin, onRegister],
  )

  const switchMode = (m: 'login' | 'register') => {
    setMode(m)
    setError('')
  }

  return (
    <div className={`login-page ${darkMode ? 'dark' : ''}`}>
      <button className="login-theme-toggle" onClick={onToggleDark} title="切换主题">
        {darkMode ? '☀️' : '🌙'}
      </button>

      <div className="login-card">
        <div className="login-brand">
          <div className="brand-mark">
            <Sparkles size={20} />
          </div>
          <h1>智问</h1>
          <p>RAG 知识库智能问答平台</p>
        </div>

        <div className="login-tabs">
          <button
            className={`login-tab ${!isRegister ? 'active' : ''}`}
            onClick={() => switchMode('login')}
          >
            <LogIn size={14} />
            登录
          </button>
          <button
            className={`login-tab ${isRegister ? 'active' : ''}`}
            onClick={() => switchMode('register')}
          >
            <UserPlus size={14} />
            注册
          </button>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          <label className="login-field">
            <UserIcon size={15} className="login-field-icon" />
            <input
              type="text"
              placeholder="用户名"
              value={username}
              onChange={e => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </label>

          <label className="login-field">
            <Lock size={15} className="login-field-icon" />
            <input
              type="password"
              placeholder={isRegister ? '密码（至少 6 位）' : '密码'}
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete={isRegister ? 'new-password' : 'current-password'}
              required
            />
          </label>

          {isRegister && (
            <label className="login-field">
              <Mail size={15} className="login-field-icon" />
              <input
                type="email"
                placeholder="邮箱（可选）"
                value={email}
                onChange={e => setEmail(e.target.value)}
                autoComplete="email"
              />
            </label>
          )}

          {error && <p className="login-error">{error}</p>}

          <button className="login-submit" type="submit" disabled={busy}>
            {busy ? '请稍候…' : isRegister ? '创建账号' : '登 录'}
          </button>
        </form>

        <p className="login-hint">
          <ShieldCheck size={13} />
          {isRegister
            ? '首个注册的用户将自动成为管理员'
            : '密码使用 bcrypt 哈希存储，令牌采用 JWT 双令牌机制'}
        </p>
      </div>
    </div>
  )
}
