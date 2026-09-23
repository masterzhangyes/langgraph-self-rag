/**
 * 管理后台 — 运营统计 / 用户管理 / 会话审计 / 操作日志
 * =====================================================
 * 仅 admin 角色可见（入口由 TopBar 控制）。
 */
import { useState, useEffect, useCallback } from 'react'
import {
  X, RefreshCw, Users, MessageSquare, Database, ShieldCheck,
  UserCog, Trash2, KeyRound, Activity, History, MessagesSquare,
} from 'lucide-react'
import { apiJson } from '../api'
import { useToast } from './Toast'
import type { User } from '../hooks/useAuth'

interface AdminStats {
  total_users: number
  active_users: number
  admin_users: number
  new_users_today: number
  total_sessions: number
  total_messages: number
  active_users_today: number
  knowledge_bases?: { name: string; chunk_count: number; document_count: number }[]
}

interface AdminUserRow {
  id: number
  username: string
  email: string | null
  role: 'user' | 'admin'
  is_active: boolean
  created_at: string
  last_login_at: string | null
  session_count: number
}

interface AdminSessionRow {
  id: string
  title: string
  created_at: string
  updated_at: string
  user_id: number | null
  username: string | null
  message_count: number
}

interface AuditLogRow {
  id: number
  user_id: number | null
  username: string | null
  action: string
  detail: string | null
  ip: string | null
  created_at: string
}

interface AdminPanelProps {
  visible: boolean
  onClose: () => void
  currentUser: User
}

type Tab = 'stats' | 'users' | 'sessions' | 'logs'

const fmtTime = (t?: string | null) => (t ? t.replace('T', ' ').slice(0, 19) : '—')

export default function AdminPanel({ visible, onClose, currentUser }: AdminPanelProps) {
  const toast = useToast()
  const [tab, setTab] = useState<Tab>('stats')
  const [stats, setStats] = useState<AdminStats | null>(null)
  const [users, setUsers] = useState<AdminUserRow[]>([])
  const [sessions, setSessions] = useState<AdminSessionRow[]>([])
  const [logs, setLogs] = useState<AuditLogRow[]>([])
  const [loading, setLoading] = useState(false)

  const loadStats = useCallback(async () => {
    try {
      setStats(await apiJson<AdminStats>('/api/admin/stats'))
    } catch (e: any) {
      toast.error('加载统计失败：' + (e?.message || e))
    }
  }, [toast])

  const loadUsers = useCallback(async () => {
    try {
      const data = await apiJson<{ users: AdminUserRow[] }>('/api/admin/users')
      setUsers(data.users || [])
    } catch (e: any) {
      toast.error('加载用户失败：' + (e?.message || e))
    }
  }, [toast])

  const loadSessions = useCallback(async () => {
    try {
      const data = await apiJson<{ sessions: AdminSessionRow[] }>('/api/admin/sessions')
      setSessions(data.sessions || [])
    } catch (e: any) {
      toast.error('加载会话失败：' + (e?.message || e))
    }
  }, [toast])

  const loadLogs = useCallback(async () => {
    try {
      const data = await apiJson<{ logs: AuditLogRow[] }>('/api/admin/audit-logs')
      setLogs(data.logs || [])
    } catch (e: any) {
      toast.error('加载日志失败：' + (e?.message || e))
    }
  }, [toast])

  const refreshAll = useCallback(() => {
    setLoading(true)
    Promise.all([loadStats(), loadUsers(), loadSessions(), loadLogs()]).finally(() =>
      setLoading(false),
    )
  }, [loadStats, loadUsers, loadSessions, loadLogs])

  useEffect(() => {
    if (visible && stats === null) refreshAll()
  }, [visible, stats, refreshAll])

  // ── 用户操作 ──

  const patchUser = async (id: number, payload: Record<string, unknown>, desc: string) => {
    try {
      await apiJson(`/api/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(payload) })
      toast.success(desc)
      loadUsers()
      loadStats()
    } catch (e: any) {
      toast.error(e?.message || '操作失败')
    }
  }

  const toggleActive = (u: AdminUserRow) =>
    patchUser(u.id, { is_active: !u.is_active }, u.is_active ? '已禁用' : '已启用')

  const toggleRole = (u: AdminUserRow) =>
    patchUser(u.id, { role: u.role === 'admin' ? 'user' : 'admin' },
      u.role === 'admin' ? '已降级为普通用户' : '已提升为管理员')

  const resetPassword = async (u: AdminUserRow) => {
    const pwd = window.prompt(`为用户 ${u.username} 设置新密码（至少 6 位）：`)
    if (!pwd) return
    if (pwd.length < 6) return toast.error('密码至少 6 位')
    await patchUser(u.id, { password: pwd }, '密码已重置')
  }

  const removeUser = async (u: AdminUserRow) => {
    if (!window.confirm(`确认删除用户 ${u.username}？其全部会话与消息将一并删除，不可恢复。`)) return
    try {
      await apiJson(`/api/admin/users/${u.id}`, { method: 'DELETE' })
      toast.success('用户已删除')
      loadUsers()
      loadStats()
      loadSessions()
    } catch (e: any) {
      toast.error(e?.message || '删除失败')
    }
  }

  const removeSession = async (s: AdminSessionRow) => {
    if (!window.confirm(`确认删除会话「${s.title}」？`)) return
    try {
      await apiJson(`/api/admin/sessions/${s.id}`, { method: 'DELETE' })
      toast.success('会话已删除')
      loadSessions()
      loadStats()
    } catch (e: any) {
      toast.error(e?.message || '删除失败')
    }
  }

  if (!visible) return null

  const statCards = stats
    ? [
        { label: '用户总数', value: stats.total_users, sub: `今日新增 ${stats.new_users_today}`, icon: <Users size={18} /> },
        { label: '会话总数', value: stats.total_sessions, sub: `今日活跃用户 ${stats.active_users_today}`, icon: <MessagesSquare size={18} /> },
        { label: '消息总数', value: stats.total_messages, sub: `管理员 ${stats.admin_users} 人`, icon: <MessageSquare size={18} /> },
        {
          label: '知识库块数',
          value: (stats.knowledge_bases || []).reduce((a, k) => a + k.chunk_count, 0),
          sub: `${(stats.knowledge_bases || []).length} 个知识库`,
          icon: <Database size={18} />,
        },
      ]
    : []

  return (
    <div className="admin-overlay">
      <div className="admin-panel">
        <header className="admin-header">
          <div className="admin-title">
            <ShieldCheck size={18} />
            <h2>管理后台</h2>
            <span className="admin-role-badge">{currentUser.username}</span>
          </div>
          <div className="admin-actions">
            <button className="btn-icon" onClick={refreshAll} title="刷新全部数据">
              <RefreshCw size={16} className={loading ? 'spinning' : ''} />
            </button>
            <button className="btn-icon" onClick={onClose} title="关闭">
              <X size={18} />
            </button>
          </div>
        </header>

        <nav className="admin-tabs">
          <button className={`admin-tab ${tab === 'stats' ? 'active' : ''}`} onClick={() => setTab('stats')}>
            <Activity size={14} />
            运营概览
          </button>
          <button className={`admin-tab ${tab === 'users' ? 'active' : ''}`} onClick={() => setTab('users')}>
            <Users size={14} />
            用户管理
          </button>
          <button className={`admin-tab ${tab === 'sessions' ? 'active' : ''}`} onClick={() => setTab('sessions')}>
            <MessagesSquare size={14} />
            会话审计
          </button>
          <button className={`admin-tab ${tab === 'logs' ? 'active' : ''}`} onClick={() => setTab('logs')}>
            <History size={14} />
            操作日志
          </button>
        </nav>

        <div className="admin-body">
          {/* ── 运营概览 ── */}
          {tab === 'stats' && (
            <>
              <div className="admin-stat-grid">
                {statCards.map(c => (
                  <div className="admin-stat-card" key={c.label}>
                    <div className="admin-stat-icon">{c.icon}</div>
                    <div>
                      <div className="admin-stat-value">{c.value}</div>
                      <div className="admin-stat-label">{c.label}</div>
                      <div className="admin-stat-sub">{c.sub}</div>
                    </div>
                  </div>
                ))}
              </div>
              {stats?.knowledge_bases && stats.knowledge_bases.length > 0 && (
                <div className="admin-section">
                  <h3><Database size={13} /> 知识库规模</h3>
                  <table className="admin-table">
                    <thead>
                      <tr><th>名称</th><th>文本块</th><th>文档数</th></tr>
                    </thead>
                    <tbody>
                      {stats.knowledge_bases.map(kb => (
                        <tr key={kb.name}>
                          <td>{kb.name}</td>
                          <td>{kb.chunk_count}</td>
                          <td>{kb.document_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {/* ── 用户管理 ── */}
          {tab === 'users' && (
            <table className="admin-table">
              <thead>
                <tr>
                  <th>用户</th><th>角色</th><th>状态</th><th>会话</th><th>注册时间</th><th>最近登录</th><th>操作</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => {
                  const isSelf = u.id === currentUser.id
                  return (
                    <tr key={u.id} className={!u.is_active ? 'row-muted' : ''}>
                      <td>
                        <strong>{u.username}</strong>
                        {u.email && <span className="admin-email"> · {u.email}</span>}
                        {isSelf && <span className="admin-self-tag">（我）</span>}
                      </td>
                      <td>
                        <span className={`role-badge ${u.role}`}>{u.role === 'admin' ? '管理员' : '用户'}</span>
                      </td>
                      <td>
                        <span className={`status-badge ${u.is_active ? 'on' : 'off'}`}>
                          {u.is_active ? '正常' : '已禁用'}
                        </span>
                      </td>
                      <td>{u.session_count}</td>
                      <td>{fmtTime(u.created_at)}</td>
                      <td>{fmtTime(u.last_login_at)}</td>
                      <td className="admin-row-actions">
                        <button title="启用/禁用" onClick={() => toggleActive(u)} disabled={isSelf}>
                          <UserCog size={14} />
                        </button>
                        <button
                          title={u.role === 'admin' ? '降级为用户' : '提升为管理员'}
                          onClick={() => toggleRole(u)}
                          disabled={isSelf || (u.role === 'admin' && users.filter(x => x.role === 'admin' && x.is_active).length <= 1)}
                        >
                          <ShieldCheck size={14} />
                        </button>
                        <button title="重置密码" onClick={() => resetPassword(u)}>
                          <KeyRound size={14} />
                        </button>
                        <button
                          className="danger"
                          title="删除用户"
                          onClick={() => removeUser(u)}
                          disabled={isSelf || (u.role === 'admin' && users.filter(x => x.role === 'admin' && x.is_active).length <= 1)}
                        >
                          <Trash2 size={14} />
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}

          {/* ── 会话审计 ── */}
          {tab === 'sessions' && (
            <table className="admin-table">
              <thead>
                <tr><th>会话</th><th>归属用户</th><th>消息数</th><th>创建时间</th><th>最近活跃</th><th>操作</th></tr>
              </thead>
              <tbody>
                {sessions.map(s => (
                  <tr key={s.id}>
                    <td className="admin-session-title">{s.title}</td>
                    <td>
                      {s.username ? (
                        s.username
                      ) : (
                        <span className="admin-anonymous">匿名</span>
                      )}
                    </td>
                    <td>{s.message_count}</td>
                    <td>{fmtTime(s.created_at)}</td>
                    <td>{fmtTime(s.updated_at)}</td>
                    <td className="admin-row-actions">
                      <button className="danger" title="删除会话" onClick={() => removeSession(s)}>
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
                {sessions.length === 0 && (
                  <tr><td colSpan={6} className="admin-empty">暂无会话</td></tr>
                )}
              </tbody>
            </table>
          )}

          {/* ── 操作日志 ── */}
          {tab === 'logs' && (
            <table className="admin-table">
              <thead>
                <tr><th>时间</th><th>用户</th><th>事件</th><th>详情</th><th>IP</th></tr>
              </thead>
              <tbody>
                {logs.map(l => (
                  <tr key={l.id}>
                    <td>{fmtTime(l.created_at)}</td>
                    <td>{l.username || '—'}</td>
                    <td>
                      <span className={`audit-badge ${l.action.includes('failed') || l.action.includes('locked') ? 'warn' : l.action.startsWith('admin') ? 'info' : ''}`}>
                        {l.action}
                      </span>
                    </td>
                    <td className="admin-detail-cell">{l.detail || '—'}</td>
                    <td>{l.ip || '—'}</td>
                  </tr>
                ))}
                {logs.length === 0 && (
                  <tr><td colSpan={5} className="admin-empty">暂无日志</td></tr>
                )}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
