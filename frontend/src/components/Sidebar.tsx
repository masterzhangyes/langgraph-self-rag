import { useState } from 'react'
import type { MouseEvent } from 'react'
import { MessageSquare, Plus, Trash2, X } from 'lucide-react'
import type { SessionItem } from '../hooks/useChat'

interface SidebarProps {
  sessions: SessionItem[]
  activeSession: string | null
  visible: boolean
  onClose: () => void
  onNew: () => void
  onDelete: (id: string) => void
  onSelect: (id: string) => void
}

function formatTime(iso?: string): string {
  if (!iso) return ''
  const d = new Date(iso.replace(' ', 'T'))
  if (Number.isNaN(d.getTime())) return iso.slice(5, 16)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  const pad = (n: number) => String(n).padStart(2, '0')
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  if (sameDay) return hm
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hm}`
}

function SessionRow({
  s,
  active,
  onSelect,
  onDelete,
}: {
  s: SessionItem
  active: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  const [confirming, setConfirming] = useState(false)

  const handleDelete = (e: MouseEvent) => {
    e.stopPropagation()
    if (!confirming) {
      setConfirming(true)
      setTimeout(() => setConfirming(false), 2500)
      return
    }
    onDelete()
  }

  return (
    <div
      className={`session-item ${active ? 'active' : ''}`}
      onClick={onSelect}
      title={s.title}
    >
      <div className="session-row">
        <MessageSquare size={14} className="session-ico" />
        <div className="session-title">{s.title || '新对话'}</div>
        <button
          className={`session-del ${confirming ? 'confirm' : ''}`}
          onClick={handleDelete}
          title={confirming ? '再次点击确认删除' : '删除会话'}
        >
          {confirming ? <X size={12} /> : <Trash2 size={13} />}
        </button>
      </div>
      <div className="session-meta">
        <span>{s.message_count || 0} 条消息</span>
        <span>{formatTime(s.updated_at)}</span>
      </div>
    </div>
  )
}

export default function Sidebar({
  sessions,
  activeSession,
  visible,
  onClose,
  onNew,
  onDelete,
  onSelect,
}: SidebarProps) {
  return (
    <aside className={`sidebar ${visible ? 'open' : ''}`}>
      <div className="sidebar-header">
        <h2>
          <MessageSquare size={15} />
          对话记录
        </h2>
        <button className="btn-icon" onClick={onClose} aria-label="关闭">
          <X size={16} />
        </button>
      </div>

      <button className="btn-primary full-width" onClick={onNew}>
        <Plus size={15} />
        新建对话
      </button>

      <div className="session-list">
        {sessions.length === 0 ? (
          <div className="empty-hint">
            <MessageSquare size={22} className="empty-ico" />
            <p>暂无历史对话</p>
            <span>开始新对话后将自动保存</span>
          </div>
        ) : (
          sessions.map(s => (
            <SessionRow
              key={s.id}
              s={s}
              active={activeSession === s.id}
              onSelect={() => onSelect(s.id)}
              onDelete={() => onDelete(s.id)}
            />
          ))
        )}
      </div>
    </aside>
  )
}
