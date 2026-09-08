import { Menu, Library, Plus, Sparkles, Trash2, Settings } from 'lucide-react'

interface KBInfo {
  status: string
  chunk_count: number
  document_count?: number
  name?: string
}

interface TopBarProps {
  currentKb: KBInfo
  activeKb: string
  activeSession: string | null
  onToggleSessions: () => void
  onToggleKB: () => void
  onToggleSettings: () => void
  onNewSession: () => void
  onDeleteSession: () => void
}

export default function TopBar({
  currentKb,
  activeKb,
  activeSession,
  onToggleSessions,
  onToggleKB,
  onToggleSettings,
  onNewSession,
  onDeleteSession,
}: TopBarProps) {
  const kbReady = currentKb.status === 'active' || (currentKb.chunk_count ?? 0) > 0

  return (
    <header className="top-bar">
      <div className="top-left">
        <button className="btn-icon" onClick={onToggleSessions} title="对话记录" aria-label="对话记录">
          <Menu size={19} />
        </button>
        <button className="btn-icon" onClick={onToggleKB} title="知识库管理" aria-label="知识库管理">
          <Library size={19} />
        </button>
        <button
          className="btn-icon btn-new-chat"
          onClick={onNewSession}
          title="新建对话 (⌘N)"
          aria-label="新建对话"
        >
          <Plus size={19} />
        </button>
      </div>

      <div className="brand">
        <div className="brand-mark">
          <Sparkles size={17} />
        </div>
        <div className="brand-meta">
          <h1 className="brand-title">智问</h1>
          <span className="brand-tag">RAG 知识库问答</span>
        </div>
      </div>

      <div className="top-right">
        <span className={`kb-indicator ${currentKb.status}`} title={`知识库：${activeKb || 'default'}`}>
          <span className="kb-dot" />
          {kbReady ? `RAG · ${activeKb || 'default'}` : '通用模式'}
        </span>
        {activeSession && (
          <button className="btn-danger-text" onClick={onDeleteSession} title="删除当前对话">
            <Trash2 size={14} />
            删除对话
          </button>
        )}
        <button className="btn-icon" onClick={onToggleSettings} title="设置" aria-label="设置">
          <Settings size={19} />
        </button>
      </div>
    </header>
  )
}
