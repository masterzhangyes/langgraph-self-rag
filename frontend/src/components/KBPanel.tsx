import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import {
  Database,
  FileText,
  Globe,
  Layers,
  Plus,
  RefreshCw,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react'
import { useToast } from './Toast'

const API_BASE = '/api'

export interface KBItem {
  name: string
  status: string
  chunk_count: number
  document_count?: number
}

interface KBPanelProps {
  kbList: KBItem[]
  activeKb: string
  setActiveKb: (name: string) => void
  visible: boolean
  onClose: () => void
  onKBListRefresh: () => void
}

export default function KBPanel({
  kbList,
  activeKb,
  setActiveKb,
  visible,
  onClose,
  onKBListRefresh,
}: KBPanelProps) {
  const currentKb = kbList.find(k => k.name === activeKb) || {
    status: 'empty',
    chunk_count: 0,
    document_count: 0,
  }
  const toast = useToast()

  const [busy, setBusy] = useState(false) // 上传 / 加载网页中
  const [webUrl, setWebUrl] = useState('')
  const [creating, setCreating] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [clearConfirm, setClearConfirm] = useState(false)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const targetKb = activeKb || 'default'

  const handleFileUpload = async (files: FileList | File[]) => {
    if (!files.length) return
    setBusy(true)
    try {
      const formData = new FormData()
      for (const f of Array.from(files)) formData.append('files', f)
      formData.append('kb_name', targetKb)
      const res = await fetch(`${API_BASE}/kb/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '上传失败')
      toast.success(`已处理 ${data.file_count} 个文件，共 ${data.chunk_count} 个文本块`)
      onKBListRefresh()
    } catch (err: any) {
      toast.error('上传失败：' + (err?.message || err))
    } finally {
      setBusy(false)
    }
  }

  const onPickFile = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) handleFileUpload(e.target.files)
    e.target.value = ''
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files?.length) handleFileUpload(e.dataTransfer.files)
  }

  const handleWebLoad = async () => {
    const url = webUrl.trim()
    if (!url) return
    setBusy(true)
    try {
      const res = await fetch(`${API_BASE}/kb/load-web`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls: [url], kb_name: targetKb }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '加载失败')
      toast.success(`加载成功！共 ${data.chunk_count} 个文本块`)
      setWebUrl('')
      onKBListRefresh()
    } catch (err: any) {
      toast.error('加载失败：' + (err?.message || err))
    } finally {
      setBusy(false)
    }
  }

  const handleCreateKb = () => {
    const name = newKbName.trim()
    if (!name) return
    if (name === 'db' || name === 'default') {
      toast.error('名称不能使用保留字 "db" 或 "default"')
      return
    }
    setActiveKb(name)
    setNewKbName('')
    setCreating(false)
    onKBListRefresh()
    toast.info(`知识库 [${name}] 已就绪，上传文档后即可检索`)
  }

  const handleClear = async () => {
    if (!clearConfirm) {
      setClearConfirm(true)
      setTimeout(() => setClearConfirm(false), 3000)
      return
    }
    setClearConfirm(false)
    try {
      const res = await fetch(`${API_BASE}/kb/${targetKb}/clear`, { method: 'DELETE' })
      if (!res.ok) throw new Error((await res.json()).detail || '清空失败')
      const remaining = kbList.filter(k => k.name !== targetKb)
      if (remaining.length > 0) setActiveKb(remaining[0].name)
      onKBListRefresh()
      toast.info('知识库已清空')
    } catch (err: any) {
      toast.error('清空失败：' + (err?.message || err))
    }
  }

  const kbReady = currentKb.status === 'active' || (currentKb.chunk_count ?? 0) > 0

  return (
    <aside className={`sidebar kb-sidebar ${visible ? 'open' : ''}`}>
      <div className="sidebar-header">
        <h2>
          <Database size={15} />
          知识库
        </h2>
        <button className="btn-icon" onClick={onClose} aria-label="关闭">
          <X size={16} />
        </button>
      </div>

      {/* 当前知识库概览 */}
      <div className="kb-info">
        <div className="kb-badge-row">
          <span className={`kb-badge ${currentKb.status}`}>
            <span className="kb-dot" />
            {kbReady ? '已就绪' : currentKb.status === 'loading' ? '索引中…' : '未加载'}
          </span>
        </div>
        <div className="kb-name">{activeKb || 'default'}</div>
        <div className="kb-stats-grid">
          <div className="kb-stat">
            <span className="stat-num">{currentKb.chunk_count ?? 0}</span>
            <span className="stat-label">文本块</span>
          </div>
          <div className="kb-stat">
            <span className="stat-num">{currentKb.document_count ?? 0}</span>
            <span className="stat-label">文档</span>
          </div>
        </div>
      </div>

      {/* 切换 / 新建 */}
      <div className="kb-section">
        <h3>
          <Layers size={13} />
          知识库切换
        </h3>
        {kbList.length > 0 ? (
          <select
            className="kb-select"
            value={kbList.some(k => k.name === activeKb) ? activeKb : ''}
            onChange={e => setActiveKb(e.target.value)}
          >
            {kbList.map(kb => (
              <option key={kb.name} value={kb.name}>
                {kb.name}（{kb.chunk_count} 块）
              </option>
            ))}
          </select>
        ) : (
          <p className="hint">暂无知识库，可上传文档自动创建 default，或新建一个。</p>
        )}

        {!creating ? (
          <button className="btn-text" onClick={() => setCreating(true)}>
            <Plus size={12} />
            新建知识库
          </button>
        ) : (
          <div className="kb-create">
            <input
              type="text"
              placeholder="输入知识库名称…"
              value={newKbName}
              onChange={e => setNewKbName(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') handleCreateKb()
                if (e.key === 'Escape') setCreating(false)
              }}
              autoFocus
            />
            <button className="btn-small primary" onClick={handleCreateKb} disabled={!newKbName.trim()}>
              创建
            </button>
            <button className="btn-small" onClick={() => setCreating(false)}>
              取消
            </button>
          </div>
        )}
        <button className="btn-refresh" onClick={onKBListRefresh} title="刷新列表">
          <RefreshCw size={12} />
          刷新
        </button>
      </div>

      {/* 上传文档 */}
      <div className="kb-section">
        <h3>
          <UploadCloud size={13} />
          上传文档
        </h3>
        <p className="hint">支持 TXT / PDF / CSV / MD / DOCX，可多选或拖拽</p>
        <div
          className={`dropzone ${dragging ? 'dragging' : ''} ${busy ? 'busy' : ''}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <UploadCloud size={22} />
          <span>{busy ? '处理中…' : dragging ? '松开以添加' : '点击选择文件 或 拖拽到此处'}</span>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".txt,.pdf,.csv,.md,.docx"
            onChange={onPickFile}
            hidden
          />
        </div>
        {!kbList.some(k => k.name === targetKb) && (
          <p className="hint warn">将自动创建/写入 [default] 知识库</p>
        )}
      </div>

      {/* 加载网页 */}
      <div className="kb-section">
        <h3>
          <Globe size={13} />
          加载网页
        </h3>
        <div className="web-input-group">
          <input
            type="url"
            placeholder="输入网页 URL…"
            value={webUrl}
            onChange={e => setWebUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleWebLoad()}
          />
          <button onClick={handleWebLoad} disabled={busy || !webUrl.trim()}>
            <FileText size={13} />
            加载
          </button>
        </div>
      </div>

      {/* 危险操作 */}
      <div className="kb-section danger-zone">
        <h3>危险操作</h3>
        <button className={`btn-danger ${clearConfirm ? 'confirm' : ''}`} onClick={handleClear}>
          <Trash2 size={13} />
          {clearConfirm ? '再次点击确认清空' : `清空 [${targetKb}]`}
        </button>
        <p className="hint">清空将删除该知识库的全部索引，且不可恢复。</p>
      </div>
    </aside>
  )
}
