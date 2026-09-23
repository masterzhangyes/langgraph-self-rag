import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import {
  ChevronDown,
  Database,
  FileText,
  Globe,
  Link2,
  Plus,
  RefreshCw,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react'
import { useToast } from './Toast'
import { apiFetch } from '../api'

const API_BASE = '/api'

export interface KBItem {
  name: string
  status: string
  chunk_count: number
  document_count?: number
  /** 是否公共知识库（所有人可读，仅管理员可写） */
  is_public?: boolean
  /** 所有者用户名（公共库为 null） */
  owner?: string | null
}

/** 单个文档条目（按 source 聚合） */
interface KBDocItem {
  source: string
  chunk_count: number
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

  // ── 面板状态 ──
  const [tab, setTab] = useState<'docs' | 'web'>('docs')
  const [busy, setBusy] = useState(false) // 上传 / 网页导入中
  const [creating, setCreating] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [clearConfirm, setClearConfirm] = useState(false)
  const [dangerOpen, setDangerOpen] = useState(false)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // ── 文档列表状态 ──
  const [docs, setDocs] = useState<KBDocItem[]>([])
  const [docsLoading, setDocsLoading] = useState(false)
  const [deletingSource, setDeletingSource] = useState<string | null>(null)

  // ── 网页批量导入状态 ──
  const [webUrls, setWebUrls] = useState('')

  const targetKb = activeKb || 'default'
  const kbReady = currentKb.status === 'active' || (currentKb.chunk_count ?? 0) > 0

  // ── 文档列表加载 ──
  const fetchDocs = useCallback(async () => {
    if (!visible) return
    setDocsLoading(true)
    try {
      const res = await apiFetch(`${API_BASE}/kb/${encodeURIComponent(targetKb)}/documents`)
      if (res.ok) {
        const data = await res.json()
        setDocs(data.documents || [])
      } else {
        setDocs([])
      }
    } catch {
      setDocs([])
    } finally {
      setDocsLoading(false)
    }
  }, [targetKb, visible])

  useEffect(() => {
    fetchDocs()
  }, [fetchDocs])

  // ── 上传文档 ──
  const handleFileUpload = async (files: FileList | File[]) => {
    if (!files.length) return
    setBusy(true)
    try {
      const formData = new FormData()
      for (const f of Array.from(files)) formData.append('files', f)
      formData.append('kb_name', targetKb)
      const res = await apiFetch(`${API_BASE}/kb/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '上传失败')
      toast.success(`已处理 ${data.file_count} 个文件，共 ${data.chunk_count} 个文本块`)
      onKBListRefresh()
      fetchDocs()
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

  // ── 删除单个文档（二次确认） ──
  const handleDeleteDoc = async (source: string) => {
    if (deletingSource !== source) {
      // 第一次点击：进入该行的确认态，3 秒未确认自动复位
      setDeletingSource(source)
      setTimeout(() => {
        setDeletingSource(prev => (prev === source ? null : prev))
      }, 3000)
      return
    }
    setDeletingSource(null)
    try {
      const res = await apiFetch(
        `${API_BASE}/kb/${encodeURIComponent(targetKb)}/documents?source=${encodeURIComponent(source)}`,
        { method: 'DELETE' },
      )
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '删除失败')
      toast.success(data.message || `已删除 [${source}]`)
      fetchDocs()
      onKBListRefresh()
    } catch (err: any) {
      toast.error('删除失败：' + (err?.message || err))
    }
  }

  // ── 网页批量导入 ──
  const handleWebLoad = async () => {
    // 多行输入，每行一个 URL；去空行、去重
    const urls = Array.from(
      new Set(
        webUrls
          .split('\n')
          .map(u => u.trim())
          .filter(u => u.length > 0),
      ),
    )
    if (urls.length === 0) return
    setBusy(true)
    try {
      const res = await apiFetch(`${API_BASE}/kb/load-web`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls, kb_name: targetKb }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '加载失败')
      toast.success(`已导入 ${urls.length} 个网页，共 ${data.chunk_count} 个文本块`)
      setWebUrls('')
      onKBListRefresh()
      fetchDocs()
    } catch (err: any) {
      toast.error('加载失败：' + (err?.message || err))
    } finally {
      setBusy(false)
    }
  }

  // ── 新建知识库（真 API） ──
  const handleCreateKb = async () => {
    const name = newKbName.trim()
    if (!name) return
    if (name === 'db' || name === 'default') {
      toast.error('名称不能使用保留字 "db" 或 "default"')
      return
    }
    try {
      const res = await apiFetch(`${API_BASE}/kb/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '创建失败')
      setActiveKb(name)
      setNewKbName('')
      setCreating(false)
      onKBListRefresh()
      toast.success(data.message || `知识库 [${name}] 已创建`)
    } catch (err: any) {
      toast.error('创建失败：' + (err?.message || err))
    }
  }

  // ── 清空知识库（二次确认） ──
  const handleClear = async () => {
    if (!clearConfirm) {
      setClearConfirm(true)
      setTimeout(() => setClearConfirm(false), 3000)
      return
    }
    setClearConfirm(false)
    try {
      const res = await apiFetch(`${API_BASE}/kb/${targetKb}/clear`, { method: 'DELETE' })
      if (!res.ok) throw new Error((await res.json()).detail || '清空失败')
      const remaining = kbList.filter(k => k.name !== targetKb)
      if (remaining.length > 0) setActiveKb(remaining[0].name)
      onKBListRefresh()
      setDocs([])
      toast.info('知识库已清空')
    } catch (err: any) {
      toast.error('清空失败：' + (err?.message || err))
    }
  }

  return (
    <aside className={`sidebar kb-sidebar ${visible ? 'open' : ''}`}>
      <div className="sidebar-header">
        <h2>
          <Database size={15} />
          知识库
        </h2>
        <button
          className="btn-icon"
          onClick={() => {
            onKBListRefresh()
            fetchDocs()
          }}
          aria-label="刷新"
          title="刷新列表与文档"
        >
          <RefreshCw size={14} />
        </button>
        <button className="btn-icon" onClick={onClose} aria-label="关闭">
          <X size={16} />
        </button>
      </div>

      {/* ── 当前知识库概览（紧凑） ── */}
      <div className="kb-info">
        <div className="kb-name">
          {activeKb || 'default'}
          {currentKb.is_public ? (
            <span className="kb-scope-tag public" title="所有人可读，仅管理员可写">公共</span>
          ) : (
            <span className="kb-scope-tag private" title="仅你和管理员可见">个人</span>
          )}
        </div>
        <div className="kb-meta-row">
          <span className={`kb-badge ${kbReady ? 'active' : currentKb.status}`}>
            <span className="kb-dot" />
            {kbReady ? '已就绪' : currentKb.status === 'loading' ? '索引中…' : '未加载'}
          </span>
          <span className="kb-meta-stats">
            {currentKb.document_count ?? 0} 文档 · {currentKb.chunk_count ?? 0} 块
          </span>
        </div>
      </div>

      {/* ── 切换 / 新建（一个区块收拢） ── */}
      <div className="kb-section">
        <div className="kb-switch-row">
          {kbList.length > 0 ? (
            <select
              className="kb-select"
              value={kbList.some(k => k.name === activeKb) ? activeKb : ''}
              onChange={e => setActiveKb(e.target.value)}
              title="切换知识库"
            >
              {kbList.map(kb => (
                <option key={`${kb.name}-${kb.is_public ? 'pub' : 'pvt'}`} value={kb.name}>
                  {kb.name}（{kb.chunk_count} 块）{kb.is_public ? ' · 公共' : ''}
                  {!kb.is_public && kb.owner ? ` · ${kb.owner}` : ''}
                </option>
              ))}
            </select>
          ) : (
            <span className="hint">暂无知识库</span>
          )}
          {!creating ? (
            <button className="btn-icon kb-add-btn" onClick={() => setCreating(true)} title="新建知识库">
              <Plus size={14} />
            </button>
          ) : (
            <button className="btn-icon kb-add-btn" onClick={() => setCreating(false)} title="取消新建">
              <X size={14} />
            </button>
          )}
        </div>

        {creating && (
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
          </div>
        )}
        {!kbList.some(k => k.name === targetKb) && (
          <p className="hint warn">当前知识库未注册，上传文档时将自动创建</p>
        )}
      </div>

      {/* ── Tab 切换：文档 / 网页 ── */}
      <div className="kb-tabs" role="tablist">
        <button
          className={`kb-tab ${tab === 'docs' ? 'active' : ''}`}
          onClick={() => setTab('docs')}
          role="tab"
          aria-selected={tab === 'docs'}
        >
          <FileText size={13} />
          文档
        </button>
        <button
          className={`kb-tab ${tab === 'web' ? 'active' : ''}`}
          onClick={() => setTab('web')}
          role="tab"
          aria-selected={tab === 'web'}
        >
          <Globe size={13} />
          网页
        </button>
      </div>

      <div className="kb-tab-body">
        {/* ═══ 文档 Tab ═══ */}
        {tab === 'docs' && (
          <>
            <div className="doc-list">
              {docsLoading ? (
                <p className="hint doc-empty">加载文档列表中…</p>
              ) : docs.length === 0 ? (
                <p className="hint doc-empty">
                  {kbReady ? '暂无文档记录（可能为旧版导入）' : '知识库为空，先上传文档吧'}
                </p>
              ) : (
                docs.map(d => (
                  <div
                    key={d.source}
                    className={`doc-item ${deletingSource === d.source ? 'confirming' : ''}`}
                  >
                    {d.source.startsWith('http') ? <Link2 size={13} /> : <FileText size={13} />}
                    <span className="doc-name" title={d.source}>
                      {d.source}
                    </span>
                    <span className="doc-chunks">{d.chunk_count} 块</span>
                    <button
                      className="doc-del"
                      onClick={() => handleDeleteDoc(d.source)}
                      title={deletingSource === d.source ? '再次点击确认删除' : '删除该文档'}
                    >
                      <Trash2 size={12} />
                      {deletingSource === d.source && <span>确认</span>}
                    </button>
                  </div>
                ))
              )}
            </div>

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
              <span>
                {busy ? '处理中…' : dragging ? '松开以添加' : '点击选择文件 或 拖拽到此处'}
              </span>
              <span className="hint">TXT / PDF / CSV / MD / DOCX</span>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".txt,.pdf,.csv,.md,.docx"
                onChange={onPickFile}
                hidden
              />
            </div>
          </>
        )}

        {/* ═══ 网页 Tab ═══ */}
        {tab === 'web' && (
          <>
            <textarea
              className="web-urls-input"
              placeholder={'每行一个网址，例如：\nhttps://example.com/article1\nhttps://example.com/article2'}
              value={webUrls}
              onChange={e => setWebUrls(e.target.value)}
              rows={5}
              disabled={busy}
            />
            <button
              className="btn-upload web-load-btn"
              onClick={handleWebLoad}
              disabled={busy || webUrls.trim().length === 0}
            >
              <Globe size={14} />
              {busy ? '抓取中…' : '批量导入'}
            </button>
            <p className="hint">抓取网页正文并切分入库，单页失败不影响其他页</p>
          </>
        )}
      </div>

      {/* ── 危险操作（默认折叠） ── */}
      <div className="kb-section danger-zone">
        <button className="danger-toggle" onClick={() => setDangerOpen(o => !o)}>
          <span>危险操作</span>
          <ChevronDown size={14} className={`chev ${dangerOpen ? 'up' : ''}`} />
        </button>
        {dangerOpen && (
          <>
            <button className={`btn-danger ${clearConfirm ? 'confirm' : ''}`} onClick={handleClear}>
              <Trash2 size={13} />
              {clearConfirm ? '再次点击确认清空' : `清空 [${targetKb}]`}
            </button>
            <p className="hint">清空将删除该知识库的全部索引，且不可恢复。</p>
          </>
        )}
      </div>
    </aside>
  )
}
