import { useState, useEffect, useCallback } from 'react'
import './App.css'

import { useChat } from './hooks/useChat'
import { ToastProvider } from './components/Toast'
import Sidebar from './components/Sidebar'
import KBPanel from './components/KBPanel'
import SettingsPanel from './components/SettingsPanel'
import TopBar from './components/TopBar'
import ChatArea from './components/ChatArea'
import InputArea from './components/InputArea'
import type { KBItem } from './components/KBPanel'

const API_BASE = '/api'

function App() {
  const {
    messages, loading, activeSession, sessions,
    fetchSessions, loadSession, newSession, deleteSession,
    sendMessage, stopGeneration,
  } = useChat()

  const [input, setInput] = useState('')
  const [showSources, setShowSources] = useState<number | null>(null)
  const [showKbPanel, setShowKbPanel] = useState(false)
  const [showSessionPanel, setShowSessionPanel] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  // ── 偏好持久化：暗色 / 温度 / 知识库 ──
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('qa.dark') === '1')
  const [temperature, setTemperature] = useState(() => {
    const v = parseFloat(localStorage.getItem('qa.temperature') || '')
    return Number.isFinite(v) ? v : 0.3
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
    localStorage.setItem('qa.dark', darkMode ? '1' : '0')
  }, [darkMode])

  useEffect(() => {
    localStorage.setItem('qa.temperature', String(temperature))
  }, [temperature])

  const [kbList, setKbList] = useState<KBItem[]>([])
  const [activeKb, setActiveKb] = useState(() => localStorage.getItem('qa.kb') || '')

  useEffect(() => {
    localStorage.setItem('qa.kb', activeKb)
  }, [activeKb])

  const fetchKBList = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/kb/list`)
      const data = await res.json()
      const filtered = (data.knowledge_bases || []).filter((k: KBItem) => k.name !== 'db')
      setKbList(filtered)
    } catch { /* ignore */ }
  }, [])

  // 知识库就绪后自动选中第一个可用的（保留用户上次选择）
  useEffect(() => {
    if (kbList.length === 0) return
    if (!kbList.some(k => k.name === activeKb)) {
      setActiveKb(kbList[0].name)
    }
  }, [kbList, activeKb])

  useEffect(() => {
    fetchSessions()
    fetchKBList()
  }, [fetchSessions, fetchKBList])

  const closeAllPanels = useCallback(() => {
    setShowSessionPanel(false)
    setShowKbPanel(false)
    setShowSettings(false)
  }, [])

  // Esc 关闭面板
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeAllPanels()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [closeAllPanels])

  const handleSend = useCallback(() => {
    if (loading || !input.trim()) return
    const q = input
    setInput('')
    sendMessage(q, activeKb, temperature)
  }, [input, activeKb, temperature, loading, sendMessage])

  // 欢迎页快捷问题：直接发送
  const handleQuickAsk = useCallback(
    (question: string) => {
      sendMessage(question, activeKb, temperature)
    },
    [sendMessage, activeKb, temperature],
  )

  const currentKb = kbList.find(k => k.name === activeKb) || { status: 'empty', chunk_count: 0, document_count: 0 }

  return (
    <ToastProvider>
      <div className={`app-container ${darkMode ? 'dark-mode' : ''}`}>
        {/* 左侧：对话记录 */}
        <Sidebar
          sessions={sessions}
          activeSession={activeSession}
          visible={showSessionPanel}
          onClose={() => setShowSessionPanel(false)}
          onNew={newSession}
          onDelete={deleteSession}
          onSelect={(id: string) => {
            loadSession(id)
            setShowSessionPanel(false)
          }}
        />

        {/* 左侧：知识库管理 */}
        <KBPanel
          kbList={kbList}
          activeKb={activeKb}
          setActiveKb={setActiveKb}
          visible={showKbPanel}
          onClose={() => setShowKbPanel(false)}
          onKBListRefresh={fetchKBList}
        />

        {/* 右侧：设置面板 */}
        <SettingsPanel
          temperature={temperature}
          setTemperature={setTemperature}
          darkMode={darkMode}
          setDarkMode={setDarkMode}
          visible={showSettings}
          onClose={() => setShowSettings(false)}
        />

        {/* 遮罩层：点击关闭所有面板 */}
        {(showSessionPanel || showKbPanel || showSettings) && (
          <div className="overlay" onClick={closeAllPanels} />
        )}

        <main className="main-area">
          <TopBar
            currentKb={currentKb}
            activeKb={activeKb}
            activeSession={activeSession}
            onToggleSessions={() => {
              if (showSessionPanel) { setShowSessionPanel(false); return }
              closeAllPanels(); setShowSessionPanel(true)
            }}
            onToggleKB={() => {
              if (showKbPanel) { setShowKbPanel(false); return }
              closeAllPanels(); setShowKbPanel(true)
            }}
            onToggleSettings={() => {
              if (showSettings) { setShowSettings(false); return }
              closeAllPanels(); setShowSettings(true)
            }}
            onNewSession={newSession}
            onDeleteSession={() => {
              if (activeSession) deleteSession(activeSession)
              else newSession()
            }}
          />

          <ChatArea
            messages={messages}
            showSources={showSources}
            setShowSources={setShowSources}
            loading={loading}
            onOpenKB={() => { closeAllPanels(); setShowKbPanel(true) }}
            onQuickAsk={handleQuickAsk}
          />

          <InputArea
            input={input}
            setInput={setInput}
            loading={loading}
            currentKb={currentKb}
            activeKb={activeKb}
            onSend={handleSend}
            onStop={stopGeneration}
          />
        </main>
      </div>
    </ToastProvider>
  )
}

export default App
