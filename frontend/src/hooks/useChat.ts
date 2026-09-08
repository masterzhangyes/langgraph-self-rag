import { useState, useCallback, useRef, useEffect } from 'react'

const API_BASE = '/api'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  sources?: SourceItem[]
}

export interface SourceItem {
  content: string
  source: string
  relevance_score: number
  metadata?: Record<string, unknown>
}

export interface SessionItem {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count?: number
}

const THINKING_PREFIX = '__THINKING__'
const isThinkingContent = (c: string) => c.startsWith(THINKING_PREFIX)

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false) // 正在请求 / 流式生成中
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [sessions, setSessions] = useState<SessionItem[]>([])

  const abortRef = useRef<AbortController | null>(null)
  const loadingRef = useRef(false)
  const messagesRef = useRef<ChatMessage[]>([])

  useEffect(() => {
    loadingRef.current = loading
  }, [loading])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`)
      if (!res.ok) return
      const data = await res.json()
      setSessions(data.sessions || [])
    } catch {
      /* ignore */
    }
  }, [])

  const loadSession = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}`)
      if (!res.ok) return
      const data = await res.json()
      const msgs: ChatMessage[] = (data.messages || [])
        .filter((m: any) => m.role === 'user' || m.role === 'assistant')
        .map((m: any) => {
          const msg: ChatMessage = { role: m.role, content: m.content }
          if (m.sources) {
            try {
              const parsed = JSON.parse(m.sources)
              if (Array.isArray(parsed)) msg.sources = parsed
            } catch {
              /* ignore */
            }
          }
          return msg
        })
      setMessages(msgs)
      setActiveSession(sessionId)
    } catch {
      /* ignore */
    }
  }, [])

  const newSession = useCallback(() => {
    abortRef.current?.abort()
    setMessages([])
    setActiveSession(null)
    setLoading(false)
  }, [])

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        const res = await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' })
        if (!res.ok) return false
        setSessions(prev => prev.filter(s => s.id !== sessionId))
        if (activeSession === sessionId) {
          setMessages([])
          setActiveSession(null)
        }
        return true
      } catch {
        return false
      }
    },
    [activeSession],
  )

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
    setLoading(false)
    // 将卡在“思考中”的占位内容收尾
    setMessages(prev => {
      const arr = [...prev]
      const last = arr[arr.length - 1]
      if (last?.role === 'assistant' && isThinkingContent(last.content)) {
        arr[arr.length - 1] = { ...last, content: '⏹ 已停止生成' }
      }
      return arr
    })
  }, [])

  /**
   * 流结束后从后端回填引用来源（SSE 只传文本，来源在落库后按会话取回）
   */
  const attachLatestSources = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}`)
      if (!res.ok) return
      const data = await res.json()
      const lastAssistant = [...(data.messages || [])]
        .reverse()
        .find((m: any) => m.role === 'assistant')
      if (!lastAssistant || !lastAssistant.sources) return

      let sources: SourceItem[] | undefined
      try {
        const parsed = JSON.parse(lastAssistant.sources)
        if (Array.isArray(parsed)) sources = parsed
      } catch {
        /* ignore */
      }
      if (!sources || sources.length === 0) return

      setMessages(prev => {
        const arr = [...prev]
        const last = arr[arr.length - 1]
        if (last?.role === 'assistant' && !last.sources) {
          arr[arr.length - 1] = { ...last, sources }
        }
        return arr
      })
    } catch {
      /* ignore */
    }
  }, [])

  const sendMessage = useCallback(
    async (query: string, activeKb: string, temperature = 0.3) => {
      const q = query.trim()
      if (!q || loadingRef.current) return false

      const userMsg: ChatMessage = { role: 'user', content: q }
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: `${THINKING_PREFIX}⏳ 正在思考中...`,
      }
      setMessages(prev => [...prev, userMsg, assistantMsg])
      setLoading(true)

      const controller = new AbortController()
      abortRef.current = controller
      let sid: string | null = activeSession

      // 历史上下文（剔除状态占位消息，取最近 20 条）
      const history = messagesRef.current
        .filter(m => !isThinkingContent(m.content))
        .slice(-20)
        .map(m => ({ role: m.role, content: m.content }))

      try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            history,
            kb_names: activeKb && activeKb !== 'default' ? [activeKb] : undefined,
            session_id: activeSession || undefined,
            temperature,
          }),
          signal: controller.signal,
        })

        if (!res.ok) {
          let detail = `请求失败 (${res.status})`
          try {
            const j = await res.json()
            if (j?.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
          } catch {
            /* ignore */
          }
          setMessages(prev => {
            const arr = [...prev]
            const last = arr[arr.length - 1]
            if (last?.role === 'assistant') {
              arr[arr.length - 1] = { ...last, content: `❌ ${detail}` }
            }
            return arr
          })
          return false
        }

        const reader = res.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let currentEvent = ''
        let hasRealContent = false

        const appendAssistant = (text: string) => {
          setMessages(prev => {
            const arr = [...prev]
            const last = arr[arr.length - 1]
            if (last?.role !== 'assistant') return arr
            const content = isThinkingContent(last.content) ? text : last.content + text
            arr[arr.length - 1] = { ...last, content }
            return arr
          })
        }

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7).trim()
              continue
            }
            if (!line.startsWith('data: ')) continue
            const data = line.slice(6).trimEnd()

            if (currentEvent === 'session') {
              if (data) {
                sid = data
                setActiveSession(data)
                fetchSessions()
              }
              currentEvent = ''
              continue
            }
            if (currentEvent === 'error') {
              setMessages(prev => {
                const arr = [...prev]
                const last = arr[arr.length - 1]
                if (last?.role === 'assistant') {
                  arr[arr.length - 1] = { ...last, content: `❌ 服务出错：${data}` }
                }
                return arr
              })
              currentEvent = ''
              continue
            }
            currentEvent = ''

            if (data === '[DONE]') continue

            // token 以 JSON 编码传输（换行等特殊字符安全）；同时兼容旧版纯文本协议
            let text: string | null = null
            try {
              const parsed = JSON.parse(data)
              if (typeof parsed === 'string') text = parsed
            } catch {
              /* not json */
            }
            if (text === null) text = data

            if (text.startsWith(THINKING_PREFIX)) {
              // 状态进度消息 → 整体替换而非追加
              setMessages(prev => {
                const arr = [...prev]
                const last = arr[arr.length - 1]
                if (last?.role === 'assistant') {
                  arr[arr.length - 1] = { ...last, content: text as string }
                }
                return arr
              })
              continue
            }

            hasRealContent = true
            appendAssistant(text)
          }
        }

        // 流正常结束 → 刷新会话列表（消息数/标题）并回填来源
        if (hasRealContent) {
          if (sid) await attachLatestSources(sid)
          fetchSessions()
        }
        return true
      } catch (err: any) {
        if (err?.name !== 'AbortError') {
          console.error('对话请求失败:', err)
          setMessages(prev => {
            const arr = [...prev]
            const last = arr[arr.length - 1]
            if (last?.role === 'assistant') {
              arr[arr.length - 1] = {
                ...last,
                content: '❌ 请求失败：无法连接服务，请确认后端已启动后重试。',
              }
            }
            return arr
          })
        }
        return false
      } finally {
        setLoading(false)
      }
    },
    [activeSession, attachLatestSources, fetchSessions],
  )

  return {
    messages,
    loading,
    activeSession,
    sessions,
    setMessages,
    setActiveSession,
    fetchSessions,
    loadSession,
    newSession,
    deleteSession,
    deleteCurrentSession: () => (activeSession ? deleteSession(activeSession) : newSession()),
    sendMessage,
    stopGeneration,
  }
}
