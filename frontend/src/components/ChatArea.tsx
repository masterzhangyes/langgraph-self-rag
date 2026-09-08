import { memo, useEffect, useRef, useState, Children, isValidElement } from 'react'
import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Check, Copy, FileText, Library, Sparkles, Globe, Send } from 'lucide-react'
import type { ChatMessage, SourceItem } from '../hooks/useChat'

interface ChatAreaProps {
  messages: ChatMessage[]
  showSources: number | null
  setShowSources: (idx: number | null) => void
  loading: boolean
  onOpenKB: () => void
  onQuickAsk: (question: string) => void
}

export default function ChatArea({
  messages,
  showSources,
  setShowSources,
  loading,
  onOpenKB,
  onQuickAsk,
}: ChatAreaProps) {
  const areaRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  const handleScroll = () => {
    const el = areaRef.current
    if (!el) return
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 90
  }

  // 智能滚动：生成中贴近底部持续跟随；用户上翻时不打扰
  useEffect(() => {
    const el = areaRef.current
    if (!el) return
    if (!stickToBottom.current) return
    bottomRef.current?.scrollIntoView({ behavior: loading ? 'auto' : 'smooth', block: 'end' })
  }, [messages, loading])

  if (messages.length === 0) {
    return (
      <div className="chat-area" ref={areaRef}>
        <Welcome onOpenKB={onOpenKB} onQuickAsk={onQuickAsk} />
      </div>
    )
  }

  return (
    <div className="chat-area" ref={areaRef} onScroll={handleScroll}>
      <div className="messages">
        {messages.map((msg, idx) => (
          <MessageBubble
            key={idx}
            msg={msg}
            index={idx}
            showSources={showSources}
            setShowSources={setShowSources}
            streaming={loading && idx === messages.length - 1 && msg.role === 'assistant'}
          />
        ))}
        <div ref={bottomRef} style={{ height: 1 }} />
      </div>
    </div>
  )
}

/* ─── 欢迎页 ─── */
const QUICK_PROMPTS = [
  '什么是 RAG？它如何工作？',
  '帮我总结知识库中的核心要点',
  '检索增强生成有什么优势？',
]

/* 个性化打字机标语 */
const TAGLINES = [
  '混合检索 + Self-RAG Agent，答案有理有据',
  '上传你的文档，它就成为你的专属行业专家',
  '多知识库 · 流式输出 · 来源可溯，安心可靠',
  '不止是聊天，更是一台知识挖掘引擎',
]

function useTypewriter(lines: string[]) {
  const [idx, setIdx] = useState(0)
  const [text, setText] = useState('')
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    const current = lines[idx % lines.length] ?? ''
    let timer: ReturnType<typeof setTimeout> | undefined

    if (!deleting && text === current) {
      timer = setTimeout(() => setDeleting(true), 2200)
    } else if (deleting && text === '') {
      setDeleting(false)
      setIdx(i => (i + 1) % lines.length)
    } else {
      const speed = deleting ? 18 : 52
      timer = setTimeout(() => {
        setText(deleting ? current.slice(0, text.length - 1) : current.slice(0, text.length + 1))
      }, speed)
    }
    return () => clearTimeout(timer)
  }, [text, deleting, idx, lines])

  return text
}

/* 按时段问候 */
function getGreeting(): string {
  const h = new Date().getHours()
  if (h < 5) return '夜深了'
  if (h < 9) return '早上好'
  if (h < 12) return '上午好'
  if (h < 14) return '中午好'
  if (h < 18) return '下午好'
  return '晚上好'
}

function Welcome({
  onOpenKB,
  onQuickAsk,
}: {
  onOpenKB: () => void
  onQuickAsk: (q: string) => void
}) {
  const typed = useTypewriter(TAGLINES)
  return (
    <div className="welcome">
      <div className="welcome-icon">
        <span className="welcome-ring" />
        <div className="welcome-logo">
          <Sparkles size={36} />
        </div>
      </div>

      <div className="welcome-greet">
        <span className="wave">👋</span>
        {getGreeting()}，我是「智问」
      </div>

      <h2>你的知识库 AI 大脑</h2>

      <p className="welcome-type">
        {typed}
        <span className="cursor" />
      </p>

      <div className="welcome-features">
        <span className="welcome-feature">混合检索</span>
        <span className="welcome-feature">Self-RAG Agent</span>
        <span className="welcome-feature">多知识库</span>
        <span className="welcome-feature">来源可溯</span>
      </div>

      <div className="quick-ask">
        <div className="quick-ask-label">试试这样提问：</div>
        <div className="quick-ask-list">
          {QUICK_PROMPTS.map(q => (
            <button key={q} className="quick-chip" onClick={() => onQuickAsk(q)}>
              {q}
              <Send size={13} className="quick-chip-arrow" />
            </button>
          ))}
        </div>
      </div>

      <div className="quick-tips">
        <span>快速开始：</span>
        <button onClick={onOpenKB}>
          <FileText size={14} /> 上传文档
        </button>
        <button onClick={onOpenKB}>
          <Globe size={14} /> 加载网页
        </button>
        <button onClick={onOpenKB}>
          <Library size={14} /> 管理知识库
        </button>
      </div>
    </div>
  )
}

/* ─── 消息气泡 ─── */
interface MessageBubbleProps {
  msg: ChatMessage
  index: number
  showSources: number | null
  setShowSources: (idx: number | null) => void
  streaming: boolean
}

const MessageBubble = memo(function MessageBubble({
  msg,
  index,
  showSources,
  setShowSources,
  streaming,
}: MessageBubbleProps) {
  const isThinking = msg.content.startsWith('__THINKING__')
  const isAssistant = msg.role === 'assistant'

  return (
    <div className={`message ${msg.role}`}>
      <div className="avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
      <div className="content-wrapper">
        <div className="content">
          {isAssistant ? (
            isThinking ? (
              <ThinkingStatus text={msg.content.slice(12)} />
            ) : msg.content ? (
              <MarkdownContent>{msg.content}</MarkdownContent>
            ) : streaming ? (
              <ThinkingStatus text="正在思考中" />
            ) : (
              <span className="typing">（空回复）</span>
            )
          ) : (
            <p className="user-text">{msg.content}</p>
          )}
        </div>

        {streaming && !isThinking && <span className="stream-caret" />}

        {isAssistant && !isThinking && msg.content && (
          <div className="msg-actions">
            <CopyButton text={msg.content} />
          </div>
        )}

        {isAssistant && msg.sources && msg.sources.length > 0 && (
          <SourcesSection
            sources={msg.sources}
            expanded={showSources === index}
            onToggle={() => setShowSources(showSources === index ? null : index)}
          />
        )}
      </div>
    </div>
  )
})

/* 思考中 / 等待状态 */
function ThinkingStatus({ text }: { text: string }) {
  const base = text.replace(/^[🔍📖⏳✨💭🧠⚙]+/, '').trim()
  return (
    <div className="thinking">
      <span className="thinking-dots">
        <i />
        <i />
        <i />
      </span>
      <span className="thinking-text">{base || '正在思考中'}</span>
    </div>
  )
}

/* 复制按钮（复用） */
function CopyButton({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  if (!text) return null
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* ignore */
    }
  }
  return (
    <button className="btn-copy" onClick={copy} title="复制文本">
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? '已复制' : label || '复制'}
    </button>
  )
}

/* ─── Markdown 渲染（GFM + 代码块复制） ─── */

function extractText(node: ReactNode): string {
  if (node == null) return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(extractText).join('')
  if (isValidElement(node)) {
    const props = node.props as { children?: ReactNode }
    return extractText(props.children)
  }
  return ''
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false)
  const firstChild = Children.toArray(children)[0]
  const codeProps = isValidElement(firstChild) ? (firstChild.props as { className?: string }) : null
  const lang = /language-([\w+-]+)/.exec(codeProps?.className || '')?.[1] || 'code'
  const text = extractText(firstChild)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-block-lang">{lang}</span>
        <button className="code-block-copy" onClick={copy}>
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? '已复制' : '复制'}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  )
}

const MarkdownContent = memo(function MarkdownContent({ children }: { children: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ children: preChildren }) => <CodeBlock>{preChildren}</CodeBlock>,
          a: ({ href, children: linkChildren }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {linkChildren}
            </a>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
})

/* ─── 来源引用 ─── */
interface SourcesSectionProps {
  sources: SourceItem[]
  expanded: boolean
  onToggle: () => void
}

function SourcesSection({ sources, expanded, onToggle }: SourcesSectionProps) {
  return (
    <div className="sources-section">
      <button className="btn-sources-toggle" onClick={onToggle}>
        <FileText size={13} />
        {sources.length} 个来源引用
        <span className={`chevron ${expanded ? 'up' : ''}`} />
      </button>
      {expanded && (
        <div className="sources-list">
          {sources.map((s, si) => (
            <div key={si} className="source-item">
              <div className="source-header">
                <span className="source-index">来源 {si + 1}</span>
                <span className="source-score">
                  相关度 {((s.relevance_score || 0) * 100).toFixed(0)}%
                </span>
              </div>
              <div className="source-path">{s.source || '未知来源'}</div>
              <div className="source-content">
                {s.content ? s.content.slice(0, 280) : ''}
                {s.content && s.content.length > 280 ? '…' : ''}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
