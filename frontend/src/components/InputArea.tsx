import { useRef, useEffect, useCallback, type KeyboardEvent } from 'react'
import { SendHorizontal, Square } from 'lucide-react'

interface InputAreaProps {
  input: string
  setInput: (value: string) => void
  loading: boolean
  currentKb: { status: string }
  activeKb: string
  onSend: () => void
  onStop: () => void
}

export default function InputArea({
  input,
  setInput,
  loading,
  currentKb,
  activeKb,
  onSend,
  onStop,
}: InputAreaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const canSend = !loading && input.trim().length > 0

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 150) + 'px'
  }, [input])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      // 中文输入法组合期间不触发
      if (e.nativeEvent.isComposing) return
      if (e.key === 'Enter' && !e.shiftKey && !loading) {
        e.preventDefault()
        onSend()
      } else if (e.key === 'Enter' && loading) {
        e.preventDefault()
        onStop()
      }
    },
    [loading, onSend, onStop],
  )

  const kbActive = currentKb.status === 'active'

  return (
    <div className="input-area">
      <div className={`input-wrapper ${loading ? 'generating' : ''}`}>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            loading
              ? '正在生成回答，可继续输入，Enter 发送下一条…'
              : kbActive
                ? `基于知识库 [${activeKb}] 提问，Enter 发送`
                : '输入问题开始对话，Enter 发送'
          }
          rows={1}
        />
        <div className="input-actions">
          {loading ? (
            <button className="btn-stop" onClick={onStop} title="停止生成">
              <Square size={13} fill="currentColor" />
              停止
            </button>
          ) : (
            <button className="btn-send" onClick={onSend} disabled={!canSend} title="发送">
              <SendHorizontal size={16} />
            </button>
          )}
        </div>
      </div>

      <div className="input-footer">
        <span className={`kb-mini ${currentKb.status}`}>
          {kbActive ? `知识库：${activeKb}` : '通用模式（未绑定知识库）'}
        </span>
        <span className="input-hint">Enter 发送 · Shift + Enter 换行</span>
      </div>
    </div>
  )
}
