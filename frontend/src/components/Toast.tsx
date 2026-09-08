import React, { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import { AlertTriangle, Check, Info, X } from 'lucide-react'

interface ToastItem {
  id: number
  message: string
  type: 'success' | 'error' | 'warning' | 'info'
}

interface ToastAPI {
  success: (msg: string, dur?: number) => number
  error: (msg: string, dur?: number) => number
  info: (msg: string, dur?: number) => number
  warning: (msg: string, dur?: number) => number
}

const ToastContext = createContext<ToastAPI | null>(null)

export function useToast(): ToastAPI {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}

let _toastId = 0

const TYPE_ICON: Record<ToastItem['type'], ReactNode> = {
  success: <Check size={14} />,
  error: <X size={14} />,
  warning: <AlertTriangle size={14} />,
  info: <Info size={14} />,
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const addToast = useCallback(
    (message: string, type: ToastItem['type'], duration = 3000): number => {
      const id = ++_toastId
      setToasts(prev => [...prev.slice(-3), { id, message, type }])
      if (duration > 0) {
        setTimeout(() => {
          setToasts(prev => prev.filter(t => t.id !== id))
        }, duration)
      }
      return id
    },
    [],
  )

  const removeToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const toast: ToastAPI = {
    success: (msg, dur) => addToast(msg, 'success', dur),
    error: (msg, dur) => addToast(msg, 'error', dur || 5000),
    info: (msg, dur) => addToast(msg, 'info', dur),
    warning: (msg, dur) => addToast(msg, 'warning', dur || 4000),
  }

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <div className="toast-container" aria-live="polite">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast-${t.type}`} role="status">
            <span className="toast-icon">{TYPE_ICON[t.type]}</span>
            <span className="toast-message">{t.message}</span>
            <button className="toast-close" onClick={() => removeToast(t.id)} aria-label="关闭提示">
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
