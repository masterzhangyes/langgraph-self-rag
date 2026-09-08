import { Moon, Settings2, Sun, X } from 'lucide-react'

interface SettingsPanelProps {
  temperature: number
  setTemperature: (v: number) => void
  darkMode: boolean
  setDarkMode: (v: boolean) => void
  visible: boolean
  onClose: () => void
}

export default function SettingsPanel({
  temperature,
  setTemperature,
  darkMode,
  setDarkMode,
  visible,
  onClose,
}: SettingsPanelProps) {
  if (!visible) return null

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h2>
          <Settings2 size={16} />
          设置
        </h2>
        <button className="btn-icon" onClick={onClose} aria-label="关闭">
          <X size={16} />
        </button>
      </div>

      <div className="settings-content">
        <div className="setting-item">
          <label>LLM 温度</label>
          <div className="setting-control">
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={temperature}
              onChange={e => setTemperature(parseFloat(e.target.value))}
            />
            <span className="setting-value">{temperature.toFixed(1)}</span>
          </div>
          <div className="setting-presets">
            {[
              { v: 0.1, label: '严谨' },
              { v: 0.3, label: '平衡' },
              { v: 0.7, label: '创意' },
            ].map(p => (
              <button
                key={p.v}
                className={`preset ${Math.abs(temperature - p.v) < 0.05 ? 'on' : ''}`}
                onClick={() => setTemperature(p.v)}
              >
                {p.label}
              </button>
            ))}
          </div>
          <p className="hint" style={{ margin: 0 }}>
            较低值更严谨，较高值更有创意
          </p>
        </div>

        <div className="setting-item">
          <label>外观</label>
          <button
            className={`toggle-btn ${darkMode ? 'on' : ''}`}
            onClick={() => setDarkMode(!darkMode)}
            role="switch"
            aria-checked={darkMode}
          >
            {darkMode ? <Moon size={14} /> : <Sun size={14} />}
            {darkMode ? '暗色模式' : '亮色模式'}
          </button>
          <p className="hint" style={{ margin: 0 }}>
            偏好会自动保存到本地
          </p>
        </div>
      </div>
    </div>
  )
}
