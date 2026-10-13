import { useState } from 'react'
import { Bot, Plus, Clock } from 'lucide-react'
import TaskComposer from './features/tasks/TaskComposer'
import type { TaskResult } from './features/tasks/types'
import './App.css'

interface HistoryEntry {
  taskId: string
  task: string
  status: TaskResult['status'] | 'RUNNING'
  ts: string
}

const STATUS_COLOR: Record<string, string> = {
  COMPLETED: 'var(--green)',
  FAILED:    'var(--red)',
  BLOCKED:   'var(--yellow)',
  CANCELLED: 'var(--text-dim)',
  RUNNING:   'var(--accent)',
}

function statusColor(s: string) {
  return STATUS_COLOR[s] ?? 'var(--blue)'
}

export default function App() {
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [composerKey, setComposerKey] = useState(0)

  function handleTaskCreated(taskId: string, taskText: string) {
    const entry: HistoryEntry = {
      taskId,
      task: taskText,
      status: 'RUNNING',
      ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }
    setHistory(prev => [entry, ...prev])
    setActiveId(taskId)
  }

  function handleTaskFinished(taskId: string, status: TaskResult['status']) {
    setHistory(prev =>
      prev.map(e => e.taskId === taskId ? { ...e, status } : e)
    )
  }

  function startNewTask() {
    setActiveId(null)
    setComposerKey(k => k + 1)
  }

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <Bot size={15} />
          </div>
          <div>
            <div className="sidebar-title">AI Agent</div>
            <div className="sidebar-subtitle">Computer tasks</div>
          </div>
        </div>

        <button className="sidebar-new-btn" onClick={startNewTask}>
          <Plus size={13} />
          New task
        </button>

        {history.length > 0 && (
          <div className="sidebar-section-label">Recent</div>
        )}

        <div className="sidebar-list">
          {history.length === 0 ? (
            <div className="sidebar-empty">
              <Clock size={20} style={{ margin: '0 auto 8px', display: 'block', opacity: 0.4 }} />
              No tasks yet.<br />Start one to see history here.
            </div>
          ) : (
            history.map(entry => (
              <div
                key={entry.taskId}
                className={`sidebar-item${activeId === entry.taskId ? ' active' : ''}`}
                onClick={() => setActiveId(entry.taskId)}
              >
                <div className="sidebar-item-task">{entry.task}</div>
                <div className="sidebar-item-meta">
                  <span
                    className="sidebar-item-dot"
                    style={{ background: statusColor(entry.status) }}
                  />
                  <span>{entry.status}</span>
                  <span style={{ marginLeft: 'auto' }}>{entry.ts}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">
        <div className="main-header">
          <span className="main-header-title">Task workspace</span>
          <span className="main-header-status">
            <span className="status-dot" />
            Backend connected
          </span>
        </div>

        <div className="main-body">
          <TaskComposer
            key={composerKey}
            onTaskCreated={handleTaskCreated}
            onTaskFinished={handleTaskFinished}
          />
        </div>
      </main>
    </div>
  )
}
