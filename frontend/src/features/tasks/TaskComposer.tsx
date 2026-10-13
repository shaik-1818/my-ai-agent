import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Globe, Play, Loader2, CheckCircle2, XCircle,
  ShieldAlert, ChevronDown, ChevronUp, Terminal,
  FileText, AlertCircle, Mic, MicOff, Square,
  ShieldCheck, ShieldX,
} from 'lucide-react'
import { createTask, getTask, getTaskEvents, approveAction } from './api'
import type { TaskEvent, TaskResult, TaskType } from './types'
import { READY_FOR_SUBMISSION_MESSAGE, isReadyForSubmission } from './types'
import { useSpeechRecognition, SPEECH_LANGS } from './useSpeechRecognition'
import './TaskComposer.css'

interface TaskComposerProps {
  onTaskCreated?: (taskId: string, taskText: string) => void
  onTaskFinished?: (taskId: string, status: TaskResult['status']) => void
}

const TERMINAL = new Set(['COMPLETED', 'FAILED', 'BLOCKED', 'CANCELLED'])

function StatusBadge({ status, awaitingReview = false }: { status: string; awaitingReview?: boolean }) {
  if (awaitingReview) {
    return <span className="badge badge-yellow">Awaiting review</span>
  }
  const map: Record<string, { label: string; cls: string }> = {
    COMPLETED: { label: 'Completed',  cls: 'badge-green'  },
    FAILED:    { label: 'Failed',     cls: 'badge-red'    },
    BLOCKED:   { label: 'Blocked',    cls: 'badge-yellow' },
    CANCELLED: { label: 'Cancelled',  cls: 'badge-dim'    },
    RUNNING:   { label: 'Running',    cls: 'badge-blue'   },
    QUEUED:    { label: 'Queued',     cls: 'badge-dim'    },
  }
  const { label, cls } = map[status] ?? { label: status, cls: 'badge-dim' }
  return <span className={`badge ${cls}`}>{label}</span>
}

function EventIcon({ type }: { type: string }) {
  if (type.includes('COMPLETED')) return <CheckCircle2 size={13} className="ev-icon green" />
  if (type.includes('FAILED'))    return <XCircle      size={13} className="ev-icon red"   />
  if (type.includes('BLOCKED'))   return <ShieldAlert  size={13} className="ev-icon yellow"/>
  if (type.includes('THINKING'))  return <Loader2      size={13} className="ev-icon blue spin" />
  if (type.includes('BROWSER'))   return <Globe        size={13} className="ev-icon accent" />
  if (type.includes('PAGE'))      return <FileText     size={13} className="ev-icon accent" />
  if (type.includes('EXECUTING')) return <Terminal     size={13} className="ev-icon dim"   />
  return <span className="ev-dot" />
}

export default function TaskComposer({ onTaskCreated, onTaskFinished }: TaskComposerProps) {
  const [task, setTask]               = useState('')
  const [interimText, setInterimText] = useState('')
  const [lang, setLang]               = useState('en-IN')
  const [url, setUrl]                 = useState('')
  const [taskType, setTaskType]       = useState<TaskType>('computer')
  const [taskId, setTaskId]           = useState<string | null>(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)
  const [taskResult, setTaskResult]   = useState<TaskResult | null>(null)
  const [taskEvents, setTaskEvents]   = useState<TaskEvent[]>([])
  const [showEvents, setShowEvents]   = useState(true)
  const [approving, setApproving]     = useState(false)
  const [approvalError, setApprovalError] = useState<string | null>(null)
  const eventsEndRef                  = useRef<HTMLDivElement>(null)
  const notifiedFinish                = useRef(false)

  // ── Voice callbacks ───────────────────────────────────────────────
  // Append each committed utterance to whatever the user has already typed.
  const handleFinalTranscript = useCallback((text: string) => {
    setTask(prev => {
      const base = prev.trimEnd()
      return base ? `${base} ${text}` : text
    })
    setInterimText('')
  }, [])

  const handleInterimTranscript = useCallback((text: string) => {
    setInterimText(text)
  }, [])

  const { micState, micError, isSupported, startListening, stopListening } =
    useSpeechRecognition({
      onFinalTranscript:   handleFinalTranscript,
      onInterimTranscript: handleInterimTranscript,
      lang,
    })

  // Clear interim text when mic goes idle/error.
  useEffect(() => {
    if (micState !== 'listening') setInterimText('')
  }, [micState])

  // ── Polling ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!taskId) return

    let mounted = true
    let intervalId: ReturnType<typeof setInterval> | null = null
    notifiedFinish.current = false

    const poll = async (): Promise<boolean> => {
      try {
        const [result, events] = await Promise.all([
          getTask(taskId),
          getTaskEvents(taskId),
        ])
        if (!mounted) return false
        setTaskResult(result)
        setTaskEvents(events)
        if (TERMINAL.has(result.status)) {
          if (!notifiedFinish.current) {
            notifiedFinish.current = true
            onTaskFinished?.(taskId, result.status)
          }
          return true
        }
      } catch (err) {
        if (!mounted) return false
        setError(err instanceof Error ? err.message : 'Failed to fetch task status.')
      }
      return false
    }

    ;(async () => {
      const done = await poll()
      if (!mounted || done) return
      intervalId = setInterval(async () => {
        const finished = await poll()
        if (finished && intervalId !== null) {
          clearInterval(intervalId)
          intervalId = null
        }
      }, 1000)
    })()

    return () => {
      mounted = false
      if (intervalId !== null) clearInterval(intervalId)
    }
  }, [taskId]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Auto-scroll events ────────────────────────────────────────────
  useEffect(() => {
    if (showEvents) eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [taskEvents, showEvents])

  // ── Approval ──────────────────────────────────────────────────────
  async function handleApproval(approved: boolean) {
    if (!taskId || !taskResult?.pending_action) return
    const { action_id } = taskResult.pending_action
    setApproving(true)
    setApprovalError(null)
    try {
      await approveAction(taskId, approved, action_id)
      // The worker will update the task status; polling will pick it up.
    } catch (err) {
      setApprovalError(err instanceof Error ? err.message : 'Approval request failed.')
    } finally {
      setApproving(false)
    }
  }

  // ── Submit ────────────────────────────────────────────────────────
  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const finalTask = (task + (interimText ? ' ' + interimText : '')).trim()
    if (!finalTask) { setError('Please enter a task.'); return }
    if (micState === 'listening') stopListening()
    setLoading(true)
    setError(null)
    setTaskId(null)
    setTaskResult(null)
    setTaskEvents([])
    setInterimText('')
    setApprovalError(null)
    try {
      const response = await createTask({
        task: finalTask,
        task_type: taskType,
        url: url.trim() || null,
        require_confirmation: true,
      })
      setTaskId(response.task_id)
      onTaskCreated?.(response.task_id, finalTask)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create task.')
    } finally {
      setLoading(false)
    }
  }

  const isTerminal           = taskResult ? TERMINAL.has(taskResult.status) : false
  const isSuccess             = taskResult?.success === true
  const isBlocked             = taskResult?.blocked === true
  const isListening           = micState === 'listening'
  const isWaitingConfirmation = taskResult?.status === 'WAITING_CONFIRMATION'
  const pendingAction         = taskResult?.pending_action ?? null
  const awaitingReview        = isReadyForSubmission(taskResult)

  return (
    <div className="composer">

      {/* ── Form card ── */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">New task</span>
          {taskResult && <StatusBadge status={taskResult.status} awaitingReview={awaitingReview} />}
        </div>

        <form onSubmit={handleSubmit} className="task-form">
          <fieldset className="field task-type-field">
            <legend className="field-label">Task type</legend>
            <div className="task-type-options" role="radiogroup" aria-label="Task type">
              <label className={`task-type-option${taskType === 'computer' ? ' is-selected' : ''}`}>
                <input
                  type="radio"
                  name="task_type"
                  value="computer"
                  checked={taskType === 'computer'}
                  onChange={() => setTaskType('computer')}
                  disabled={loading}
                />
                <span className="task-type-option-title">General computer task</span>
                <span className="task-type-option-hint">Browse and complete a page task</span>
              </label>
              <label className={`task-type-option${taskType === 'coding' ? ' is-selected' : ''}`}>
                <input
                  type="radio"
                  name="task_type"
                  value="coding"
                  checked={taskType === 'coding'}
                  onChange={() => setTaskType('coding')}
                  disabled={loading}
                />
                <span className="task-type-option-title">Coding task</span>
                <span className="task-type-option-hint">Solve, test, and prepare — never auto-submit</span>
              </label>
            </div>
          </fieldset>

          <div className="field">
            <div className="field-label-row">
              <label htmlFor="task" className="field-label">Task description</label>

              {/* Voice controls — only when supported */}
              {isSupported && (
                <div className="mic-controls">
                  {/* Language selector */}
                  <select
                    className="mic-lang-select"
                    value={lang}
                    onChange={e => setLang(e.target.value)}
                    disabled={isListening || loading}
                    aria-label="Speech recognition language"
                  >
                    {SPEECH_LANGS.map(l => (
                      <option key={l.value} value={l.value}>{l.label}</option>
                    ))}
                  </select>

                  {/* Mic toggle button */}
                  <button
                    type="button"
                    className={`mic-btn${isListening ? ' mic-btn--active' : ''}`}
                    onClick={isListening ? stopListening : startListening}
                    disabled={loading}
                    title={isListening ? 'Stop recording' : 'Start voice input'}
                    aria-label={isListening ? 'Stop recording' : 'Start voice input'}
                  >
                    {isListening
                      ? <><Square size={11} className="mic-stop-icon" /> Stop</>
                      : <><Mic size={12} /> Dictate</>
                    }
                  </button>
                </div>
              )}
            </div>

            {/* Listening indicator */}
            {isListening && (
              <div className="mic-listening-bar" aria-live="polite">
                <span className="mic-pulse" />
                Listening… speak now. Click Stop when done, then edit if needed.
              </div>
            )}

            {/* Textarea — shows committed text; interim shown below */}
            <textarea
              id="task"
              className={`field-textarea${isListening ? ' field-textarea--recording' : ''}`}
              value={task}
              onChange={e => setTask(e.target.value)}
              placeholder={
                taskType === 'coding'
                  ? 'Describe the coding problem or what to solve…'
                  : 'Tell the agent what you want it to do…'
              }
              rows={4}
              disabled={loading}
            />

            {/* Interim transcript preview */}
            {interimText && (
              <div className="mic-interim" aria-live="polite">
                <span className="mic-interim-label">Hearing:</span> {interimText}
              </div>
            )}
          </div>

          <div className="field">
            <label htmlFor="url" className="field-label">
              <Globe size={12} style={{ display: 'inline', marginRight: 5, verticalAlign: 'middle' }} />
              {taskType === 'coding' ? 'Coding website URL' : 'Website URL'}
              <span className="field-optional">optional</span>
            </label>
            <input
              id="url"
              type="url"
              className="field-input"
              value={url}
              onChange={e => setUrl(e.target.value)}
              placeholder={
                taskType === 'coding'
                  ? 'https://example.com/problem'
                  : 'https://example.com'
              }
              disabled={loading}
            />
            {taskType === 'coding' && (
              <p className="field-hint">
                With a problem-page URL, the agent runs tests on the site and stops for your review.
                It never clicks Submit.
              </p>
            )}
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading
              ? <><Loader2 size={14} className="spin" /> Starting…</>
              : <><Play size={14} /> Run task</>
            }
          </button>
        </form>
      </div>

      {/* ── Mic / voice alerts ── */}
      {!isSupported && (
        <div className="alert alert-warn" role="status">
          <MicOff size={14} />
          Voice input is not supported in this browser. Use Chrome or Edge for dictation.
        </div>
      )}
      {micState === 'error' && micError && (
        <div className="alert alert-warn" role="alert">
          <MicOff size={14} />
          {micError}
        </div>
      )}

      {/* ── General error banner ── */}
      {error && (
        <div className="alert alert-error" role="alert">
          <AlertCircle size={14} />
          {error}
        </div>
      )}

      {/* ── Result card ── */}
      {taskResult && isTerminal && (
        <div className={`card result-card ${isBlocked ? 'result-blocked' : awaitingReview ? 'result-review' : isSuccess ? 'result-success' : 'result-failed'}`}>
          <div className="card-header">
            {isBlocked
              ? <ShieldAlert size={15} className="result-icon yellow" />
              : awaitingReview
                ? <FileText size={15} className="result-icon yellow" />
              : isSuccess
                ? <CheckCircle2 size={15} className="result-icon green" />
                : <XCircle size={15} className="result-icon red" />
            }
            <span className="card-title">
              {isBlocked
                ? 'Task blocked'
                : awaitingReview
                  ? READY_FOR_SUBMISSION_MESSAGE
                  : isSuccess
                    ? 'Task completed'
                    : 'Task failed'}
            </span>
            <StatusBadge status={taskResult.status} awaitingReview={awaitingReview} />
          </div>

          {awaitingReview && (
            <p className="result-review-note">
              Tests ran where possible. This solution has not been submitted.
            </p>
          )}

          {taskResult.summary && (
            <div className="result-summary">
              {taskResult.summary.split('\n').map((line, i) => (
                <p key={i} className={line.startsWith('•') ? 'bullet-line' : ''}>{line}</p>
              ))}
            </div>
          )}

          {taskResult.error && (
            <div className="result-error-text">
              <span className="result-error-label">Error:</span> {taskResult.error}
            </div>
          )}

          {taskResult.result && Object.keys(taskResult.result).length > 0 && (
            <details className="result-details">
              <summary>Raw result data</summary>
              <pre className="result-pre">{JSON.stringify(taskResult.result, null, 2)}</pre>
            </details>
          )}

          <div className="result-meta">
            <span>Task ID: <code>{taskResult.task_id}</code></span>
            <span>Stage: <code>{taskResult.current_stage}</code></span>
            {taskResult.verified && <span className="verified-badge">✓ Verified</span>}
          </div>
        </div>
      )}

      {/* ── Running status ── */}
      {taskResult && !isTerminal && !isWaitingConfirmation && (
        <div className="card running-card">
          <div className="running-row">
            <Loader2 size={14} className="spin accent" />
            <span className="running-label">
              {taskResult.current_stage || taskResult.status}
            </span>
            <StatusBadge status={taskResult.status} />
          </div>
        </div>
      )}

      {/* ── Approval card ── */}
      {isWaitingConfirmation && pendingAction && (
        <div className="card approval-card" role="region" aria-label="Approval required">
          <div className="card-header approval-header">
            <ShieldAlert size={15} className="result-icon yellow" />
            <span className="card-title">Approval required</span>
            <span className="badge badge-yellow">Waiting</span>
          </div>

          <div className="approval-body">
            <p className="approval-desc">
              The agent wants to perform a sensitive browser action.
              Review the details below and approve or reject.
            </p>

            <div className="approval-detail-grid">
              <span className="approval-detail-label">Action</span>
              <span className="approval-detail-value approval-action-name">
                {pendingAction.action.action === 'browser_click' ? 'Click' :
                 pendingAction.action.action === 'browser_type'  ? 'Type into field' :
                 pendingAction.action.action}
              </span>

              {pendingAction.action.target_text && (
                <>
                  <span className="approval-detail-label">Button / link</span>
                  <span className="approval-detail-value">{pendingAction.action.target_text}</span>
                </>
              )}

              {pendingAction.action.target_type && (
                <>
                  <span className="approval-detail-label">Field type</span>
                  <span className="approval-detail-value">{pendingAction.action.target_type}</span>
                </>
              )}

              {pendingAction.action.target_placeholder && (
                <>
                  <span className="approval-detail-label">Placeholder</span>
                  <span className="approval-detail-value">{pendingAction.action.target_placeholder}</span>
                </>
              )}

              <span className="approval-detail-label">Selector</span>
              <code className="approval-detail-value approval-selector">
                {pendingAction.action.selector}
              </code>
            </div>

            {approvalError && (
              <div className="alert alert-error approval-alert" role="alert">
                <AlertCircle size={13} />
                {approvalError}
              </div>
            )}

            <div className="approval-actions">
              <button
                type="button"
                className="approval-btn approval-btn--approve"
                onClick={() => handleApproval(true)}
                disabled={approving}
                aria-label="Approve action"
              >
                {approving
                  ? <Loader2 size={13} className="spin" />
                  : <ShieldCheck size={13} />}
                Approve
              </button>
              <button
                type="button"
                className="approval-btn approval-btn--reject"
                onClick={() => handleApproval(false)}
                disabled={approving}
                aria-label="Reject action"
              >
                <ShieldX size={13} />
                Reject
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Events ── */}
      {taskEvents.length > 0 && (
        <div className="card events-card">
          <button
            className="events-toggle"
            onClick={() => setShowEvents(v => !v)}
            type="button"
          >
            <Terminal size={13} />
            <span>Execution log</span>
            <span className="events-count">{taskEvents.length}</span>
            <span style={{ marginLeft: 'auto' }}>
              {showEvents ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </span>
          </button>

          {showEvents && (
            <div className="events-list">
              {taskEvents.map((evt, i) => (
                <div key={i} className="event-row">
                  <EventIcon type={evt.event_type} />
                  <span className="event-time">
                    {evt.timestamp
                      ? new Date(evt.timestamp).toLocaleTimeString([], {
                          hour: '2-digit', minute: '2-digit', second: '2-digit',
                        })
                      : ''}
                  </span>
                  <span className="event-type">{evt.event_type}</span>
                  <span className="event-msg">{evt.message}</span>
                </div>
              ))}
              <div ref={eventsEndRef} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
