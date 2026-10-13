import { useState, useRef, useCallback, useEffect } from 'react'

// ── Public types ────────────────────────────────────────────────────────────

export type MicState = 'idle' | 'listening' | 'unsupported' | 'error'

export interface SpeechLang {
  label: string
  value: string
}

export const SPEECH_LANGS: SpeechLang[] = [
  { label: 'English (India)', value: 'en-IN' },
  { label: 'English (US)',    value: 'en-US' },
  { label: 'Hindi',           value: 'hi-IN' },
  { label: 'Telugu',          value: 'te-IN' },
]

export interface UseSpeechRecognitionOptions {
  /** Called with each committed (final) utterance to append. */
  onFinalTranscript: (text: string) => void
  /** Called with the current in-progress text. Empty string clears it. */
  onInterimTranscript: (text: string) => void
  lang?: string
}

export interface UseSpeechRecognitionResult {
  micState: MicState
  micError: string | null
  isSupported: boolean
  startListening: () => void
  stopListening: () => void
}

// ── Internal browser-API shape ──────────────────────────────────────────────

interface SRLike {
  continuous: boolean
  interimResults: boolean
  maxAlternatives: number
  lang: string
  start(): void
  stop(): void
  abort(): void
  onstart:       (() => void) | null
  onaudiostart:  (() => void) | null
  onspeechstart: (() => void) | null
  onresult:      ((e: SRResultEvent) => void) | null
  onerror:       ((e: SRErrorEvent) => void) | null
  onend:         (() => void) | null
}

interface SRResultEvent {
  resultIndex: number
  results: SpeechRecognitionResultList
}

interface SRErrorEvent {
  error: string
}

type SRCtor = new () => SRLike

// ── Constants ───────────────────────────────────────────────────────────────

// Edge needs at least 500 ms after abort() before a new start() is safe.
// Use 800 ms to give the browser's async cleanup enough time.
const RESTART_DELAY_MS = 800

// Errors that must never trigger an auto-restart.
const FATAL_ERRORS = new Set([
  'not-allowed',
  'service-not-allowed',
  'audio-capture',
])

// ── Helpers ─────────────────────────────────────────────────────────────────

function getSRCtor(): SRCtor | null {
  const w = window as unknown as Record<string, unknown>
  const ctor =
    (w['SpeechRecognition'] as SRCtor | undefined) ??
    (w['webkitSpeechRecognition'] as SRCtor | undefined) ??
    null
  console.debug(
    '[SR] API check:',
    ctor ? 'SpeechRecognition available' : 'NOT available',
    '| SpeechRecognition' in window ? '(standard)' : '',
    'webkitSpeechRecognition' in window ? '(webkit)' : '',
  )
  return ctor
}

function errorMessage(code: string): string {
  switch (code) {
    case 'not-allowed':
    case 'service-not-allowed':
      return (
        'Microphone access was denied. ' +
        'Click the lock icon in the address bar, allow microphone access, then try again.'
      )
    case 'audio-capture':
      return 'No microphone was found. Connect a microphone and try again.'
    case 'network':
      return 'Speech recognition needs an internet connection. Check your network and try again.'
    case 'no-speech':
      return 'No speech detected. Make sure your microphone is working and try again.'
    case 'aborted':
      return '' // intentional — show nothing
    default:
      return `Speech recognition error: ${code}`
  }
}

// ── Hook ────────────────────────────────────────────────────────────────────

export function useSpeechRecognition({
  onFinalTranscript,
  onInterimTranscript,
  lang = 'en-IN',
}: UseSpeechRecognitionOptions): UseSpeechRecognitionResult {

  // Resolve the constructor once per hook instance (stable across renders).
  const CtorRef = useRef<SRCtor | null>(null)
  if (CtorRef.current === null) {
    CtorRef.current = getSRCtor()
  }
  const Ctor = CtorRef.current
  const isSupported = Ctor !== null

  const [micState, setMicState] = useState<MicState>(
    isSupported ? 'idle' : 'unsupported',
  )
  const [micError, setMicError] = useState<string | null>(null)

  // ── Stable refs ──────────────────────────────────────────────────
  const recognitionRef      = useRef<SRLike | null>(null)
  const restartTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null)
  const userStoppedRef      = useRef(false)
  const fatalErrorRef       = useRef(false)
  const audioStartedRef     = useRef(false)
  // Counts how many times we have restarted after an 'aborted' error on a
  // fresh (non-user-stopped) start. Prevents infinite abort loops.
  const abortRetryCountRef  = useRef(0)
  const mountedRef          = useRef(true)
  const langRef             = useRef(lang)
  const sessionIdRef        = useRef(0) // incremented on each startListening call

  // Keep langRef in sync without recreating callbacks.
  useEffect(() => { langRef.current = lang }, [lang])

  // Stable callback refs — recognition handlers always call the latest version.
  const onFinalRef   = useRef(onFinalTranscript)
  const onInterimRef = useRef(onInterimTranscript)
  useEffect(() => { onFinalRef.current   = onFinalTranscript }, [onFinalTranscript])
  useEffect(() => { onInterimRef.current = onInterimTranscript }, [onInterimTranscript])

  // Unmount cleanup.
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      userStoppedRef.current = true
      if (restartTimerRef.current !== null) {
        clearTimeout(restartTimerRef.current)
        restartTimerRef.current = null
      }
      if (recognitionRef.current) {
        detachHandlers(recognitionRef.current)
        try { recognitionRef.current.abort() } catch { /* ignore */ }
        recognitionRef.current = null
      }
    }
  }, [])

  // ── Utility: detach all handlers from an instance ────────────────
  function detachHandlers(sr: SRLike) {
    sr.onstart       = null
    sr.onaudiostart  = null
    sr.onspeechstart = null
    sr.onresult      = null
    sr.onerror       = null
    sr.onend         = null
  }

  // ── Core: create and wire one recognition instance ───────────────
  // `sessionId` lets handlers ignore events from a superseded session.
  const startInstance = useCallback((sessionId: number) => {
    if (!Ctor || !mountedRef.current) return

    console.debug('[SR] startInstance called, session', sessionId)

    // Tear down any lingering instance.
    if (recognitionRef.current) {
      detachHandlers(recognitionRef.current)
      try { recognitionRef.current.abort() } catch { /* ignore */ }
      recognitionRef.current = null
    }

    audioStartedRef.current = false

    const sr = new Ctor()
    sr.continuous      = true
    sr.interimResults  = true
    sr.maxAlternatives = 3
    sr.lang            = langRef.current
    recognitionRef.current = sr

    console.debug('[SR] Recognition configured:', {
      continuous: sr.continuous,
      interimResults: sr.interimResults,
      maxAlternatives: sr.maxAlternatives,
      lang: sr.lang,
    })

    sr.onstart = () => {
      if (!mountedRef.current || sessionIdRef.current !== sessionId) return
      console.debug('[SR] onstart — recognition active, session', sessionId)
      abortRetryCountRef.current = 0 // successful start resets the abort-retry counter
      setMicState('listening')
      setMicError(null)
    }

    sr.onaudiostart = () => {
      if (sessionIdRef.current !== sessionId) return
      console.debug('[SR] onaudiostart — microphone audio stream opened')
      audioStartedRef.current = true
    }

    sr.onspeechstart = () => {
      if (sessionIdRef.current !== sessionId) return
      console.debug('[SR] onspeechstart — speech detected')
      audioStartedRef.current = true
    }

    sr.onresult = (e: SRResultEvent) => {
      if (!mountedRef.current || sessionIdRef.current !== sessionId) return

      let interimText = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const result = e.results[i]
        const text   = result[0].transcript
        console.debug(
          '[SR] onresult index', i,
          'isFinal:', result.isFinal,
          'text:', JSON.stringify(text),
        )
        if (result.isFinal) {
          onFinalRef.current(text.trim())
          onInterimRef.current('')
        } else {
          interimText += text
        }
      }
      if (interimText) onInterimRef.current(interimText)
    }

    sr.onerror = (e: SRErrorEvent) => {
      if (!mountedRef.current || sessionIdRef.current !== sessionId) return
      console.error('[SR] onerror — code:', e.error, '| session', sessionId)

      if (e.error === 'no-speech') {
        // Non-fatal. onend will fire next; we'll restart from there.
        // Show a brief hint in the UI so the user knows what happened.
        setMicError('No speech detected — still listening, speak clearly into your microphone.')
        return
      }

      if (e.error === 'aborted') {
        // Could be our own abort() from a previous teardown that arrived late,
        // or the browser aborting a fresh start due to a race condition.
        // onend will fire; the restart logic there handles it.
        console.debug('[SR] aborted error received (may be stale from previous session)')
        return
      }

      if (FATAL_ERRORS.has(e.error)) {
        fatalErrorRef.current = true
        const msg = errorMessage(e.error)
        console.error('[SR] FATAL error:', e.error, '—', msg)
        setMicError(msg)
        setMicState('error')
        return
      }

      // Non-fatal (network, etc.) — show message, onend will restart.
      const msg = errorMessage(e.error)
      console.error('[SR] non-fatal error:', e.error, '—', msg)
      if (msg) setMicError(msg)
    }

    sr.onend = () => {
      if (!mountedRef.current || sessionIdRef.current !== sessionId) return
      console.debug('[SR] onend — session', sessionId,
        '| userStopped:', userStoppedRef.current,
        '| fatal:', fatalErrorRef.current,
        '| audioStarted:', audioStartedRef.current,
      )

      recognitionRef.current = null
      onInterimRef.current('')

      // User clicked Stop or a fatal error occurred — stay idle.
      if (userStoppedRef.current || fatalErrorRef.current) {
        setMicState(prev => (prev === 'listening' ? 'idle' : prev))
        return
      }

      // Check if this end was caused by an 'aborted' error on a fresh start
      // (Edge race condition). Limit retries to avoid an infinite loop.
      if (!audioStartedRef.current) {
        abortRetryCountRef.current += 1
        console.debug('[SR] onend before audio started — abort retry count:', abortRetryCountRef.current)

        if (abortRetryCountRef.current > 3) {
          console.error('[SR] Too many abort retries — stopping. Check browser microphone permissions.')
          setMicError(
            'Speech recognition could not start after several attempts. ' +
            'Check that your microphone is connected and that this site has permission to use it.',
          )
          setMicState('error')
          return
        }
      }

      // Schedule a restart. Use a longer delay so Edge's async abort()
      // processing completes before we call start() again.
      console.debug('[SR] scheduling restart in', RESTART_DELAY_MS, 'ms')
      setMicState('listening') // keep the indicator visible during the gap
      restartTimerRef.current = setTimeout(() => {
        restartTimerRef.current = null
        if (!mountedRef.current || userStoppedRef.current) return
        console.debug('[SR] restarting recognition after delay')
        startInstance(sessionId)
      }, RESTART_DELAY_MS)
    }

    console.debug('[SR] calling recognition.start()')
    try {
      sr.start()
    } catch (err) {
      console.error('[SR] start() threw synchronously:', err)
      recognitionRef.current = null
      setMicError('Could not start speech recognition. Try again.')
      setMicState('error')
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Ctor])

  // ── Public: startListening ────────────────────────────────────────
  const startListening = useCallback(() => {
    if (!Ctor) {
      console.error('[SR] startListening called but SpeechRecognition is not supported')
      setMicState('unsupported')
      return
    }

    console.debug('[SR] startListening called')

    // Cancel any pending restart timer from a previous session.
    if (restartTimerRef.current !== null) {
      clearTimeout(restartTimerRef.current)
      restartTimerRef.current = null
    }

    userStoppedRef.current     = false
    fatalErrorRef.current      = false
    audioStartedRef.current    = false
    abortRetryCountRef.current = 0

    // Increment session ID so stale handlers from the previous session
    // are ignored even if their events arrive after the new session starts.
    sessionIdRef.current += 1
    const sessionId = sessionIdRef.current

    setMicError(null)
    startInstance(sessionId)
  }, [Ctor, startInstance])

  // ── Public: stopListening ─────────────────────────────────────────
  const stopListening = useCallback(() => {
    console.debug('[SR] stopListening called')
    userStoppedRef.current = true

    if (restartTimerRef.current !== null) {
      clearTimeout(restartTimerRef.current)
      restartTimerRef.current = null
    }

    if (recognitionRef.current) {
      detachHandlers(recognitionRef.current)
      try { recognitionRef.current.stop() } catch { /* ignore */ }
      recognitionRef.current = null
    }

    onInterimRef.current('')
    setMicState('idle')
  }, [])

  return { micState, micError, isSupported, startListening, stopListening }
}
