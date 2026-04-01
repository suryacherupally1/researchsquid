/**
 * Session store — reactive state for current session.
 * Subscribes to SSE for real-time updates.
 */

import { reactive, ref } from 'vue'
import { api } from '../api/index.js'

export function useSessionStore() {
  const session = ref(null)
  const status = reactive({
    workflow_step: 0,
    agents: { total: 0, active: 0, sleeping: 0 },
    findings: { total: 0, active: 0, with_verification: 0 },
    experiments: { total: 0, pending: 0, running: 0, completed: 0, failed: 0 },
    contradictions: 0,
    budget: { llm_spent: 0, llm_budget: 0, compute_spent: 0, compute_budget: 0 },
    best_answer: { label: 'Initializing', confidence: 0 },
  })
  const events = ref([])
  const agents = ref([])
  const findings = ref([])
  const loading = ref(false)
  const error = ref(null)

  let eventSource = null

  const WORKFLOW_LABELS = {
    0: 'Initializing',
    1: 'Question / Goal',
    2: 'Setup',
    3: 'Agent Search',
    4: 'Experiments',
    5: 'Clusters / Debates',
    6: 'Report',
    7: 'Interview / Follow-up',
  }

  async function load(sessionId) {
    loading.value = true
    error.value = null
    try {
      session.value = await api.getSession(sessionId)
      const s = await api.getSessionStatus(sessionId)
      Object.assign(status, s)
      connectSSE(sessionId)
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  function connectSSE(sessionId) {
    if (eventSource) eventSource.close()
    eventSource = new EventSource(api.streamUrl(sessionId))
    eventSource.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data)
        if (event.event_type === 'keepalive') return
        events.value.unshift(event)
        if (events.value.length > 200) events.value.pop()
        handleEvent(event)
      } catch {}
    }
    eventSource.onerror = () => {
      setTimeout(() => connectSSE(sessionId), 5000)
    }
  }

  function handleEvent(event) {
    const t = event.event_type
    if (t === 'finding_posted') {
      status.findings.total++
      status.findings.active++
    } else if (t === 'contradiction_opened') {
      status.contradictions++
    } else if (t === 'experiment_submitted') {
      status.experiments.total++
      status.experiments.pending++
    } else if (t === 'experiment_completed') {
      status.experiments.completed++
      status.experiments.running = Math.max(0, status.experiments.running - 1)
    } else if (t === 'experiment_failed') {
      status.experiments.failed++
      status.experiments.running = Math.max(0, status.experiments.running - 1)
    } else if (t === 'budget_updated') {
      Object.assign(status.budget, event.data)
    } else if (t === 'session_completed') {
      status.workflow_step = 6
    }
  }

  function disconnect() {
    if (eventSource) eventSource.close()
  }

  return {
    session, status, events, agents, findings,
    loading, error, WORKFLOW_LABELS,
    load, disconnect,
  }
}
