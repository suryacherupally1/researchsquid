<template>
  <div class="report-panel">
    <h3>Report</h3>
    <div class="actions">
      <button @click="plan" :disabled="loading">Plan Outline</button>
      <button @click="generate" :disabled="loading" class="primary">
        {{ loading ? 'Generating...' : 'Generate Report' }}
      </button>
    </div>

    <div v-if="outline.length" class="outline">
      <h4>Outline</h4>
      <div v-for="(s, i) in outline" :key="i" class="outline-item">
        <span class="num">{{ i + 1 }}</span>
        <span class="title">{{ s.title }}</span>
      </div>
    </div>

    <div v-if="report" class="report-content">
      <h4>Generated Report</h4>
      <pre class="report-text">{{ report }}</pre>
    </div>

    <div v-if="auditLog.length" class="audit">
      <h4>Report Agent Audit ({{ auditLog.length }} steps)</h4>
      <div v-for="(log, i) in auditLog.slice(0, 20)" :key="i" class="audit-entry">
        <span class="time">{{ log.timestamp?.slice(11, 19) }}</span>
        <span class="action">{{ log.action }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { api } from '../api/index.js'

const props = defineProps({ sessionId: { type: String, required: true } })

const outline = ref([])
const report = ref('')
const auditLog = ref([])
const loading = ref(false)

async function plan() {
  loading.value = true
  try {
    const result = await api.planReport(props.sessionId)
    outline.value = result.outline || []
    auditLog.value = result.audit_log || []
  } catch (e) { alert(e.message) }
  finally { loading.value = false }
}

async function generate() {
  loading.value = true
  try {
    const result = await api.getReport(props.sessionId)
    report.value = result.report || ''
    outline.value = result.sections?.map(s => ({ title: s.title })) || []
    auditLog.value = result.audit_log || []
  } catch (e) { alert(e.message) }
  finally { loading.value = false }
}
</script>

<style scoped>
.report-panel { margin: 24px 0; }
h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
h4 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin: 16px 0 8px; }
.actions { display: flex; gap: 8px; margin-bottom: 16px; }
button { padding: 8px 16px; border: 1px solid #111; background: #fff; font-family: inherit; font-size: 12px; cursor: pointer; text-transform: uppercase; letter-spacing: 1px; }
button:hover { background: #f5f5f5; }
button.primary { background: #111; color: #fff; }
button.primary:hover { background: #333; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.outline-item { display: flex; gap: 8px; padding: 4px 0; font-size: 12px; }
.num { color: #999; width: 20px; }
.report-text { background: #fafafa; border: 1px solid #eee; padding: 16px; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; max-height: 600px; overflow-y: auto; }
.audit-entry { display: flex; gap: 8px; font-size: 11px; padding: 2px 0; border-bottom: 1px solid #f5f5f5; }
.time { color: #999; font-family: monospace; width: 55px; }
.action { color: #555; }
</style>
