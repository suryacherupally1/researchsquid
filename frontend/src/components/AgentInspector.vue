<template>
  <div class="agent-inspector">
    <h3>Agent Inspector</h3>

    <div class="search">
      <input v-model="agentId" placeholder="Agent ID" @keyup.enter="inspect" />
      <button @click="inspect" :disabled="!agentId || loading">Inspect</button>
    </div>

    <div v-if="data" class="inspector-content">
      <!-- Hypothesis -->
      <div class="section" v-if="data.current_hypothesis">
        <h4>Current Hypothesis</h4>
        <div class="hypothesis">
          <span class="claim">{{ data.current_hypothesis.claim }}</span>
          <span class="conf">{{ (data.current_hypothesis.confidence * 100).toFixed(0) }}%</span>
        </div>
      </div>

      <!-- Persona -->
      <div class="section">
        <h4>Persona</h4>
        <div v-if="data.persona" class="persona">
          <div class="field"><span>Specialty:</span> {{ data.persona.specialty }}</div>
          <div class="field"><span>Skepticism:</span> {{ (data.persona.skepticism_level * 100).toFixed(0) }}%</div>
          <div class="field"><span>Source strictness:</span> {{ (data.persona.source_strictness * 100).toFixed(0) }}%</div>
          <div class="field"><span>Experiment appetite:</span> {{ (data.persona.experiment_appetite * 100).toFixed(0) }}%</div>
          <div class="field"><span>Revision:</span> {{ data.persona.revision }}</div>
        </div>
        <div v-else class="empty">No persona set</div>

        <!-- Persona Editor -->
        <div class="persona-editor">
          <h5>Edit Persona</h5>
          <select v-model="editTemplate">
            <option value="">— custom —</option>
            <option v-for="(_, name) in templates" :key="name" :value="name">{{ name }}</option>
          </select>
          <div class="slider-row">
            <label>Skepticism</label>
            <input type="range" min="0" max="100" v-model.number="editSkepticism" />
            <span>{{ editSkepticism }}%</span>
          </div>
          <div class="slider-row">
            <label>Source strictness</label>
            <input type="range" min="0" max="100" v-model.number="editSourceStrict" />
            <span>{{ editSourceStrict }}%</span>
          </div>
          <div class="slider-row">
            <label>Experiment appetite</label>
            <input type="range" min="0" max="100" v-model.number="editExperiment" />
            <span>{{ editExperiment }}%</span>
          </div>
          <select v-model="editReporting">
            <option value="concise">Concise</option>
            <option value="detailed">Detailed</option>
            <option value="critical">Critical</option>
          </select>
          <div class="model-tier-select">
            <label>Model Tier</label>
            <select v-model="editModelTier">
              <option value="fast">⚡ Fast (cheapest — Haiku, GPT-4o-mini)</option>
              <option value="balanced">⚖️ Balanced (mid — Sonnet, GPT-4o)</option>
              <option value="powerful">🧠 Powerful (best — Opus, o1)</option>
            </select>
          </div>
          <button @click="savePersona" class="save-btn">Save Persona</button>
        </div>
      </div>

      <!-- Recent Findings -->
      <div class="section">
        <h4>Recent Findings ({{ data.findings_count }})</h4>
        <div v-for="f in data.findings_recent" :key="f.id" class="finding-card">
          <span class="conf-badge" :class="confClass(f.confidence)">{{ (f.confidence * 100).toFixed(0) }}%</span>
          <span class="claim-text">{{ f.claim }}</span>
        </div>
      </div>

      <!-- Recent Experiments -->
      <div class="section">
        <h4>Experiments ({{ data.experiments_count }})</h4>
        <div v-for="e in data.experiments_recent" :key="e.id" class="exp-card">
          <span class="status-dot" :class="e.status"></span>
          <span class="goal">{{ e.goal || e.backend_type }}</span>
        </div>
      </div>

      <!-- Interview -->
      <div class="section">
        <h4>Interview Agent</h4>
        <textarea v-model="interviewPrompt" rows="2"
          placeholder="Ask this agent a question..."></textarea>
        <button @click="interview" :disabled="!interviewPrompt || interviewing" class="interview-btn">
          {{ interviewing ? 'Interviewing...' : 'Ask' }}
        </button>
        <div v-if="interviewResponse" class="interview-response">
          <pre>{{ interviewResponse }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api } from '../api/index.js'

const props = defineProps({ sessionId: { type: String, required: true } })

const agentId = ref('')
const data = ref(null)
const loading = ref(false)
const templates = ref({})
const interviewPrompt = ref('')
const interviewResponse = ref('')
const interviewing = ref(false)

// Edit state
const editTemplate = ref('')
const editSkepticism = ref(50)
const editSourceStrict = ref(70)
const editExperiment = ref(50)
const editReporting = ref('concise')
const editModelTier = ref('fast')

onMounted(async () => {
  try { templates.value = (await api.getTemplates()).templates } catch {}
})

async function inspect() {
  if (!agentId.value) return
  loading.value = true
  try {
    data.value = await api.inspectAgent(props.sessionId, agentId.value)
    if (data.value.persona) {
      editSkepticism.value = Math.round(data.value.persona.skepticism_level * 100)
      editSourceStrict.value = Math.round(data.value.persona.source_strictness * 100)
      editExperiment.value = Math.round(data.value.persona.experiment_appetite * 100)
      editReporting.value = data.value.persona.reporting_style
      editModelTier.value = data.value.persona.model_tier || 'fast'
    }
  } catch (e) { alert(e.message) }
  finally { loading.value = false }
}

async function savePersona() {
  try {
    const result = await api.setPersona(props.sessionId, agentId.value, {
      template: editTemplate.value || undefined,
      skepticism_level: editSkepticism.value / 100,
      source_strictness: editSourceStrict.value / 100,
      experiment_appetite: editExperiment.value / 100,
      reporting_style: editReporting.value,
      model_tier: editModelTier.value,
    })
    data.value.persona = result.persona
    alert(`Persona saved (revision ${result.persona.revision})`)
  } catch (e) { alert(e.message) }
}

async function interview() {
  interviewing.value = true
  interviewResponse.value = ''
  try {
    const result = await api.interviewAgent(props.sessionId, {
      agent_id: agentId.value,
      prompt: interviewPrompt.value,
    })
    interviewResponse.value = result.grounded_response || 'No response'
  } catch (e) { interviewResponse.value = `Error: ${e.message}` }
  finally { interviewing.value = false }
}

function confClass(c) {
  if (c >= 0.85) return 'high'
  if (c >= 0.65) return 'moderate'
  if (c >= 0.45) return 'tentative'
  return 'weak'
}
</script>

<style scoped>
.agent-inspector { }
h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
h4 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin: 16px 0 6px; }
h5 { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #999; margin: 8px 0 4px; }
.search { display: flex; gap: 8px; margin-bottom: 16px; }
.search input { flex: 1; padding: 6px 10px; border: 1px solid #ccc; font-family: monospace; font-size: 12px; }
button { padding: 6px 12px; border: 1px solid #111; background: #fff; font-family: inherit; font-size: 11px; cursor: pointer; text-transform: uppercase; }
button:hover { background: #f5f5f5; }
.section { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #f0f0f0; }
.hypothesis { display: flex; justify-content: space-between; align-items: center; }
.claim { font-size: 12px; flex: 1; }
.conf { font-family: monospace; font-size: 14px; font-weight: 700; margin-left: 8px; }
.persona .field { font-size: 11px; padding: 1px 0; }
.persona .field span { color: #666; }
.persona-editor { margin-top: 8px; padding: 8px; background: #fafafa; border: 1px solid #eee; }
.persona-editor select { width: 100%; padding: 4px; margin-bottom: 6px; font-size: 11px; border: 1px solid #ccc; }
.slider-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; font-size: 11px; }
.slider-row label { width: 100px; color: #666; }
.slider-row input[type=range] { flex: 1; }
.slider-row span { width: 35px; text-align: right; font-family: monospace; }
.save-btn { width: 100%; margin-top: 8px; background: #111; color: #fff; border: none; }
.model-tier-select { margin: 8px 0; }
.model-tier-select label { display: block; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 4px; }
.model-tier-select select { width: 100%; padding: 6px; font-size: 11px; border: 1px solid #ccc; }
.finding-card, .exp-card { display: flex; gap: 6px; align-items: center; padding: 3px 0; font-size: 11px; }
.conf-badge { font-family: monospace; font-size: 10px; padding: 1px 4px; }
.conf-badge.high { background: #e8f5e9; color: #2e7d32; }
.conf-badge.moderate { background: #e3f2fd; color: #1565c0; }
.conf-badge.tentative { background: #fff3e0; color: #e65100; }
.conf-badge.weak { background: #fce4ec; color: #c62828; }
.claim-text, .goal { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.status-dot.completed { background: #2e7d32; }
.status-dot.pending { background: #f57c00; }
.status-dot.running { background: #1565c0; }
.status-dot.failed { background: #c62828; }
.interview-btn { width: 100%; margin-top: 4px; }
.interview-response { margin-top: 8px; }
.interview-response pre { background: #fafafa; border: 1px solid #eee; padding: 8px; font-size: 11px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
.empty { font-size: 11px; color: #999; font-style: italic; }
textarea { width: 100%; padding: 6px; border: 1px solid #ccc; font-family: inherit; font-size: 12px; resize: vertical; }
</style>
