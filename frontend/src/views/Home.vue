<template>
  <div class="home">
    <header class="header">
      <div class="brand">HIVERESEARCH</div>
      <div class="tagline">Tier 1 proposes. Tier 2 validates. DAG remembers.</div>
    </header>

    <main class="main">
      <div class="start-card">
        <h2>Start Research</h2>
        <form @submit.prevent="startSession">
          <div class="field">
            <label>Research Question</label>
            <textarea v-model="question" rows="3"
              placeholder="e.g. What is a cheaper alternative to acetaminophen with similar efficacy?">
            </textarea>
          </div>
          <div class="row">
            <div class="field half">
              <label>Modality</label>
              <select v-model="modality">
                <option value="general">General Research</option>
                <option value="llm_optimization">LLM Optimization</option>
                <option value="drug_discovery">Drug Discovery</option>
                <option value="engineering_simulation">Engineering Simulation</option>
              </select>
            </div>
            <div class="field half">
              <label>Agents</label>
              <input type="number" v-model.number="agentCount" min="1" max="25" />
            </div>
          </div>
          <div class="row">
            <div class="field half">
              <label>LLM Budget ($)</label>
              <input type="number" v-model.number="llmBudget" min="1" step="5" />
            </div>
            <div class="field half">
              <label>Compute Budget ($)</label>
              <input type="number" v-model.number="computeBudget" min="0" step="5" />
            </div>
          </div>
          <button type="submit" :disabled="!question || loading">
            {{ loading ? 'Starting...' : 'Start Research' }}
          </button>
        </form>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api/index.js'

const router = useRouter()
const question = ref('')
const modality = ref('general')
const agentCount = ref(5)
const llmBudget = ref(20)
const computeBudget = ref(20)
const loading = ref(false)

async function startSession() {
  loading.value = true
  try {
    const session = await api.startResearch({
      question: question.value,
      modality: modality.value,
      agent_count: agentCount.value,
      llm_budget_usd: llmBudget.value,
      compute_budget_usd: computeBudget.value,
    })
    router.push(`/session/${session.id}`)
  } catch (e) {
    alert(`Error: ${e.message}`)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.home { min-height: 100vh; background: #fafafa; }
.header { padding: 40px; text-align: center; border-bottom: 2px solid #111; }
.brand { font-size: 28px; font-weight: 700; letter-spacing: 4px; }
.tagline { margin-top: 8px; color: #666; font-size: 14px; }
.main { max-width: 640px; margin: 40px auto; padding: 0 20px; }
.start-card { background: #fff; border: 1px solid #ddd; padding: 32px; }
h2 { font-size: 18px; margin-bottom: 20px; }
.field { margin-bottom: 16px; }
.field label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 4px; }
textarea, input, select { width: 100%; padding: 8px 12px; border: 1px solid #ccc; font-family: inherit; font-size: 14px; }
textarea:focus, input:focus, select:focus { outline: none; border-color: #111; }
.row { display: flex; gap: 16px; }
.half { flex: 1; }
button { width: 100%; padding: 12px; background: #111; color: #fff; border: none; font-family: inherit; font-size: 14px; cursor: pointer; text-transform: uppercase; letter-spacing: 1px; }
button:hover { background: #333; }
button:disabled { background: #999; cursor: not-allowed; }
</style>
