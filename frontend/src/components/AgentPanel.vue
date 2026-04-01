<template>
  <div class="agent-panel">
    <h3>Agents</h3>
    <div class="agent-list">
      <div v-for="agent in agents" :key="agent.id" class="agent-card"
        :class="{ selected: selectedId === agent.id }"
        @click="$emit('select', agent.id)">
        <div class="agent-header">
          <span class="agent-id">{{ agent.id }}</span>
          <span class="status-badge" :class="agent.status">{{ agent.status }}</span>
        </div>
        <div class="agent-stats">
          <span>{{ agent.findings_posted || 0 }} findings</span>
          <span>{{ agent.experiments_submitted || 0 }} experiments</span>
        </div>
        <div class="agent-rep" v-if="agent.reputation !== undefined">
          Rep: {{ agent.reputation }}
        </div>
      </div>
      <div v-if="!agents.length" class="empty">No agents spawned yet</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api } from '../api/index.js'

const props = defineProps({
  sessionId: { type: String, required: true },
  selectedId: { type: String, default: null },
})
defineEmits(['select'])

const agents = ref([])

onMounted(async () => {
  try {
    const s = await api.getSessionStatus(props.sessionId)
    // Agent list would come from a dedicated endpoint
    // For now, show placeholder
  } catch {}
})
</script>

<style scoped>
.agent-panel h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.agent-list { display: flex; flex-direction: column; gap: 8px; }
.agent-card { padding: 10px; border: 1px solid #eee; cursor: pointer; }
.agent-card:hover { border-color: #999; }
.agent-card.selected { border-color: #111; background: #fafafa; }
.agent-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.agent-id { font-family: monospace; font-size: 12px; font-weight: 600; }
.status-badge { font-size: 10px; padding: 1px 6px; text-transform: uppercase; }
.status-badge.researching { background: #e8f5e9; color: #2e7d32; }
.status-badge.sleeping { background: #f5f5f5; color: #666; }
.status-badge.stopped { background: #fce4ec; color: #c62828; }
.agent-stats { font-size: 11px; color: #666; display: flex; gap: 12px; }
.agent-rep { font-size: 10px; color: #999; margin-top: 2px; }
.empty { color: #999; font-size: 12px; font-style: italic; }
</style>
