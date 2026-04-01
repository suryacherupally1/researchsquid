<template>
  <div class="status-panel">
    <h3>Session</h3>
    <div class="stat-row">
      <span class="label">Question</span>
      <span class="value question">{{ session?.question || '—' }}</span>
    </div>
    <div class="stat-row">
      <span class="label">Status</span>
      <span class="value badge" :class="statusClass">{{ session?.status || '—' }}</span>
    </div>

    <h4>Agents</h4>
    <div class="stat-row">
      <span>Active</span>
      <span class="value mono">{{ status.agents?.active || 0 }} / {{ status.agents?.total || 0 }}</span>
    </div>
    <div class="stat-row">
      <span>Sleeping</span>
      <span class="value mono">{{ status.agents?.sleeping || 0 }}</span>
    </div>

    <h4>Findings</h4>
    <div class="stat-row">
      <span>Active</span>
      <span class="value mono">{{ status.findings?.active || 0 }}</span>
    </div>
    <div class="stat-row">
      <span>Verified</span>
      <span class="value mono">{{ status.findings?.with_verification || 0 }}</span>
    </div>

    <h4>Experiments</h4>
    <div class="stat-row">
      <span>Pending</span>
      <span class="value mono">{{ status.experiments?.pending || 0 }}</span>
    </div>
    <div class="stat-row">
      <span>Completed</span>
      <span class="value mono">{{ status.experiments?.completed || 0 }}</span>
    </div>
    <div class="stat-row">
      <span>Failed</span>
      <span class="value mono">{{ status.experiments?.failed || 0 }}</span>
    </div>

    <h4>Best Answer</h4>
    <div class="best-answer">
      <span class="confidence-badge" :class="confidenceClass">
        {{ status.best_answer?.label || '—' }}
      </span>
      <span class="claim" v-if="status.best_answer?.claim">
        {{ status.best_answer.claim }}
      </span>
    </div>

    <div class="stat-row">
      <span>Contradictions</span>
      <span class="value mono">{{ status.contradictions || 0 }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({ status: { type: Object, default: () => ({}) }, session: { type: Object, default: null } })
const statusClass = computed(() => ({
  active: props.session?.status === 'active',
  stopped: props.session?.status === 'stopped',
  paused: props.session?.status === 'paused',
}))
const confidenceClass = computed(() => {
  const c = props.status.best_answer?.confidence || 0
  if (c >= 0.85) return 'high'
  if (c >= 0.65) return 'moderate'
  if (c >= 0.45) return 'tentative'
  return 'weak'
})
</script>

<style scoped>
.status-panel { }
h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #eee; }
h4 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin: 16px 0 6px; }
.stat-row { display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; }
.label { color: #666; }
.value { font-weight: 500; }
.mono { font-family: 'JetBrains Mono', monospace; }
.question { font-size: 11px; color: #333; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge { padding: 2px 8px; font-size: 10px; text-transform: uppercase; }
.badge.active { background: #e8f5e9; color: #2e7d32; }
.badge.stopped { background: #fce4ec; color: #c62828; }
.badge.paused { background: #fff3e0; color: #e65100; }
.best-answer { margin: 6px 0; }
.confidence-badge { display: inline-block; padding: 2px 8px; font-size: 10px; text-transform: uppercase; font-weight: 600; }
.confidence-badge.high { background: #e8f5e9; color: #2e7d32; }
.confidence-badge.moderate { background: #e3f2fd; color: #1565c0; }
.confidence-badge.tentative { background: #fff3e0; color: #e65100; }
.confidence-badge.weak { background: #fce4ec; color: #c62828; }
.claim { display: block; font-size: 11px; color: #555; margin-top: 4px; }
</style>
