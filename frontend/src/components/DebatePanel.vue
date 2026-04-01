<template>
  <div class="debate-panel">
    <h3>Debates & Contradictions</h3>
    <div class="stat">
      <span class="num">{{ contradictions }}</span>
      <span class="lbl">Active Contradictions</span>
    </div>
    <p class="hint" v-if="contradictions > 0">
      Agents are disagreeing. CONTRADICTS edges require counter-claims.
    </p>
    <p class="hint" v-else>
      No active contradictions. Research consensus or early stage.
    </p>
  </div>
</template>

<script setup>
defineProps({ contradictions: { type: Number, default: 0 } })
</style>

<style scoped>
.debate-panel { margin: 24px 0; }
h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.stat { display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px; }
.num { font-size: 32px; font-weight: 700; font-family: monospace; color: #c62828; }
.lbl { font-size: 12px; color: #666; text-transform: uppercase; }
.hint { font-size: 12px; color: #888; font-style: italic; }
</style>
