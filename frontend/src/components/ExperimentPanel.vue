<template>
  <div class="experiment-panel">
    <h3>Experiments</h3>
    <div class="exp-stats">
      <div class="stat">
        <span class="num">{{ status.experiments?.pending || 0 }}</span>
        <span class="lbl">Pending</span>
      </div>
      <div class="stat">
        <span class="num">{{ status.experiments?.running || 0 }}</span>
        <span class="lbl">Running</span>
      </div>
      <div class="stat success">
        <span class="num">{{ status.experiments?.completed || 0 }}</span>
        <span class="lbl">Completed</span>
      </div>
      <div class="stat danger">
        <span class="num">{{ status.experiments?.failed || 0 }}</span>
        <span class="lbl">Failed</span>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({ status: { type: Object, default: () => ({}) } })
</script>

<style scoped>
.experiment-panel { margin: 24px 0; }
h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.exp-stats { display: flex; gap: 16px; }
.stat { text-align: center; padding: 12px 20px; border: 1px solid #eee; }
.num { display: block; font-size: 24px; font-weight: 700; font-family: monospace; }
.lbl { font-size: 10px; text-transform: uppercase; color: #666; letter-spacing: 1px; }
.stat.success .num { color: #2e7d32; }
.stat.danger .num { color: #c62828; }
</style>
