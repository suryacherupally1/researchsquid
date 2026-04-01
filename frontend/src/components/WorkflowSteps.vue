<template>
  <div class="workflow-steps">
    <div v-for="(label, idx) in labels" :key="idx"
      class="step"
      :class="{ active: idx === current, completed: idx < current, future: idx > current }">
      <span class="step-num">{{ idx }}</span>
      <span class="step-label">{{ label }}</span>
    </div>
  </div>
</template>

<script setup>
defineProps({
  current: { type: Number, default: 0 },
  labels: { type: Object, default: () => ({}) },
})
</script>

<style scoped>
.workflow-steps { display: flex; gap: 4px; align-items: center; }
.step { display: flex; align-items: center; gap: 4px; padding: 4px 10px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #999; border: 1px solid transparent; }
.step.active { color: #111; border-color: #111; font-weight: 600; }
.step.completed { color: #555; }
.step-num { font-size: 10px; width: 16px; height: 16px; display: flex; align-items: center; justify-content: center; border: 1px solid currentColor; border-radius: 50%; }
</style>
