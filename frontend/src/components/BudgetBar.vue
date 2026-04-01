<template>
  <div class="budget-bar">
    <div class="budget-item">
      <span class="label">LLM</span>
      <div class="bar">
        <div class="fill" :style="{ width: llmPct + '%', background: llmColor }"></div>
      </div>
      <span class="value">${{ budget.llm_spent?.toFixed(2) }} / ${{ budget.llm_budget }}</span>
    </div>
    <div class="budget-item">
      <span class="label">Compute</span>
      <div class="bar">
        <div class="fill" :style="{ width: computePct + '%', background: computeColor }"></div>
      </div>
      <span class="value">${{ budget.compute_spent?.toFixed(2) }} / ${{ budget.compute_budget }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({ budget: { type: Object, default: () => ({}) } })
const llmPct = computed(() => props.budget.llm_budget ? (props.budget.llm_spent / props.budget.llm_budget) * 100 : 0)
const computePct = computed(() => props.budget.compute_budget ? (props.budget.compute_spent / props.budget.compute_budget) * 100 : 0)
const llmColor = computed(() => llmPct.value > 90 ? '#d32f2f' : llmPct.value > 70 ? '#f57c00' : '#111')
const computeColor = computed(() => computePct.value > 90 ? '#d32f2f' : computePct.value > 70 ? '#f57c00' : '#111')
</script>

<style scoped>
.budget-bar { display: flex; gap: 16px; }
.budget-item { display: flex; align-items: center; gap: 6px; font-size: 11px; }
.label { text-transform: uppercase; color: #666; width: 55px; }
.bar { width: 80px; height: 6px; background: #eee; }
.fill { height: 100%; transition: width 0.5s; }
.value { font-size: 10px; color: #666; white-space: nowrap; }
</style>
