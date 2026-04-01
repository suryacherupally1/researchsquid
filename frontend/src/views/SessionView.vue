<template>
  <div class="session-view">
    <!-- Header -->
    <header class="header">
      <div class="header-left">
        <span class="brand" @click="$router.push('/')">HIVERESEARCH</span>
      </div>
      <div class="header-center">
        <WorkflowSteps :current="store.status.workflow_step" :labels="store.WORKFLOW_LABELS" />
      </div>
      <div class="header-right">
        <BudgetBar :budget="store.status.budget" />
      </div>
    </header>

    <!-- Main Content -->
    <div class="content">
      <!-- Left: Session Status + Events -->
      <div class="panel left">
        <StatusPanel :status="store.status" :session="store.session" />
        <EventLog :events="store.events" />
      </div>

      <!-- Center: Workflow Step Content -->
      <div class="panel center">
        <AgentPanel v-if="store.status.workflow_step >= 3"
          :sessionId="sessionId" />
        <ExperimentPanel v-if="store.status.workflow_step >= 4"
          :status="store.status" />
        <DebatePanel v-if="store.status.workflow_step >= 5"
          :contradictions="store.status.contradictions" />
        <ReportPanel v-if="store.status.workflow_step >= 6"
          :sessionId="sessionId" />
      </div>

      <!-- Right: Agent Inspector -->
      <div class="panel right">
        <AgentInspector :sessionId="sessionId" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { useSessionStore } from '../store/session.js'
import WorkflowSteps from '../components/WorkflowSteps.vue'
import BudgetBar from '../components/BudgetBar.vue'
import StatusPanel from '../components/StatusPanel.vue'
import EventLog from '../components/EventLog.vue'
import AgentPanel from '../components/AgentPanel.vue'
import ExperimentPanel from '../components/ExperimentPanel.vue'
import DebatePanel from '../components/DebatePanel.vue'
import ReportPanel from '../components/ReportPanel.vue'
import AgentInspector from '../components/AgentInspector.vue'

const route = useRoute()
const store = useSessionStore()
const sessionId = computed(() => route.params.id)

onMounted(() => store.load(sessionId.value))
onUnmounted(() => store.disconnect())
</script>

<style scoped>
.session-view { min-height: 100vh; display: flex; flex-direction: column; }
.header { display: flex; align-items: center; padding: 12px 24px; border-bottom: 2px solid #111; background: #fff; }
.header-left { flex: 0 0 auto; }
.brand { font-size: 16px; font-weight: 700; letter-spacing: 3px; cursor: pointer; }
.header-center { flex: 1; display: flex; justify-content: center; }
.header-right { flex: 0 0 auto; }
.content { display: flex; flex: 1; overflow: hidden; }
.panel { overflow-y: auto; padding: 16px; }
.left { flex: 0 0 280px; border-right: 1px solid #eee; }
.center { flex: 1; }
.right { flex: 0 0 320px; border-left: 1px solid #eee; }
</style>
