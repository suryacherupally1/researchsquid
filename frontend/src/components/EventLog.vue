<template>
  <div class="event-log">
    <h3>Live Events</h3>
    <div class="events">
      <div v-for="(ev, i) in events" :key="i" class="event" :class="ev.event_type">
        <span class="time">{{ formatTime(ev.timestamp) }}</span>
        <span class="type">{{ ev.event_type }}</span>
        <span class="detail" v-if="ev.data?.claim">{{ truncate(ev.data.claim, 60) }}</span>
        <span class="detail" v-else-if="ev.data?.spec_id">spec: {{ ev.data.spec_id }}</span>
        <span class="agent" v-if="ev.agent_id">{{ ev.agent_id }}</span>
      </div>
      <div v-if="!events.length" class="empty">No events yet</div>
    </div>
  </div>
</template>

<script setup>
defineProps({ events: { type: Array, default: () => [] } })
function formatTime(ts) { return ts ? ts.slice(11, 19) : '' }
function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + '...' : s }
</script>

<style scoped>
.event-log { margin-top: 16px; }
h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 8px; }
.events { max-height: 400px; overflow-y: auto; }
.event { display: flex; gap: 6px; padding: 4px 0; border-bottom: 1px solid #f5f5f5; font-size: 11px; align-items: center; }
.time { color: #999; font-family: monospace; width: 55px; flex-shrink: 0; }
.type { color: #111; font-weight: 500; width: 150px; flex-shrink: 0; }
.detail { color: #555; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.agent { color: #999; font-size: 10px; flex-shrink: 0; }
.event.finding_posted { border-left: 3px solid #2e7d32; padding-left: 6px; }
.event.contradiction_opened { border-left: 3px solid #c62828; padding-left: 6px; }
.event.experiment_submitted { border-left: 3px solid #1565c0; padding-left: 6px; }
.event.experiment_completed { border-left: 3px solid #2e7d32; padding-left: 6px; }
.event.experiment_failed { border-left: 3px solid #c62828; padding-left: 6px; }
.empty { color: #999; font-size: 12px; font-style: italic; padding: 8px 0; }
</style>
