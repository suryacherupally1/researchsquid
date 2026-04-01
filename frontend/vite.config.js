import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
      '/session': 'http://localhost:8000',
      '/research': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
