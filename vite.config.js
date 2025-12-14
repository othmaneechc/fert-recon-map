import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  publicDir: 'public',
  // Serve the data folder as static assets
  server: {
    fs: {
      // Allow serving files from the data directory
      allow: ['..'],
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  // Copy data folder to dist on build
  build: {
    rollupOptions: {
      external: [],
    },
  },
})
