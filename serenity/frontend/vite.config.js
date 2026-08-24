import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: Number(process.env.FRONTEND_PORT || 5173),
    // Sharing the app remotely means exposing it through a tunnel. Proxying the
    // API through the dev server keeps everything on one origin, so a single
    // tunnel is enough and CORS never enters into it.
    proxy: {
      '/chat': { target: process.env.BACKEND_PROXY_TARGET || 'http://backend:8000', changeOrigin: true },
      '/health': { target: process.env.BACKEND_PROXY_TARGET || 'http://backend:8000', changeOrigin: true },
    },
    // Tunnels hand out hostnames that cannot be known ahead of time, and Vite
    // rejects unknown Host headers by default.
    allowedHosts: process.env.VITE_ALLOWED_HOSTS
      ? process.env.VITE_ALLOWED_HOSTS.split(',').map((host) => host.trim())
      : true,
    watch: {
      // docker-compose bind-mounts ./frontend into the container. Filesystem
      // events do not cross a Windows host into the Linux VM, so without
      // polling Vite never notices an edit and keeps serving the previous
      // module - changes appear to have no effect until the container is
      // restarted. Opt out with VITE_DISABLE_POLLING=1 on native Linux/macOS.
      usePolling: process.env.VITE_DISABLE_POLLING !== '1',
      interval: 300,
    },
  },
  preview: {
    host: '0.0.0.0',
    port: Number(process.env.FRONTEND_PREVIEW_PORT || 4173),
  },
});
