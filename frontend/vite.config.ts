import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    open: true, // Automatically open the app in the default browser on startup
    port: 5173, // (Optional) Specify a custom port if needed
  },
});
