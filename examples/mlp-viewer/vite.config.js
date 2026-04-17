import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  resolve: {
    alias: {
      // Use the local (modified) gsplat source instead of the npm package.
      // This gives access to MLPLoader, SplatMLPData, MLP utilities, etc.
      'gsplat': resolve(__dirname, '../../src/index.ts'),
    },
  },
  // Serve web_data/ as the static public directory so that
  // scene.splat-mlp and weights.json are available at the root URL.
  publicDir: resolve(__dirname, '../../web_data'),
  build: {
    target: 'esnext',
  },
  optimizeDeps: {
    exclude: ['gsplat'],
  },
});
