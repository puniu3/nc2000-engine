// Vite config for the nc2000 browser demo.
//
// Three things are non-standard:
//
// 1. The wasm pkg lives OUTSIDE the web root (../crates/wasm/pkg-web) and is
//    imported by relative path. `server.fs.allow` widens dev serving to the
//    repo root; the build follows the import and emits the .wasm as a hashed
//    asset (wasm-bindgen's `new URL('..._bg.wasm', import.meta.url)` is
//    bundler-visible).
//
// 2. Battle data (meta pool + baked preview tables) is NEVER bundled: a
//    background bake keeps writing pair files into data/preview-tables-v0/,
//    so the app fetches <base>data/* read-only at runtime. The same
//    middleware is installed in the dev server AND the preview (built dist)
//    server; the GH Pages build instead copies data/ into dist/data/
//    (.github/workflows/pages.yml), and any other production server must
//    map <base>data/ to the repo data/ dir.
//
// 3. The deploy base is env-driven: `NC2000_BASE=/nc2000-engine/` for the
//    GH Pages build (project pages live under a subpath), unset = `/` for
//    local dev/preview. Runtime data fetches follow via
//    `import.meta.env.BASE_URL` (src/data.ts).

import { defineConfig, type Plugin, type Connect } from "vite";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const webDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webDir, "..");
const dataDir = path.join(repoRoot, "data");
const base = process.env.NC2000_BASE ?? "/";

function serveRepoData(): Plugin {
  // The app fetches `${BASE_URL}data/...`; accept both the based path and
  // a bare /data/ (robust when the dist is re-served under another base).
  const prefixes = [...new Set([`${base}data/`, "/data/"])];
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    const url = (req.url ?? "").split("?")[0];
    const prefix = prefixes.find((p) => url.startsWith(p));
    if (!prefix) return next();
    const rel = decodeURIComponent(url.slice(prefix.length));
    const file = path.normalize(path.join(dataDir, rel));
    if (!file.startsWith(dataDir + path.sep) || !file.endsWith(".json")) {
      res.statusCode = 403;
      res.end("forbidden");
      return;
    }
    let stat;
    try {
      stat = fs.statSync(file);
    } catch {
      res.statusCode = 404;
      res.end("not found");
      return;
    }
    if (!stat.isFile()) {
      res.statusCode = 404;
      res.end("not found");
      return;
    }
    res.setHeader("Content-Type", "application/json");
    fs.createReadStream(file).pipe(res);
  };
  return {
    name: "serve-repo-data",
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

export default defineConfig({
  base,
  plugins: [serveRepoData()],
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "preact",
  },
  server: {
    fs: { allow: [repoRoot] },
  },
  worker: {
    format: "es",
  },
  build: {
    target: "es2020",
  },
});
