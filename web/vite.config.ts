// Vite config for the nc2000 browser demo.
//
// Two things are non-standard:
//
// 1. The wasm pkg lives OUTSIDE the web root (../crates/wasm/pkg-web) and is
//    imported by relative path. `server.fs.allow` widens dev serving to the
//    repo root; the build follows the import and emits the .wasm as a hashed
//    asset (wasm-bindgen's `new URL('..._bg.wasm', import.meta.url)` is
//    bundler-visible).
//
// 2. Battle data (meta pool + baked preview tables) is NEVER bundled: a
//    background bake keeps writing pair files into data/preview-tables-v0/,
//    so the app fetches /data/* read-only at runtime. The same middleware is
//    installed in the dev server AND the preview (built dist) server; any
//    other production server must map /data/ to the repo data/ dir.

import { defineConfig, type Plugin, type Connect } from "vite";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const webDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webDir, "..");
const dataDir = path.join(repoRoot, "data");

function serveRepoData(): Plugin {
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    const url = (req.url ?? "").split("?")[0];
    if (!url.startsWith("/data/")) return next();
    const rel = decodeURIComponent(url.slice("/data/".length));
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
