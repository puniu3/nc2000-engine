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
// 2. Battle data (team files, prior, dex tables) is NEVER bundled: the app
//    fetches <base>data/* read-only at runtime. The same
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
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

const webDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webDir, "..");
const dataDir = path.join(repoRoot, "data");
const base = process.env.NC2000_BASE ?? "/";
// Research builds execute live search exclusively; the default production
// build still requires a freshly baked, source-matching preview artifact.
const compiledPreviewEnabled = process.env.NC2000_DISABLE_COMPILED_PREVIEW !== "1";

function serveRepoData(): Plugin {
  // The app fetches `${BASE_URL}data/...`; accept both the based path and
  // a bare /data/ (robust when the dist is re-served under another base).
  // The resolved base wins over NC2000_BASE when a caller overrides it.
  let prefixes = [...new Set([`${base}data/`, "/data/"])];
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
    configResolved(config) {
      prefixes = [...new Set([`${config.base}data/`, "/data/"])];
    },
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

// A production bundle must contain a complete artifact from these exact
// policy inputs. Source changes require rebaking or a separately audited
// exact-behavior migration; stale inputs are never accepted.
function verifyCompiledPreview(): Plugin {
  return {
    name: "verify-compiled-preview",
    apply: "build",
    buildStart() {
      if (!compiledPreviewEnabled) return;
      const artifact = JSON.parse(fs.readFileSync(path.join(dataDir, "compiled-preview-v1.json"), "utf8"));
      const catalog = JSON.parse(fs.readFileSync(path.join(dataDir, "team-pool-v2/team-pool.json"), "utf8"));
      const profiles = JSON.parse(fs.readFileSync(path.join(dataDir, "search-profiles.json"), "utf8"));
      const publicRosters = new Set(catalog.teams.map((t: { sets: { species: string; level: number }[] }) => JSON.stringify(t.sets.map(p => [p.species, p.level]))));
      if (artifact.matchups !== catalog.teams.length * publicRosters.size)
        throw new Error("Compiled previews do not cover the complete catalog matchup grid.");
      if (artifact.format !== "nc2000-compiled-preview-v1" || !artifact.inputs || !artifact.matchups || Object.keys(artifact.policies ?? {}).length !== artifact.matchups)
        throw new Error("Compiled previews are incomplete. Run compile_preview and tools/compile-preview-artifact.py.");
      if (artifact.c !== profiles.blind.c) throw new Error("Compiled previews use a different search coefficient. Rebuild the preview artifact.");
      for (const [relative, expected] of Object.entries(artifact.inputs)) {
        const actual = createHash("sha256").update(fs.readFileSync(path.join(repoRoot, relative))).digest("hex");
        if (actual !== expected) throw new Error(`Stale compiled previews: ${relative}. Rebuild the preview artifact.`);
      }
    },
  };
}

function evaluatorBuildId(): string {
  const hash = createHash("sha256");
  const add = (relative: string) => {
    const absolute = path.join(repoRoot, relative);
    if (!fs.existsSync(absolute)) return;
    if (fs.statSync(absolute).isDirectory()) {
      for (const child of fs.readdirSync(absolute).sort())
        add(`${relative}/${child}`);
    } else {
      hash.update(relative).update(fs.readFileSync(absolute));
    }
  };
  for (const relative of [
    "web/src",
    "web/vite.config.ts",
    "web/package-lock.json",
    "Cargo.lock",
    "Cargo.toml",
    "data/live-search.json",
    "data/search-profiles.json",
    "crates/engine/src",
    "crates/bot/src",
    "crates/wasm/src",
    "crates/wasm/pkg-web/nc2000_wasm_bg.wasm",
  ])
    add(relative);
  hash.update(process.env.VITE_NC2000_TEST_BUDGET ?? "production");
  return hash.digest("hex");
}

export default defineConfig({
  base,
  define: {
    __EVALUATOR_BUILD__: JSON.stringify(evaluatorBuildId()),
    __COMPILED_PREVIEW_ENABLED__: compiledPreviewEnabled,
  },
  plugins: [verifyCompiledPreview(), serveRepoData()],
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
