import { defineConfig } from "@playwright/test";

const port = Number(process.env.NC2000_E2E_PORT ?? 8000);

// Sandboxed runners (e.g. Claude Code on the web) pre-install one Chromium
// and block Playwright's own browser download; this points the launch at it
// without touching the default resolution anywhere else.
const chromiumExe = process.env.NC2000_CHROMIUM_EXE;

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 12 * 60 * 1000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    headless: true,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    ...(chromiumExe ? { launchOptions: { executablePath: chromiumExe } } : {}),
  },
});
