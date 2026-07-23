import { spawn } from "node:child_process";
import { createServer } from "node:net";

function stopGroup(child) {
  if (!child.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    // Already exited.
  }
}

async function freePort() {
  for (let port = 8000; port < 8100; port++) {
    const free = await new Promise((resolve) => {
      const probe = createServer();
      probe.once("error", () => resolve(false));
      probe.listen(port, "0.0.0.0", () =>
        probe.close(() => resolve(true)),
      );
    });
    if (free) return port;
  }
  throw new Error("no free E2E port in 8000-8099");
}

const port = await freePort();
const server = spawn(
  "npx",
  ["vite", "preview", "--host", "0.0.0.0", "--port", String(port), "--strictPort"],
  {
    detached: true,
    stdio: ["ignore", "pipe", "inherit"],
  },
);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => {
    stopGroup(server);
    process.exit(signal === "SIGINT" ? 130 : 143);
  });
}

try {
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("preview startup timeout")), 30_000);
    server.once("exit", (code) =>
      reject(new Error(`preview exited before startup (${code})`)),
    );
    server.stdout.on("data", (chunk) => {
      const text = String(chunk);
      process.stdout.write(text);
      if (text.includes("Local:")) {
        clearTimeout(timer);
        resolve();
      }
    });
  });

  const tests = spawn("npx", ["playwright", "test"], {
    stdio: "inherit",
    env: { ...process.env, NC2000_E2E_PORT: String(port) },
  });
  const code = await new Promise((resolve, reject) => {
    tests.once("error", reject);
    tests.once("exit", (status) => resolve(status ?? 1));
  });
  process.exitCode = code;
} finally {
  stopGroup(server);
}
