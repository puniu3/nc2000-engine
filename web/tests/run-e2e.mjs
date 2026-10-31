import { spawn } from "node:child_process";
import { preview } from "vite";

const server = await preview({ preview: { host: "127.0.0.1", port: 0 } });
const address = server.httpServer.address();
if (!address || typeof address === "string") throw new Error("missing preview address");
const tests = spawn("npx", ["playwright", "test", ...process.argv.slice(2)], {
  stdio: "inherit",
  env: { ...process.env, NC2000_E2E_PORT: String(address.port) },
});
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => { tests.kill(signal); server.httpServer.close(); });
}
try {
  process.exitCode = await new Promise((resolve, reject) => {
    tests.once("error", reject);
    tests.once("exit", code => resolve(code ?? 1));
  });
} finally {
  await new Promise(resolve => server.httpServer.close(resolve));
}
