import { copyFileSync, readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(readFileSync(resolve(root, "manifest.json"), "utf-8"));
manifest.background.service_worker = "background.js";
manifest.content_scripts = [
  { matches: ["https://www.youtube.com/*"], js: ["youtube.js"], run_at: "document_idle" },
  { matches: ["https://www.linkedin.com/*"], js: ["linkedin.js"], run_at: "document_idle" },
];
const dist = resolve(root, "dist");
writeFileSync(resolve(dist, "manifest.json"), JSON.stringify(manifest, null, 2));
let panel = readFileSync(resolve(dist, "sidepanel.html"), "utf-8");
if (panel.includes("/src/sidepanel.ts")) {
  panel = panel.replace('src="/src/sidepanel.ts"', 'src="./sidepanel.js"');
  writeFileSync(resolve(dist, "sidepanel.html"), panel);
}
console.log("Copied manifest.json to dist/");