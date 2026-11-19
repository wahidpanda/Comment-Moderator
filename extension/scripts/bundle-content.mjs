import * as esbuild from "esbuild";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

for (const name of ["youtube", "linkedin"]) {
  await esbuild.build({
    entryPoints: [resolve(root, `src/content/${name}.ts`)],
    bundle: true,
    format: "iife",
    platform: "browser",
    target: "chrome100",
    outfile: resolve(root, `dist/${name}.js`),
  });
  console.log(`Bundled ${name}.js (IIFE, no imports)`);
}