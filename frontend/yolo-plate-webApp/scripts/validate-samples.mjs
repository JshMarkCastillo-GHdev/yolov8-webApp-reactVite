import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = join(root, "public/samples/samples.json");

const raw = readFileSync(manifestPath, "utf8");
const samples = JSON.parse(raw);

if (!Array.isArray(samples)) {
  throw new Error("samples.json must be a JSON array");
}

const required = ["id", "src", "title"];

for (const [index, entry] of samples.entries()) {
  if (typeof entry !== "object" || entry === null) {
    throw new Error(`Entry ${index}: must be an object`);
  }

  for (const key of required) {
    if (typeof entry[key] !== "string" || entry[key].length === 0) {
      throw new Error(`Entry ${index}: missing or invalid "${key}"`);
    }
  }

  if (!entry.src.startsWith("/samples/")) {
    throw new Error(`Entry ${index}: src must start with /samples/`);
  }

  const filename = entry.src.replace("/samples/", "");
  const imagePath = join(root, "public/samples", filename);
  if (!existsSync(imagePath)) {
    throw new Error(`Entry ${index}: file not found at public/samples/${filename}`);
  }
}

console.log(`samples.json OK (${samples.length} entries)`);
