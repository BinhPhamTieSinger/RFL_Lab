import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import YAML from "yaml";
const root = path.resolve("../..");
const destination = path.resolve("public/research");
fs.mkdirSync(destination, { recursive: true });
const manifest = JSON.parse(
  fs.readFileSync("research/history-manifest.json", "utf8"),
);
const selectedSources = new Set(manifest.map((entry) => entry.source));
const additional = JSON.parse(
  fs.readFileSync("research/additional-history.json", "utf8"),
);
const entries = [];
for (const selection of manifest) {
  const folder = path.resolve(root, selection.source);
  if (
    !folder.startsWith(root + path.sep) ||
    !selection.source.startsWith("outputs")
  )
    throw Error("Invalid history source");
  if (!fs.existsSync(folder)) continue;
  const source = path.relative(root, folder).split(path.sep).join("/");
  const id =
    "archive-" +
    crypto.createHash("sha256").update(source).digest("hex").slice(0, 24);
  const configPath = path.join(folder, "config.yaml");
  const resultPath = path.join(folder, "metrics.json");
  let config = {},
    result = null,
    error;
  try {
    config = fs.existsSync(configPath)
      ? (YAML.parse(fs.readFileSync(configPath, "utf8")) ?? {})
      : {};
    // Python writes NaN/Infinity for undefined diagnostics; JSON has no such values, so publish null.
    result = fs.existsSync(resultPath)
      ? JSON.parse(
          fs
            .readFileSync(resultPath, "utf8")
            .replace(/([:,[]\s*)-?(?:NaN|Infinity)(?=\s*[,\]}])/g, "$1null"),
        )
      : null;
  } catch (e) {
    error = e.message;
  }
  const entry = {
    id,
    source,
    archived: true,
    name: selection.title ?? config.name ?? path.basename(folder),
    notes: selection.notes,
    status: error ? "unreadable" : result?.final ? "completed" : "incomplete",
    created:
      fs.statSync(fs.existsSync(resultPath) ? resultPath : configPath).mtimeMs /
      1000,
    date_basis: "file modification time",
    config: {
      method: config.method ?? result?.method ?? "unknown",
      overrides: Object.fromEntries(
        Object.entries(config)
          .filter(([, value]) => value && typeof value === "object")
          .flatMap(([section, values]) =>
            Object.entries(values).map(([key, value]) => [
              `${section}.${key}`,
              value,
            ]),
          ),
      ),
    },
    error,
  };
  entries.push(entry);
  fs.writeFileSync(
    path.join(destination, id + ".json"),
    JSON.stringify({ ...entry, result, logs: [] }),
  );
}
for (const saved of additional) {
  if (
    !selectedSources.has(saved.source) ||
    entries.some((entry) => entry.id === saved.id)
  )
    continue;
  const { result, logs, ...entry } = saved;
  entries.push(entry);
  fs.writeFileSync(
    path.join(destination, saved.id + ".json"),
    JSON.stringify(saved),
  );
}
fs.writeFileSync(
  path.join(destination, "history.json"),
  JSON.stringify(entries),
);
console.log(`Exported ${entries.length} research experiments`);
// The manifest is curated by hand, so name completed runs it does not list yet.
const unlisted = fs
  .readdirSync(root)
  .filter((name) => name.startsWith("outputs"))
  .flatMap((name) =>
    fs
      .readdirSync(path.join(root, name), { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => `${name}/${entry.name}`),
  )
  .filter(
    (source) =>
      !selectedSources.has(source) &&
      fs.existsSync(path.join(root, source, "metrics.json")),
  );
if (unlisted.length)
  console.warn(
    `${unlisted.length} completed runs are not in research/history-manifest.json: ${unlisted.join(", ")}`,
  );

const captureRoot = path.resolve("research/captures");
fs.mkdirSync(path.join(destination, "captures"), { recursive: true });
const captures = [];
if (fs.existsSync(captureRoot)) {
  for (const name of fs
    .readdirSync(captureRoot)
    .filter((name) => name.endsWith(".json"))) {
    const capture = JSON.parse(
      fs.readFileSync(path.join(captureRoot, name), "utf8"),
    );
    fs.copyFileSync(
      path.join(captureRoot, name),
      path.join(destination, "captures", name),
    );
    captures.push({
      id: capture.id,
      method: capture.config.method,
      attack: capture.config.overrides["attack.kind"],
      config: capture.config,
    });
  }
}
fs.writeFileSync(
  path.join(destination, "captures.json"),
  JSON.stringify(captures),
);
const documents = [];
for (const relative of [
  "README.md",
  ...fs
    .readdirSync(path.join(root, "docs"))
    .filter((name) => name.endsWith(".md"))
    .map((name) => "docs/" + name),
  "visualizer/README.md",
]) {
  if (!fs.existsSync(path.join(root, relative))) continue;
  const content = fs.readFileSync(path.join(root, relative), "utf8");
  documents.push({
    path: relative,
    title: content.match(/^#\s+(.+)$/m)?.[1] ?? relative,
    content,
  });
}
fs.writeFileSync(
  path.join(destination, "documentation.json"),
  JSON.stringify(documents),
);
// Preserve relative image and static HTML paths in the published Markdown documents.
function copyImages(folder) {
  for (const entry of fs.readdirSync(folder, { withFileTypes: true })) {
    const location = path.join(folder, entry.name);
    if (entry.isDirectory()) copyImages(location);
    else if (/\.(png|svg|jpe?g|webp|html)$/i.test(entry.name)) {
      const target = path.join(
        destination,
        "documents",
        path.relative(root, location),
      );
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.copyFileSync(location, target);
    }
  }
}
copyImages(path.join(root, "docs"));
console.log(
  `Published ${captures.length} visualization captures and ${documents.length} Markdown documents`,
);
