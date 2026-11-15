import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import YAML from "yaml";
const root = path.resolve("../..");
const destination = path.resolve("public/research");
fs.mkdirSync(destination, { recursive: true });
function walk(folder) {
  return fs.readdirSync(folder, { withFileTypes: true }).flatMap((entry) => {
    const location = path.join(folder, entry.name);
    return entry.isDirectory()
      ? walk(location)
      : ["metrics.json", "config.yaml"].includes(entry.name)
        ? [location]
        : [];
  });
}
const files = fs
  .readdirSync(root, { withFileTypes: true })
  .filter((x) => x.isDirectory() && x.name.startsWith("outputs"))
  .flatMap((x) => walk(path.join(root, x.name)));
const entries = [];
for (const folder of [
  ...new Set(files.map((file) => path.dirname(file))),
].sort()) {
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
    result = fs.existsSync(resultPath)
      ? JSON.parse(fs.readFileSync(resultPath, "utf8"))
      : null;
  } catch (e) {
    error = e.message;
  }
  const entry = {
    id,
    source,
    archived: true,
    name: config.name ?? path.basename(folder),
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
const recorded = JSON.parse(fs.readFileSync('research/recorded-round.json', 'utf8'));
const {result: recordedResult, events: recordedEvents, ...recordedEntry} = recorded;
entries.unshift(recordedEntry);
const additional = JSON.parse(fs.readFileSync('research/additional-history.json', 'utf8'));
for (const saved of additional) {
  if (entries.some(entry => entry.id === saved.id)) continue;
  const {result, logs, ...entry} = saved;
  entries.push(entry);
  fs.writeFileSync(path.join(destination, saved.id + '.json'), JSON.stringify(saved));
}
fs.writeFileSync(
  path.join(destination, "history.json"),
  JSON.stringify(entries),
);
console.log(`Exported ${entries.length} research experiments`);
