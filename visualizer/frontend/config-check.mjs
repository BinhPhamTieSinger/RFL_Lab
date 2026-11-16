import { chromium, expect } from "@playwright/test";
import fs from "node:fs";

const base = process.env.RFL_TEST_URL ?? "http://127.0.0.1:3000";
const browser = await chromium.launch({
  executablePath:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
});
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
const key =
  process.env.RFL_TEST_CPU === "1"
    ? fs.readFileSync("../data/online-cpu/access-key", "utf8").trim()
    : "";
if (key)
  await page.addInitScript(
    (value) => sessionStorage.setItem("rfl-cpu-key", value),
    key,
  );
const history = await page.request
  .get(base + "/api/history")
  .then((r) => r.json());
const target = history.at(-1);
await page.goto(`${base}/?experiment=${target.id}&section=config`);
const entry = page.locator(`#${target.id}`);
await expect(entry).toHaveAttribute("open", "");
await expect(
  entry
    .locator("details")
    .filter({ has: page.locator("summary", { hasText: /^Configuration$/ }) }),
).toHaveAttribute("open", "");
await expect(
  entry.getByRole("button", { name: "Copy configuration link", exact: true }),
).toBeVisible();
await expect(entry.locator(".recharts-wrapper").first()).toBeVisible({
  timeout: 30000,
});
await page.getByRole("button", { name: "Live Workbench", exact: true }).click();
await page
  .getByRole("button", { name: "Load captured visualization", exact: true })
  .click();
await page.locator(".round-walkthrough").waitFor();
await page.getByLabel("num clients", { exact: true }).fill("4");
await expect(page.locator(".react-flow__node")).toHaveCount(0);
await expect(page.getByLabel("num clients", { exact: true })).toHaveValue("4");
await page.getByLabel("Method", { exact: true }).selectOption("fedavg");
await expect(page.getByLabel("num clients", { exact: true })).toHaveValue("6");
await page.getByLabel("Method", { exact: true }).selectOption("fedhep_plus");
await expect(page.getByLabel("num clients", { exact: true })).toHaveValue("6");
let runId;
if (key) {
  await page.getByLabel("num clients", { exact: true }).fill("4");
  await page.getByLabel("train subset", { exact: true }).fill("120");
  await page.getByLabel("test subset", { exact: true }).fill("60");
  await page.getByLabel("alpha", { exact: true }).fill("0.05");
  await page
    .locator(".configuration summary")
    .filter({ hasText: /^fedhep$/ })
    .click();
  await page.getByLabel("num groups", { exact: true }).fill("2");
  await page
    .locator(".configuration summary")
    .filter({ hasText: /^attack$/ })
    .click();
  await page.getByLabel("kind", { exact: true }).selectOption("sign_flip");
  await page.getByLabel("count", { exact: true }).fill("1");
  const started = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/experiments") && r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Run visualization", exact: true })
    .click();
  const response = await started;
  if (!response.ok()) throw Error(await response.text());
  runId = (await response.json()).id;
  await expect(page.locator(".react-flow__node.client")).toHaveCount(4, {
    timeout: 90000,
  });
  const held = await page
    .getByLabel("Execution event", { exact: true })
    .inputValue();
  await expect(page.locator(".run-status b")).toHaveText("completed", {
    timeout: 120000,
  });
  await expect(page.getByLabel("Execution event", { exact: true })).toHaveValue(
    held,
  );
  const batch = await page.request
    .get(`${base}/api/experiments/${runId}/event-batch`, {
      headers: { "X-RFL-Key": key },
    })
    .then((r) => r.json());
  const initialized = batch.events.find((e) => e.type === "initialized");
  if (initialized.clients.length !== 4 || initialized.attackers.length !== 1)
    throw Error("Runtime configuration differs from sidebar");
  const detail = await page.request
    .get(`${base}/api/experiments/${runId}`, { headers: { "X-RFL-Key": key } })
    .then((r) => r.json());
  if (
    detail.config.overrides["data.alpha"] !== 0.05 ||
    detail.config.overrides["fedhep.num_groups"] !== 2 ||
    detail.config.overrides["run.device"] !== "cpu"
  )
    throw Error("Saved run configuration mismatch");
  await page
    .getByRole("button", { name: "Next operation", exact: true })
    .click();
  if (
    (await page.getByLabel("Execution event", { exact: true }).inputValue()) ===
    held
  )
    throw Error("Next operation did not advance");
  const currentHistory = await page.request
    .get(base + "/api/history")
    .then((r) => r.json());
  if (currentHistory.some((r) => r.id === runId))
    throw Error("Visualization run entered curated history");
}
await browser.close();
if (errors.length) throw Error(errors.join("\n"));
console.log(
  JSON.stringify({
    base,
    sharedExperiment: target.id,
    configurationEditing: true,
    manualAdvance: true,
    runId,
    errors,
  }),
);
