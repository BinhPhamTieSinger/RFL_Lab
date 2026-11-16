import { chromium, expect } from "@playwright/test";
import fs from "node:fs";
import dns from "node:dns";
dns.setDefaultResultOrder("ipv4first");

const base = process.env.RFL_TEST_URL ?? "http://127.0.0.1:3000";
const hostname = new URL(base).hostname;
const address = process.env.RFL_TEST_IP
  ? { address: process.env.RFL_TEST_IP }
  : await dns.promises.lookup(hostname, { family: 4 });
const browser = await chromium.launch({
  args: [`--host-resolver-rules=MAP ${hostname} ${address.address}`],
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
const endpoint = key
  ? JSON.parse(fs.readFileSync("../data/online-cpu/connection.json", "utf8"))
      .endpoint
  : "";
if (key)
  await page.addInitScript(
    ({ key, endpoint }) => {
      sessionStorage.setItem("rfl-cpu-key", key);
      sessionStorage.setItem("rfl-cpu-url", endpoint);
    },
    { key, endpoint },
  );
async function read(path, authenticated = false) {
  return page.evaluate(
    async ({ path, key, endpoint }) => {
      const response = await fetch(path, {
        headers: key ? { "X-RFL-Key": key, "X-RFL-Endpoint": endpoint } : {},
      });
      if (!response.ok) throw Error(`Request failed ${response.status}`);
      return response.json();
    },
    { path, key: authenticated ? key : "", endpoint },
  );
}
await page.goto(base);
const history = await read("/api/history");
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
if (key || process.env.RFL_TEST_LOCAL === "1") {
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
  await page
    .locator(".configuration label")
    .filter({ hasText: /^kind/ })
    .locator("select")
    .selectOption("sign_flip");
  await page.getByLabel("count", { exact: true }).fill("1");
  const started = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/experiments") && r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Run visualization", exact: true })
    .click();
  const response = await started;
  const submitted = response.request().postDataJSON();
  if (
    submitted.overrides["data.num_clients"] !== 4 ||
    submitted.overrides["data.train_subset"] !== 120 ||
    submitted.overrides["data.alpha"] !== 0.05
  )
    throw Error("A schema response overwrote the edited configuration");
  if (!response.ok()) throw Error(await response.text());
  runId = (await response.json()).id;
  await expect(page.locator('.react-flow__node[data-id^="c"]')).toHaveCount(4, {
    timeout: 90000,
  });
  await expect(page.locator('.react-flow__node[data-id^="p"]')).toHaveCount(2);
  const held = await page
    .getByLabel("Execution event", { exact: true })
    .inputValue();
  await expect(page.locator(".run-status b")).toHaveText("completed", {
    timeout: 120000,
  });
  await expect(page.getByLabel("Execution event", { exact: true })).toHaveValue(
    held,
  );
  const batch = await read(`/api/experiments/${runId}/event-batch`, true);
  const initialized = batch.events.find((e) => e.type === "initialized");
  if (
    initialized.clients.length !== 4 ||
    initialized.attackers.length !== 1 ||
    initialized.clients.reduce((total, c) => total + c.train, 0) !== 120
  )
    throw Error("Runtime configuration differs from sidebar");
  const detail = await read(`/api/experiments/${runId}`, true);
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
  const currentHistory = await read("/api/history");
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
