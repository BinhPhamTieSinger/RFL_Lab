import { chromium } from "@playwright/test";
import fs from "node:fs";
import dns from "node:dns";
dns.setDefaultResultOrder("ipv4first");
const base = process.env.RFL_TEST_URL ?? "http://127.0.0.1:3000";
const hostname = new URL(base).hostname;
const address = await dns.promises.lookup(hostname, { family: 4 });
const browser = await chromium.launch({
  args: [`--host-resolver-rules=MAP ${hostname} ${address.address}`],
  executablePath:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
});
const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
await page.goto(base);
await page.getByRole("button", { name: "Documentation", exact: true }).click();
await page.locator(".markdown-document h1").waitFor();
const docOptions = await page.locator(".documentation select option").count();
await page
  .locator(".documentation select")
  .selectOption("docs/EXPERIMENT_LOG.md");
await page.locator(".markdown-document table").first().waitFor();
await page.screenshot({
  path: "../docs/documentation-rendered.png",
  fullPage: false,
});
await page
  .getByRole("button", { name: "Experiment history", exact: true })
  .click();
await page.locator(".experiment-entry").first().waitFor();
const history = await page.request
  .get(base + "/api/history")
  .then((r) => r.json());
if (
  history.some(
    (r) => r.recorded || r.purpose === "visualization" || !r.archived,
  )
)
  throw Error("Visualization leaked into curated history");
await page.getByRole("button", { name: "Live Workbench", exact: true }).click();
await page
  .getByLabel("Visualization capture", { exact: true })
  .selectOption("fedhep_plus-sign_flip");
await page
  .getByRole("button", { name: /^Load (captured )?visualization$/ })
  .first()
  .click();
await page.locator(".round-walkthrough").waitFor();
const attack = await page
  .locator("label")
  .filter({ hasText: /^kind/ })
  .locator("select")
  .inputValue();
if (attack !== "sign_flip")
  throw Error("Sidebar attack does not match capture");
await page
  .getByRole("button", { name: /r1 · aclm applied/ })
  .first()
  .click();
await page.locator(".class-mask").waitFor();
await page.locator(".execution-animation .katex").first().waitFor();
await page
  .locator(".execution-animation")
  .screenshot({ path: "../docs/aclm-inspection.png" });
await page
  .getByRole("button", { name: /r1 · sccf computed/ })
  .first()
  .click();
await page
  .locator(".execution-animation")
  .screenshot({ path: "../docs/sccf-inspection.png" });
await page
  .getByRole("button", { name: /r1 · aggregation started/ })
  .first()
  .click();
await page.getByLabel("Follow camera").uncheck();
await page.getByRole("button", { name: "Play animation", exact: true }).click();
const heldCursor = await page
  .getByLabel("Execution event", { exact: true })
  .inputValue();
await page.waitForTimeout(1800);
if (
  (await page.getByLabel("Execution event", { exact: true }).inputValue()) !==
  heldCursor
)
  throw Error(
    "Inspection advanced without the researcher accepting the next step",
  );
await page
  .locator(".weight-packet animateMotion")
  .first()
  .waitFor({ state: "attached" });
const packet = page.locator(".weight-packet").first();
const before = await packet.boundingBox();
await page.waitForTimeout(250);
const after = await packet.boundingBox();
if (before && after && before.x === after.x && before.y === after.y)
  throw Error("Weight packet did not move");
await page
  .getByRole("button", { name: "Pause animation", exact: true })
  .click();
await page
  .locator(".center")
  .screenshot({ path: "../docs/weight-transfer.png" });
await page
  .getByLabel("Visualization capture", { exact: true })
  .selectOption("fedhep_plus-clean");
await page
  .getByRole("button", { name: /^(Reload|Load) (captured )?visualization$/ })
  .first()
  .click();
await page.getByText("Attack disabled", { exact: false }).waitFor();
if (await page.locator(".react-flow__node.malicious").count())
  throw Error("Clean capture contains malicious client");
await page
  .getByLabel("Visualization capture", { exact: true })
  .selectOption("fedhep_defended-sign_flip");
await page
  .getByRole("button", { name: /^(Reload|Load) (captured )?visualization$/ })
  .first()
  .click();
await page
  .getByRole("button", { name: /r1 · temporal trust updated/ })
  .first()
  .click();
await page
  .locator(".execution-animation")
  .screenshot({ path: "../docs/ttt-inspection.png" });
await page.getByRole("button", { name: "Toggle theme" }).click();
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForFunction(
  () => document.documentElement.scrollWidth <= innerWidth,
  undefined,
  { timeout: 5000 },
);
await page.screenshot({ path: "../docs/playback-mobile.png", fullPage: true });
const result = {
  base,
  errors,
  docOptions,
  historyEntries: history.length,
  attackStateMatched: true,
  cleanHasNoAttackers: true,
  animatedPacketMoved: true,
  mobileOverflow: false,
};
fs.writeFileSync(
  "../docs/playback-verification.json",
  JSON.stringify(result, null, 2),
);
await browser.close();
if (errors.length) process.exit(1);
