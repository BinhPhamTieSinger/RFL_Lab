import { chromium, expect } from "@playwright/test";
import dns from "node:dns/promises";
const base = process.env.RFL_TEST_URL ?? "http://127.0.0.1:3000";
const host = new URL(base).hostname;
const ip =
  process.env.RFL_TEST_IP ?? (await dns.lookup(host, { family: 4 })).address;
const browser = await chromium.launch({
  executablePath:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  args: [`--host-resolver-rules=MAP ${host} ${ip}`],
});
const page = await browser.newPage({ viewport: { width: 1450, height: 950 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
try {
  await page.goto(base);
  const toolbar = await page
    .locator(".sticky-workbench-controls")
    .boundingBox();
  const settings = await page
    .getByRole("button", { name: "Hide settings", exact: true })
    .boundingBox();
  expect(settings.y + settings.height).toBeLessThanOrEqual(
    toolbar.y + toolbar.height,
  );

  await page
    .getByLabel("Visualization capture", { exact: true })
    .selectOption("fedhep_plus-label_flip");
  await page
    .getByRole("button", { name: "Load captured visualization", exact: true })
    .click();
  await page.locator(".round-walkthrough").waitFor();
  const center = await page.locator(".center").boundingBox();
  const canvas = await page.locator(".center .canvas").boundingBox();
  expect(
    Math.abs(center.y + center.height - canvas.y - canvas.height),
  ).toBeLessThan(2);
  await page.screenshot({ path: "/tmp/rfl-layout.png" });
  await page.locator(".sticky-trace > summary").click();
  await page
    .getByRole("button", { name: /r1 · attack executed/ })
    .first()
    .click();
  await expect(page.locator(".attack-transformation")).toContainText(
    "Poisoned target",
  );
  await expect(
    page.locator(".attack-transformation > table tbody tr"),
  ).toHaveCount(12);
  await page
    .getByRole("button", { name: /aclm applied · parent head/ })
    .first()
    .click();
  await expect(page.locator(".execution-animation")).toContainText(
    "personal local head",
  );
  const slider = page.getByLabel("Execution event", { exact: true });
  const initial = Number(await slider.inputValue());
  await page
    .getByRole("button", { name: "Next operation", exact: true })
    .click();
  await expect
    .poll(async () => Number(await slider.inputValue()))
    .toBeGreaterThan(initial);
  await page
    .getByRole("button", { name: "Previous operation", exact: true })
    .click();
  await expect(slider).toHaveValue(String(initial));
  await page.getByLabel("Animation speed", { exact: true }).selectOption("4");
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .click();
  await expect
    .poll(async () => Number(await slider.inputValue()))
    .toBeGreaterThan(initial);
  await page
    .getByRole("button", { name: "Pause animation", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Documentation", exact: true })
    .click();
  const hiddenStep = new URL(page.url()).searchParams.get("step");
  await page.waitForTimeout(700);
  expect(new URL(page.url()).searchParams.get("step")).toBe(hiddenStep);
  await page
    .getByRole("button", { name: "Live Workbench", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Play animation", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .click();
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(
    page.getByRole("button", { name: "Play animation", exact: true }),
  ).toBeVisible();
  await page.evaluate(() => {
    delete document.hidden;
    document.dispatchEvent(new Event("visibilitychange"));
  });
  for (const tab of [
    "Architecture",
    "Mathematical inspection",
    "Experiment history",
    "Documentation",
  ]) {
    await page.getByRole("button", { name: tab, exact: true }).click();
    await page
      .getByRole("button", { name: "Live Workbench", exact: true })
      .click();
    await expect
      .poll(
        () =>
          page.locator(".center .react-flow").evaluate((flow) => {
            const canvas = flow.getBoundingClientRect();
            return Array.from(flow.querySelectorAll(".react-flow__node")).some(
              (node) => {
                const box = node.getBoundingClientRect();
                return (
                  box.width > 0 &&
                  box.height > 0 &&
                  box.right > canvas.left &&
                  box.left < canvas.right &&
                  box.bottom > canvas.top &&
                  box.top < canvas.bottom
                );
              },
            );
          }),
        { timeout: 4000 },
      )
      .toBe(true);
  }
  const paused = await slider.inputValue();
  await page.waitForTimeout(700);
  await expect(slider).toHaveValue(paused);
  await page.evaluate(() => window.scrollTo(0, 1400));
  const bounds = await page
    .getByRole("button", { name: "Next operation", exact: true })
    .boundingBox();
  if (!bounds || bounds.y < 0 || bounds.y > 180)
    throw Error("Playback controls did not remain at top");
  await page
    .getByRole("button", { name: "Hide settings", exact: true })
    .click();
  await expect(page.locator(".configuration")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Show settings", exact: true })
    .click();
  await expect(page.locator(".configuration")).toBeVisible();
  await page.reload();
  await expect(slider).toHaveValue(paused);
  await page
    .getByRole("button", { name: "Experiment history", exact: true })
    .click();
  await expect(page).toHaveURL(/tab=Experiment\+history/);
  await page.locator(".experiment-entry input[type=checkbox]").nth(0).check();
  await page.locator(".experiment-entry input[type=checkbox]").nth(1).check();
  await page
    .getByRole("button", { name: "Show selected (2)", exact: true })
    .click();
  await expect(page.locator(".experiment-entry")).toHaveCount(2);
  expect(new URL(page.url()).searchParams.has("experiments")).toBe(false);
  await page.reload();
  await expect(page.locator(".experiment-entry")).toHaveCount(2);
  const legacyUrl = new URL(page.url());
  const savedSelection = await page.evaluate(() => {
    const ids = localStorage.getItem("rfl-selected-experiments");
    localStorage.removeItem("rfl-selected-experiments");
    return ids;
  });
  legacyUrl.searchParams.set("experiments", savedSelection);
  await page.goto(legacyUrl.toString());
  await expect(page.locator(".experiment-entry")).toHaveCount(2);
  expect(new URL(page.url()).searchParams.has("experiments")).toBe(false);
  expect(
    await page.evaluate(() => localStorage.getItem("rfl-selected-experiments")),
  ).toBe(savedSelection);
  await page
    .getByRole("button", { name: "Copy selected link", exact: true })
    .click();
  const shareUrl = await page
    .getByLabel("Selected experiments link", { exact: true })
    .inputValue();
  expect(shareUrl.length).toBeLessThan(legacyUrl.href.length);
  const professorPage = await browser.newPage();
  await professorPage.goto(shareUrl);
  await expect(professorPage.locator(".experiment-entry")).toHaveCount(2);
  await professorPage.close();
  let downloads = 0;
  page.on("request", (request) => {
    if (/\/research\/archive-.*\.json/.test(request.url())) downloads++;
  });
  await page.locator(".experiment-entry > summary").first().click();
  await page.locator(".experiment-content .recharts-wrapper").first().waitFor();
  const firstDownloads = downloads;
  await page
    .getByRole("button", { name: "Documentation", exact: true })
    .click();
  await expect(page).toHaveURL(/tab=Documentation/);
  await page
    .getByRole("button", { name: "Experiment history", exact: true })
    .click();
  await page.locator(".experiment-entry > summary").first().click();
  await page.locator(".experiment-content .recharts-wrapper").first().waitFor();
  if (downloads !== firstDownloads)
    throw Error("Result was downloaded again after changing tabs");
  if (errors.length) throw Error(errors.join("\n"));
  console.log(
    JSON.stringify({
      base,
      autoplay: true,
      previous: true,
      sticky: true,
      sidebar: true,
      labelFlip: true,
      urlRestoration: true,
      selectedHistory: true,
      cachedResults: true,
      errors,
    }),
  );
} finally {
  await browser.close();
}
