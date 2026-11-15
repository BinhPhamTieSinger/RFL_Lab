import {chromium} from '@playwright/test';
import fs from 'node:fs';
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const page=await browser.newPage({viewport:{width:1500,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:3000');
await page.getByRole('option',{name:'FedHEP+',exact:true}).waitFor({state:'attached'});
await page.getByLabel('train subset',{exact:true}).fill('300');
await page.getByLabel('test subset',{exact:true}).fill('150');
await page.getByLabel('rounds',{exact:true}).fill('2');
await page.locator('summary').filter({hasText:/^attack$/i}).click();
await page.locator('label').filter({hasText:/^kind/}).locator('select').selectOption('sign_flip');
await page.getByLabel('count',{exact:true}).fill('1');
await page.getByRole('button',{name:'Run visualization',exact:true}).click();
await page.getByText('completed',{exact:true}).first().waitFor({timeout:120000});
await page.locator('footer').scrollIntoViewIfNeeded();
await page.locator('header').scrollIntoViewIfNeeded();
const panels = await page.evaluate(() => {
  const workbench = document.querySelector('.workbench').getBoundingClientRect();
  return ['.timeline', '.metrics'].map(selector => {
    const panel = document.querySelector(selector).getBoundingClientRect();
    return {selector, width: panel.width, left: panel.left, expectedWidth: workbench.width, expectedLeft: workbench.left};
  });
});
for (const panel of panels) {
  if (Math.abs(panel.width - panel.expectedWidth) > 1 || Math.abs(panel.left - panel.expectedLeft) > 1) {
    throw new Error(`${panel.selector} must span the full workbench width`);
  }
}

await page.screenshot({path:'../docs/workbench-dark.png',fullPage:true});
await page.getByRole('button',{name:'Toggle theme'}).click();
await page.screenshot({path:'../docs/workbench-light.png',fullPage:true});
await page.setViewportSize({width:390,height:844});
await page.screenshot({path:'../docs/workbench-mobile.png',fullPage:true});
const accent=await page.locator('input[type=range]').evaluate(el=>getComputedStyle(el).accentColor);
const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
await page.getByRole('button',{name:'Mathematical inspection',exact:true}).click();
await page.locator('.katex').first().waitFor();
await page.getByRole('button',{name:'Experiment history',exact:true}).click();
await page.locator('.experiment-entry > summary').first().click();
await page.getByRole('heading',{name:'Saved evaluation table'}).first().waitFor();
await page.getByRole('button',{name:'Live Workbench',exact:true}).click();
await page.getByRole('button',{name:'Load captured visualization',exact:true}).click();
await page.locator('.react-flow__node').first().waitFor();
fs.writeFileSync('../docs/browser-check.json',JSON.stringify({errors,mobileOverflow:overflow,rangeAccent:accent},null,2));
await browser.close();
if(errors.length||overflow)process.exit(1);
