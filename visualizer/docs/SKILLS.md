# Installed design guidance

All installations were performed with the repositories' documented CLIs. Existing installations were retained. The host is Codex on macOS with Node 26 and Python 3.11. Skills are available as project files and can be discovered by Codex on the next turn. Their instructions were read directly during implementation.

| Package | Verified source/version | Installed location | Applied guidance |
| --- | --- | --- | --- |
| Impeccable | Skill 4.9.1, engine 0.1.24. Source revision 2b86d9c07b97d4bed7de23583e1bd1e58dd2baae | `RFL_Lab/.agents/skills/impeccable` | Product context, operate-mode layout, monochrome craft review, focus and theme states |
| UI UX Pro Max | Source inspected at 0e8d408204fdc66b8189960dd28cc20d71db188e. Installed from current ui-ux-pro-max-cli templates | `visualizer/.agents/skills/ui-ux-pro-max` | Ran design-system search. Applied accessibility, typography and responsive guidance. Owner's explicit palette overrides the generated chromatic palette |
| drawio-skill | Source inspected at 6550495129d9a42eff7e71434d15bfc9846b6202. Installed hash in skills-lock.json | `visualizer/.agents/skills/drawio-skill` | Read XML authoring instructions, authored editable architecture diagram, ran structural validation |
| Vercel React best practices | 1.0.0, source 1a2c79d4e6c4a2283dc2c614b6538cbe1ed5ca62 | `visualizer/.agents/skills/vercel-react-best-practices` | Parallel initial fetches, lazy terminal import, buffered event updates, memoized topology |
| Vercel web design guidelines | 1.0.0, installed hash in skills-lock.json | `visualizer/.agents/skills/web-design-guidelines` | Instructions read. Fresh remote guideline review remains unverified |

Commands used were `npx impeccable install --providers=codex --scope=project`, `npx ui-ux-pro-max-cli init --ai codex`, and `npx skills add` with explicit relevant skill selections. React guidance used the downloaded source checkout after the initial selector matched only web-design-guidelines.

The UI UX installer includes supplementary design, brand, slides and banner assets in its bundle. These were installed by that upstream command and were not invoked. Impeccable installed its hook configuration under `RFL_Lab/.codex`. Hook execution requires the owner's Codex hook approval and is not claimed active. The design instructions were applied directly. Impeccable's interview and delegated-review flow were replaced by the owner's supplied brief and explicit sole-agent authorization.

Draw.io desktop export was not installed. The editable XML diagram was validated with zero errors, warnings, overlaps or crossings by the skill's validator. This is structural validation, and does not claim desktop-rendered verification.
