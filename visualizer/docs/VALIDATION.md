# Validation record

Verified on the local workstation on 2026-11-15.

- 30 Python tests passed. All ten methods produce identical scientific result payloads with and without the observation adapter in two-round tests. Names and resource timing are excluded from the equality comparison.
- Tests execute sign flipping in every method, six additional attack kinds in FedHEP+, and all six robust aggregation rules. FLTrust's reference training is identified separately from client training.
- API integration verifies configuration rejection, a real worker run, serialized queue submission and queued cancellation, persisted result retrieval, event replay, cancellation and foreign-origin rejection.
- Production Next.js compilation and TypeScript validation passed with webpack. The sandbox Turbopack build stalled and was stopped. The webpack build completed successfully.
- Browser tests ran real CIFAR-100 through the UI. Two one-round 600-image clean runs and one two-round 300-image sign-flip run completed on CPU. These smoke runs check integration and have low accuracy because of their small training budgets.
- The final two-round run recorded 853 events. Client 0 was selected as the attacker by the original seeded selector and isolated by the original screening implementation. Its training and evaluation continued according to FedHEP behavior.
- Browser checks verified both themes, a 390-pixel viewport, mathematical rendering and history replay. No page JavaScript errors or horizontal document overflow were recorded. The native timeline slider was corrected to neutral gray after screenshot inspection.
- Both desktop screenshots and the mobile screenshot were inspected. The process terminal displays actual runner output. Topology controls provide pan and zoom, and the canvas refits on viewport resizing.
- draw.io structural validation reported zero errors, warnings, overlaps and crossings. The diagram remains editable XML.
- Python lint passed. Research source has no Git diff from this implementation.

`browser-check.json`, `smoke-results.json`, `python-versions.json`, and `skill-manifest.json` retain the evidence and exact installed skill file hashes. The frontend package lock records dependency versions. Full worker logs and result artifacts remain in the ignored local `data` directory.

## Validation limits

GPU execution was not tested. Operation duration is host-call time, with the original runner's synchronized round timings available in results. Full activation tensors, per-class test accuracy, test loss and checkpoint resume are unavailable in this interface. Operation capture records the first invocation per leaf module per client training call and all optimizer-step counts. The Architecture view displays captured call order and the instantiated module tree. Residual tensor routing is available in the source code.

Large-scale experiments and multi-user deployment were not run. Browser testing used installed Google Chrome on macOS. Netlify hosts the archive and recorded replay. New training remains local at the owner’s request. The current service binds to loopback.

The fresh application follows the owner's project brief. The old visualizer folder was already absent when the requested archive rename was attempted. Its tracked deletions were left untouched. No repository reset, history rewrite, or research-output deletion occurred.

## Round walkthrough and archive update

The latest update adds a real one-round CIFAR-100 CPU trace with 300 training images, 150 test images, six clients and one sign-flip attacker. Its captured transcript is included for hosted replay. The hosted interface explicitly keeps new training local. The recorded run is labeled in history.

Client thumbnails are decoded from the actual normalized client tensors. A regression test compares them pixel-for-pixel with source images and checks that preview capture preserves tensors and RNG state. R_skew calls the original label_stats implementation. Initial previews show the original partition before attack transforms.

Queue tests replace the former conflict rejection test. Workers remain serialized through the process lock. Queued cancellation and active cancellation are covered. Archive indexing tests cover config-only entries and subsequently completed results.

Each experiment owns its configuration, evaluation table, accuracy chart, MAD chart when captured, and detection curves. File-only historical entries use modification dates, explicitly labeled. Missing final metrics remain incomplete. Historical files lack granular training events. An included recorded run supports exact event replay.

Green and red identify honest and malicious clients following the owner's latest request. Sample photographs retain their actual colors. Remaining interface controls and plots use neutral colors.

The expanded Python suite passed 32 tests. Production compilation and TypeScript validation passed after adding archived history, client images and queued execution.

## Documentation and animated execution

The expanded passive-observation tests validate ACLM's masked values and unchanged active logits, TTT ledger arithmetic, and the use of screening evidence in FedHEP+. All ten methods still match their uninstrumented scientific results. Eleven real CIFAR-100 CPU captures cover all ten methods plus clean FedHEP+. Each capture uses one round, six clients and 300 training images.

The production build and TypeScript checks passed with the documentation reader, separate capture endpoints, manual history manifest and event-driven weight-transfer rendering. Documentation is exported from the current repository Markdown during builds, including its figures. GFM tables, lists and KaTeX are rendered in the reading view.

Browser verification passed on the local workbench with zero page errors, 14 rendered Markdown documents, 783 curated history entries, consistent attack configuration, a clean capture with zero malicious nodes, measured SVG packet motion and no mobile document overflow. The real local one-round training browser test also passed after the playback changes.
