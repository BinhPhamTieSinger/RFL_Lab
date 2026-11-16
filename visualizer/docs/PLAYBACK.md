# Execution playback

The Live Workbench visualizes captured execution events and local Python training. Visualization captures are separate from manually curated research history.

1. Choose a visualization capture by model and attack setting.
2. Load it. The configuration rail and graph are populated together from that capture.
3. Play animation advances automatically through recorded operations. Pause holds the current operation. Previous operation and Next operation each move one step and pause. At 1×, individual forward/backward operations advance after 90 ms, other events after 600 ms, and weight packets traverse a link in 350 ms. Every speed setting scales these intervals. Each operation retains its original captured timestamp and duration.
4. Select a timeline event to inspect a particular calculation. Phase buttons navigate to the corresponding part of the round.
5. Enable Follow camera to focus on the active client, transfer endpoints or processing server.

Weight packets move along client-to-Global edges when uploads become available, and along client-to-parent edges when the parent's aggregation consumes those updates. Source events contain actual client identities, parent membership, payload sizes where captured, and aggregation inputs. The research implements communication through local tensors. There is no measured network transfer latency.

ACLM shows the actual first sample's logits before and after masking, plus the full active-class mask. The parent and local losses mask absent classes to -1e9. The root loss uses all classes. Only the first batch is captured at this granularity.

SCCF shows bounded update norms, computed cosines and returned trust weights. Zero weights indicate SCCF rejection. Nonzero weights indicate a contribution to aggregation. Client attack-role labels come from the configured ground truth. Isolation states come from the detector's own events.

TTT shows the actual ledger update in Defended FedHEP. Scores start at 1, use momentum 0.85 toward min(1, n times trust), and isolate below 0.25. Updates occur at parent and Global tiers. FedHEP+ displays its screening evidence update instead.

The hosted site supports playback for all ten models, plus clean and attacked FedHEP+ captures. Custom runs execute through the local Python service. The source algorithms remain unchanged. Reduced-motion preferences retain static transfer markers and manual event navigation.

Sidebar settings are editable after a capture or completed run. Editing clears the previous trace. Run visualization executes the new configuration on the connected computer CPU. Training continues while the researcher inspects a result. Arriving events accumulate, and the round progress and terminal update independently. Play advances through available events. Paused inspection waits for Previous, Next, a phase button, a timeline selection or Inspect latest available. Playback controls and the collapsible execution trace stay at the top while scrolling. Hide settings collapses the configuration sidebar. The URL preserves the current tab, capture and inspection position on refresh.

Label-flip inspection shows captured original and poisoned training targets and full target histograms. The implemented mapping is y′ = C − 1 − y. The additional FedHEP+ label-flip capture comes from a real one-round CIFAR-100 CPU run. ACLM entries identify the parent and local heads separately.

Leaving Live Workbench or hiding the browser tab pauses animation. Returning keeps it paused until Play animation is pressed. Local running and queued visualization jobs are also paused through the authenticated API. Use Resume execution to continue them. On the supported macOS/Linux backend, process-group suspension preserves model state and the training resource lock. GPU allocations remain resident and already submitted accelerator work may finish. The hosted CPU bridge continues to enforce CPU execution. Restart an existing local API after updating to enable the pause/resume endpoints.
