# Online CPU bridge

The Netlify workbench can submit real configurations to the owner’s computer while the bridge is running. FastAPI binds to loopback port 8001. An authenticated HTTPS tunnel carries bounded event polling requests. The bridge forces CPU, uses two training threads and shares the training lock with the local workbench. Existing research algorithms and manually curated history keep their existing behavior.

## Start and stop

From RFL_Lab, run the following command with the existing research environment.

```sh
../.venv/bin/python visualizer/scripts/online_cpu.py
```

The script starts FastAPI and cloudflared, then opens a browser tab with the tunnel address and private CPU key in its URL fragment. The page removes the fragment and remembers the connection in this browser. Re-running the command while the bridge is active opens another connected tab. The Netlify proxy accepts HTTPS trycloudflare.com endpoints and rejects redirects. Its requests carry the owner’s access key. The Forget CPU connection button clears the remembered address and access key. Tabs in the same browser profile use the remembered CPU connection. Other browsers need the private connection link to run training. The key is in `visualizer/data/online-cpu/access-key` with owner-only file permissions.

Keep the process running while using the site. Ctrl-C stops the bridge and its API. Computer sleep, shutdown or loss of network makes custom execution unavailable. Restart the command to reconnect after reboot. Every fresh quick tunnel has a new hostname, and the newly opened browser tab uses that address. There is no login-item installation. `--no-open` suppresses the browser launch.

The installed CLI is cloudflared 2026.11.0 for macOS arm64, kept in ignored `visualizer/data/bin/cloudflared`. The official release archive SHA-256 is `8c2f90277ef06a6080bb40b964d33f09f1188102f8798f2e344174b6cf41accf`. Installation was checked against the release asset digest. On a new checkout, install the matching official platform binary at that path. The bridge requires the existing Python research dependencies. Netlify’s environment-management API rejected updates during setup, so the bridge supplies its current endpoint through the private browser connection. Site publication continues through the existing GitHub deployment.

The account-free quick tunnel has a temporary address and no uptime guarantee. Its supported transport excludes SSE, so the workbench fetches ordered batches with a persisted sequence cursor. A failed request retries from the same cursor. Training runs in a separate process and continues between HTTP requests. The current network uses IPv4 HTTP/2 transport. The launcher retries temporary URL-creation failures and waits for a registered tunnel connection. [Cloudflare quick tunnel documentation](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)

## Verification and logs

`/api/capabilities` reports CPU connectivity every 15 seconds. Requests to computation endpoints require the private key. The key travels in a header through the same-origin Netlify proxy and never enters a shared history URL or the frontend bundle. The local API authenticates all paths except its health check. CPU runs, logs and credentials stay under ignored `visualizer/data/online-cpu/`.

For a real one-round browser check against production, run the following command from `visualizer/frontend`. It uses 120 real CIFAR-100 training examples, 60 test examples, four clients, two parents, alpha 0.05 and one sign-flip attacker.

```sh
RFL_TEST_URL=https://rfl-lab-visualizer.netlify.app RFL_TEST_CPU=1 node config-check.mjs
```

The check verifies the requested configuration, actual client identities, completed CPU execution, held inspection cursor, manual Next, shared historical configuration and exclusion from curated history. `node playback-check.mjs` additionally checks Markdown, formulas, animated transfer packets, autoplay, pause, themes and the mobile layout.

## Local accelerators

CPU remains the default. To expose accelerators through the hosted sidebar, stop the running bridge with Ctrl-C and restart it with the following command.

```sh
../.venv/bin/python visualizer/scripts/online_cpu.py --allow-accelerators
```

The connected Python environment reports its available devices. Select one under Run → device. CPU runs on the processor, MPS uses a supported Mac GPU, CUDA uses a GPU supported by the installed CUDA/ROCm PyTorch build, and XPU uses a supported Intel GPU. Auto follows the existing research order of CUDA, XPU, MPS, then CPU. Unavailable devices are rejected before training. Selecting an accelerator never changes the default for a new configuration.

Intel Iris availability depends on the exact hardware, OS, drivers and PyTorch build. XPU is offered only when `torch.xpu.is_available()` succeeds. PyTorch's validated Intel GPU list should be checked for the target machine. The current bridge implementation uses POSIX process groups and file locks, so its supported host environments are macOS and Linux. Windows requires a compatible Linux/WSL environment and corresponding GPU support. [PyTorch Intel GPU setup](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)

The local-only workbench launched with `visualizer/scripts/dev.sh` also discovers installed devices. Browser rendering uses the viewing computer's hardware. Training executes on the machine running the connected Python API.
