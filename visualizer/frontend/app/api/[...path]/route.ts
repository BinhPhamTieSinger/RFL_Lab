import catalog from "../../../research/catalog.json";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
type Context = { params: Promise<{ path: string[] }> };
async function handle(request: Request, context: Context) {
  const { path } = await context.params;
  const route = path.join("/");
  if (request.method !== "GET") {
    const origin = request.headers.get("origin");
    if (origin && new URL(origin).host !== request.headers.get("host")) {
      return Response.json({ detail: "Untrusted origin" }, { status: 403 });
    }
  }

  const local = process.env.RFL_ARCHIVE_ONLY !== "1";
  if (route === "capabilities")
    return Response.json({
      training: local,
      mode: local
        ? "Local Python execution"
        : "Execution visualization and curated history",
    });
  let artifact: string | undefined;
  if (route === "history") artifact = "history.json";
  if (path[0] === "history" && /^archive-[a-f0-9]{24}$/.test(path[1] ?? ""))
    artifact = `${path[1]}.json`;
  if (route === "live/captures") artifact = "captures.json";
  if (
    path[0] === "live" &&
    path[1] === "captures" &&
    /^[a-z0-9_-]+$/.test(path[2] ?? "")
  )
    artifact = `captures/${path[2]}.json`;
  if (route === "documentation") artifact = "documentation.json";
  if (artifact && request.method === "GET") {
    const response = await fetch(
      new URL(`/research/${artifact}`, request.url),
      { cache: "no-store" },
    );
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  }
  if (local) {
    try {
      const upstream = await fetch(
        `${process.env.RFL_BACKEND_URL ?? "http://127.0.0.1:8000"}/${route}${new URL(request.url).search}`,
        {
          method: request.method,
          headers: {
            "Content-Type": "application/json",
            ...(request.headers.get("last-event-id")
              ? { "Last-Event-ID": request.headers.get("last-event-id")! }
              : {}),
          },
          body: request.method === "GET" ? undefined : await request.text(),
          signal: request.signal,
          cache: "no-store",
        },
      );
      return new Response(upstream.body, {
        status: upstream.status,
        headers: {
          "Content-Type":
            upstream.headers.get("content-type") ?? "application/json",
          "Cache-Control": "no-store",
          "X-Accel-Buffering": "no",
        },
      });
    } catch {
      return Response.json(
        {
          detail:
            "Local training API is unavailable. Start visualizer/scripts/dev.sh.",
        },
        { status: 503 },
      );
    }
  }
  if (request.method !== "GET")
    return Response.json(
      {
        detail:
          "Training runs locally. Start visualizer/scripts/dev.sh in the research checkout.",
      },
      { status: 409 },
    );
  if (route === "models") return Response.json(catalog.models);
  if (route === "formulas") return Response.json(catalog.formulas);
  if (path[0] === "source") {
    const source =
      catalog.sources[path.slice(1).join("/") as keyof typeof catalog.sources];
    return new Response(source ?? "Source unavailable", {
      status: source ? 200 : 404,
      headers: { "Content-Type": "text/plain" },
    });
  }
  if (path[0] === "models" && path[2] === "schema") {
    const schema = catalog.schemas[path[1] as keyof typeof catalog.schemas];
    return Response.json(schema ?? { detail: "Unknown model" }, {
      status: schema ? 200 : 404,
    });
  }
  if (
    route === "experiments" ||
    (path.length === 2 &&
      path[0] === "experiments" &&
      /^archive-[a-f0-9]{24}$/.test(path[1]))
  ) {
    const file = route === "experiments" ? "history" : path[1];
    const url = new URL(`/research/${file}.json`, request.url);
    const response = await fetch(url, { cache: "no-store" });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  }
  return Response.json({ detail: "Unknown resource" }, { status: 404 });
}
export const GET = handle;
export const POST = handle;
