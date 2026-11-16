"use client";
import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

type Document = { path: string; title: string; content: string };
export default function Documentation() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selected, setSelected] = useState("docs/ARCHITECTURE.md");
  const [error, setError] = useState("");
  useEffect(() => {
    setSelected(
      new URL(window.location.href).searchParams.get("document") ??
        "docs/ARCHITECTURE.md",
    );
    fetch("/research/documentation.json")
      .then(async (response) => {
        if (!response.ok) throw Error("Documentation could not be loaded");
        setDocuments(await response.json());
      })
      .catch((e) => setError(e.message));
  }, []);
  const document =
    documents.find((item) => item.path === selected) ?? documents[0];
  function relativePath(value: string) {
    return new URL(
      value,
      `https://research.local/${document?.path ?? "README.md"}`,
    ).pathname.slice(1);
  }
  return (
    <div className="documentation">
      <h1>Documentation</h1>
      <label>
        Research document
        <select
          value={document?.path ?? selected}
          onChange={(e) => {
            setSelected(e.target.value);
            const url = new URL(window.location.href);
            url.searchParams.set("document", e.target.value);
            window.history.replaceState(null, "", url);
          }}
        >
          {documents.map((item) => (
            <option key={item.path} value={item.path}>
              {item.title}
            </option>
          ))}
        </select>
      </label>
      {error && <p role="alert">{error}</p>}
      {!document && !error && <p>Loading research Markdown…</p>}
      {document && (
        <article className="markdown-document">
          <p className="document-source">{document.path}</p>
          <Markdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              a: ({ href, children }) => {
                const match =
                  href &&
                  documents.find(
                    (item) => item.path === relativePath(href).split("#")[0],
                  );
                return match ? (
                  <a
                    href="#"
                    onClick={(e) => {
                      e.preventDefault();
                      setSelected(match.path);
                    }}
                  >
                    {children}
                  </a>
                ) : (
                  <a href={href}>{children}</a>
                );
              },
              img: ({ src, alt }) => (
                <img
                  src={
                    typeof src === "string" && !/^https?:|^data:/.test(src)
                      ? `/research/documents/${relativePath(src)}`
                      : src
                  }
                  alt={alt ?? "Research figure"}
                  loading="lazy"
                />
              ),
              table: ({ children }) => (
                <div className="table-scroll">
                  <table>{children}</table>
                </div>
              ),
            }}
          >
            {document.content}
          </Markdown>
        </article>
      )}
    </div>
  );
}
