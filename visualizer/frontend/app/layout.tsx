import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@xyflow/react/dist/style.css";
import "katex/dist/katex.min.css";
import "@xterm/xterm/css/xterm.css";
import "./style.css";
export const metadata = {
  title: "RFL Lab · Research workbench",
  description: "Observe real federated learning execution",
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
