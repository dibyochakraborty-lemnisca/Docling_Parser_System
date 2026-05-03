import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "fermdocs — hypothesis stage",
  description: "Multi-agent fermentation-hypothesis debate viewer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <header className="border-b">
          <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
            <a href="/" className="text-base font-semibold">
              fermdocs
              <span className="ml-2 text-muted-foreground font-normal">
                hypothesis stage
              </span>
            </a>
            <nav className="text-sm text-muted-foreground">
              <a className="hover:text-foreground" href="/">
                runs
              </a>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
