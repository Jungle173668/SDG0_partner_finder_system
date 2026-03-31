import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SDG: Zero Partner Finder",
  description: "AI-powered sustainability business partner matching",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen">
        {/* Brand top bar */}
        <header className="bg-navy-light text-white px-6 py-3 flex items-center gap-3 shadow-md">
          <div className="bg-white rounded-md px-2 py-1">
            <img
              src="/static/SDG0logo.png"
              alt="SDG: Zero"
              className="h-6 object-contain"
            />
          </div>
          <span className="text-lg font-bold tracking-wide">Partner Finder</span>
          <span className="ml-auto text-xs text-white/50">AI-powered · SDG: Zero</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
