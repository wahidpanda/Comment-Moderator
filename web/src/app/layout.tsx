import type { Metadata } from "next";
import { SiteHeader } from "@/components/SiteHeader";
import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "Comment Moderator",
  description: "Hybrid YouTube comment analysis",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen">
        <ThemeProvider>
          <SiteHeader />
          <main className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}