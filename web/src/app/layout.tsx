import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import { SiteHeader } from "@/components/SiteHeader";
import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

const sans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Comment Moderator · Creator triage",
  description: "Hybrid YouTube comment analysis for creators",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${sans.variable}`} suppressHydrationWarning>
      <body className="min-h-screen font-[family-name:var(--font-sans)]">
        <div className="mesh-bg" aria-hidden />
        <ThemeProvider>
          <SiteHeader />
          <main className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 lg:py-10">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}