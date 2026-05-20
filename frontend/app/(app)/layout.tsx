"use client";

import { Navbar } from "@/components/layout/navbar";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-screen flex-col">
      <Navbar />
      <main className="flex-1">{children}</main>
      <footer className="border-t py-4 text-center text-xs text-muted-foreground">
        <p>
          ⚠️ AI-generated content — not a substitute for professional medical
          advice. Always consult a qualified healthcare provider.
        </p>
      </footer>
    </div>
  );
}
