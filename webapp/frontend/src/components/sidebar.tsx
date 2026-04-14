"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: "01" },
  { href: "/pipeline", label: "Data Pipeline", icon: "02" },
  { href: "/mislabels", label: "Mislabels", icon: "03" },
  { href: "/training", label: "Training", icon: "04" },
  { href: "/eval", label: "Evaluation", icon: "05" },
  { href: "/dashboard", label: "Dashboard", icon: "06" },
  { href: "/explorer", label: "Explorer", icon: "07" },
  { href: "/playground", label: "Playground", icon: "08" },
  { href: "/human-eval", label: "Human Eval", icon: "09" },
];

export function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  // Close drawer on route change
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  // Lock body scroll when drawer is open
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <>
      {/* Mobile top bar */}
      <header className="md:hidden fixed top-0 left-0 right-0 h-14 bg-white border-b border-border-light z-40 flex items-center justify-between px-4">
        <Link href="/" className="flex items-baseline gap-2">
          <span
            className="text-[20px] leading-none text-foreground"
            style={{ fontFamily: "var(--font-dm-serif)" }}
          >
            LLMao
          </span>
          <span
            className="text-[9px] tracking-[0.16px] uppercase text-muted"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Sarcasm Transfer
          </span>
        </Link>
        <button
          aria-label="Toggle navigation"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="w-10 h-10 flex flex-col items-center justify-center gap-[5px] -mr-2"
        >
          <span
            className={`block w-5 h-px bg-foreground transition-transform duration-200 ${
              open ? "translate-y-[6px] rotate-45" : ""
            }`}
          />
          <span
            className={`block w-5 h-px bg-foreground transition-opacity duration-200 ${
              open ? "opacity-0" : ""
            }`}
          />
          <span
            className={`block w-5 h-px bg-foreground transition-transform duration-200 ${
              open ? "-translate-y-[6px] -rotate-45" : ""
            }`}
          />
        </button>
      </header>

      {/* Backdrop (mobile drawer only) */}
      {open && (
        <div
          className="md:hidden fixed inset-0 bg-black/30 z-40"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar (drawer on mobile, fixed on desktop) */}
      <nav
        className={`fixed top-0 bottom-0 w-[260px] md:w-[240px] bg-white border-r border-border-light flex flex-col z-50 transition-transform duration-300 ease-out ${
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {/* Logo (desktop only — mobile uses the top bar) */}
        <div className="hidden md:block px-8 pt-8 pb-6">
          <Link href="/" className="block group">
            <h1
              className="text-[28px] leading-none tracking-[-0.56px] text-foreground font-display"
              style={{ fontFamily: "var(--font-dm-serif)" }}
            >
              LLMao
            </h1>
            <span
              className="text-[11px] tracking-[0.28px] uppercase text-muted mt-1.5 block font-code"
              style={{ fontFamily: "var(--font-jetbrains-mono)" }}
            >
              Sarcasm Transfer
            </span>
          </Link>
        </div>

        {/* Top spacing on mobile (below the header) */}
        <div className="md:hidden h-14" />

        {/* Nav links */}
        <div className="flex-1 px-4 pt-4 md:pt-0 overflow-y-auto">
          <div
            className="text-[11px] tracking-[0.28px] uppercase text-muted px-4 mb-3 font-code"
            style={{ fontFamily: "var(--font-jetbrains-mono)" }}
          >
            Navigation
          </div>
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-2.5 rounded-xl text-[15px] transition-all duration-200 group mb-0.5 ${
                  isActive
                    ? "bg-foreground text-white"
                    : "text-foreground-secondary hover:text-accent-blue"
                }`}
              >
                <span
                  className={`text-[11px] tracking-[0.16px] font-code ${
                    isActive
                      ? "text-white/50"
                      : "text-muted group-hover:text-accent-blue/50"
                  }`}
                  style={{ fontFamily: "var(--font-jetbrains-mono)" }}
                >
                  {item.icon}
                </span>
                {item.label}
              </Link>
            );
          })}
        </div>

        {/* Footer */}
        <div className="px-8 py-6 border-t border-border-light">
          <p className="text-[12px] text-muted leading-relaxed">
            CS4248 Team 14
          </p>
          <p className="text-[11px] text-muted/60 mt-0.5">NUS AY2025/26 S2</p>
        </div>
      </nav>
    </>
  );
}
