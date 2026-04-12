"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: "01" },
  { href: "/pipeline", label: "Data Pipeline", icon: "02" },
  { href: "/mislabels", label: "Mislabels", icon: "03" },
  { href: "/dashboard", label: "Dashboard", icon: "04" },
  { href: "/explorer", label: "Explorer", icon: "05" },
  { href: "/playground", label: "Playground", icon: "06" },
  { href: "/human-eval", label: "Human Eval", icon: "07" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <nav className="fixed left-0 top-0 bottom-0 w-[240px] bg-white border-r border-border-light flex flex-col z-50">
      {/* Logo */}
      <div className="px-8 pt-8 pb-6">
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

      {/* Nav links */}
      <div className="flex-1 px-4">
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
                  isActive ? "text-white/50" : "text-muted group-hover:text-accent-blue/50"
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
  );
}
