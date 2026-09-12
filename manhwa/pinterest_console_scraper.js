/**
 * Pinterest Image URL Scraper
 * ──────────────────────────────────────────────────────────────────────────
 * HOW TO USE:
 *   1. Open Pinterest in your browser and navigate to a board or search page.
 *   2. Press F12 (Windows/Linux) or Cmd+Option+I (Mac) to open DevTools.
 *   3. Click the "Console" tab.
 *   4. Paste this entire script and press Enter.
 *   5. Wait — it scrolls automatically and shows progress in the console.
 *   6. When done, a JSON file downloads automatically.
 *   7. Give that file to the AI to import into your library.
 *
 * CONFIGURATION (edit these before running if needed):
 */
const CONFIG = {
  scrollDelay:    1800,   // ms to wait between scrolls (increase if internet is slow)
  maxScrolls:     120,    // max scroll steps before stopping (120 ≈ ~600 pins)
  staleAfter:     6,      // stop if no new images found for this many scrolls
  preferredSize:  "736x", // "originals" | "736x" | "474x" — prefer originals when available
};

/* ── Main ── */
(async function scrapePinterest() {
  console.log("%c📌 Pinterest Scraper starting…", "color:#e60023;font-weight:bold;font-size:14px");
  console.log(`Config: maxScrolls=${CONFIG.maxScrolls}, staleAfter=${CONFIG.staleAfter}`);

  const seen   = new Set();
  const urls   = [];
  let   stale  = 0;

  /** Upgrade a Pinterest CDN URL to the best available resolution. */
  function upgradeUrl(src) {
    if (!src || !src.includes("pinimg.com")) return null;
    // Strip query params
    src = src.split("?")[0];
    // Replace any size segment like /236x/, /474x/, /736x/, /60x60/ etc.
    // Try for originals first, fall back to 736x
    const withOrig = src.replace(/\/\d+x\d*\//, "/originals/").replace(/\/\d+x\//, "/originals/");
    return withOrig;
  }

  /** Collect all visible Pinterest image URLs on the page right now. */
  function collect() {
    const before = urls.length;

    // Primary: <img> tags
    document.querySelectorAll("img").forEach(img => {
      for (const attr of ["src", "data-src", "data-original"]) {
        const raw = img.getAttribute(attr);
        if (!raw) continue;
        const u = upgradeUrl(raw);
        if (u && !seen.has(u)) {
          seen.add(u);
          urls.push(u);
        }
      }
      // srcset — pick the largest entry
      const ss = img.getAttribute("srcset");
      if (ss) {
        const candidates = ss.split(",").map(s => s.trim().split(/\s+/)[0]);
        for (const c of candidates) {
          const u = upgradeUrl(c);
          if (u && !seen.has(u)) { seen.add(u); urls.push(u); }
        }
      }
    });

    // Secondary: background-image inline styles
    document.querySelectorAll("[style*='pinimg']").forEach(el => {
      const m = (el.getAttribute("style") || "").match(/url\(['"]?(https?:\/\/[^'")\s]+)['"]?\)/i);
      if (m) {
        const u = upgradeUrl(m[1]);
        if (u && !seen.has(u)) { seen.add(u); urls.push(u); }
      }
    });

    return urls.length - before;
  }

  // Initial pass before scrolling
  collect();

  // Auto-scroll loop
  for (let i = 0; i < CONFIG.maxScrolls; i++) {
    window.scrollTo(0, document.body.scrollHeight);
    await new Promise(r => setTimeout(r, CONFIG.scrollDelay));

    const added = collect();
    if (added === 0) {
      stale++;
      if (stale >= CONFIG.staleAfter) {
        console.log(`%c⏹ No new images for ${CONFIG.staleAfter} scrolls — stopping early.`, "color:#888");
        break;
      }
    } else {
      stale = 0;
    }

    if (i % 5 === 0 || added > 0) {
      console.log(`Scroll ${i + 1}/${CONFIG.maxScrolls} — ${urls.length} images (+${added})`);
    }
  }

  // Filter: keep only actual image files (not icons/avatars/spinners)
  const imageUrls = urls.filter(u =>
    /\.(jpg|jpeg|png|webp)/i.test(u) &&
    !/\/avatars?\//i.test(u) &&
    !/\/user-avatars\//i.test(u) &&
    !/favicon/i.test(u)
  );

  console.log(`%c✅ Done — ${imageUrls.length} unique images collected.`, "color:#22c55e;font-weight:bold");

  // Build output JSON
  const output = {
    source_url:   window.location.href,
    scraped_at:   new Date().toISOString(),
    total_images: imageUrls.length,
    urls:         imageUrls,
  };

  // Download as JSON
  const blob = new Blob([JSON.stringify(output, null, 2)], { type: "application/json" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = `pinterest_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  console.log(`%c📥 File downloaded: ${a.download}`, "color:#3b82f6;font-weight:bold");
  console.log("Give this file to the AI and it will import the images into your library.");

  return output;
})();
