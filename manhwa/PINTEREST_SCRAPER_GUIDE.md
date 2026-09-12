# Pinterest Image Scraper — Quick Guide

Use this when you want to pull images from Pinterest into your library without
manually scrolling and screenshotting.  The script runs entirely in your browser
— no login credentials leave your computer.

---

## Step-by-step

### 1. Open Pinterest in your browser
Go to any page you want to scrape:
- A specific board: `https://www.pinterest.com/yourname/board-name/`
- A search result: `https://www.pinterest.com/search/pins/?q=manhwa+medieval`
- Your saved pins: `https://www.pinterest.com/yourname/_saved/`

Make sure you are **logged in** so Pinterest loads the full content.

### 2. Open DevTools Console
| OS | Shortcut |
|----|----------|
| Windows / Linux | `F12` then click **Console** tab |
| Mac | `Cmd + Option + I` then click **Console** tab |

### 3. Paste and run the script
Open the file `pinterest_console_scraper.js` in this project.  
Copy **the entire contents**.  
Paste into the Console and press **Enter**.

You will see progress messages like:
```
📌 Pinterest Scraper starting…
Scroll 1/120 — 28 images (+28)
Scroll 5/120 — 140 images (+32)
…
✅ Done — 347 unique images collected.
📥 File downloaded: pinterest_1720000000000.json
```

The script scrolls automatically — just wait, don't touch the page.

### 4. A JSON file downloads automatically
Your browser saves a file named something like `pinterest_1720000000000.json`.

### 5. Give the file to the AI
Drag the downloaded JSON file into the chat and say:
> "Import these Pinterest images into my library"

The AI will read the URL list and batch-import them.

---

## Tips

**Too few images?**  
Increase `maxScrolls` at the top of the script (default 120 ≈ 600 pins).

**Script stops too early?**  
Increase `staleAfter` (default 6 — stops after 6 scrolls with no new images).

**Pinterest loads slowly?**  
Increase `scrollDelay` to `2500` or `3000` ms.

**Want only high-res?**  
The script already upgrades URLs to `/originals/` automatically.

---

## What the JSON file looks like

```json
{
  "source_url": "https://www.pinterest.com/yourname/board-name/",
  "scraped_at": "2026-08-07T12:00:00.000Z",
  "total_images": 347,
  "urls": [
    "https://i.pinimg.com/originals/ab/cd/ef/abcdef123.jpg",
    "https://i.pinimg.com/originals/12/34/56/123456abc.jpg",
    ...
  ]
}
```

The AI reads the `urls` array and downloads each image into your library.
