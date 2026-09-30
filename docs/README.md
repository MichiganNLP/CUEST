# IQUEST project website

Static project page for “The Curious Case of Curiosity across Cultures: Evaluating Cross-Cultural Information-Seeking Questions in Humans and LLMs,” accepted to AACL 2026 Main Conference.

## Run locally

No build step or package installation is required.

```bash
cd /home/anganab/curiosity_culture_site
python3 -m http.server 8000
```

Open <http://localhost:8000> in a browser. Stop the server with `Ctrl+C`.

## Files

- `index.html` — page content and accessible chart markup
- `styles.css` — responsive layout, visuals, and chart styling
- `script.js` — scroll reveal and smooth navigation
- `assets/paper.pdf` — local paper linked from the page

All charts are rebuilt from the paper's reported values with HTML and CSS. No paper figures or screenshots are used.
