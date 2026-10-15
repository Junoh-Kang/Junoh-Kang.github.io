# Blog Instructions

## Writing

- Write top-down: state the main point before its explanation or evidence.
- Give each paragraph one core claim.
- Open with the post's scope and main takeaway.
- Choose sections around the questions the post answers. Do not force every post into the same outline.
- Define unfamiliar terms and notation before relying on them.
- Use emphasis selectively for key claims, not entire paragraphs.

## Evidence

- Distinguish source findings, source authors' interpretations, and the post author's interpretations.
- Keep claims within the conditions supported by the evidence.
- Make external claims, numerical results, and reused figures traceable to their sources.
- Never invent results, references, or bibliographic metadata. Report missing evidence explicitly.

## Figures, Tables, and Mathematics

- Introduce each figure, table, or equation with its purpose.
- Use captions for source attribution and information needed to read the visual independently.
- Keep substantive interpretation and implications in the body.
- Keep terminology and notation consistent throughout the post.
- Check readability at the actual displayed size.
- Math uses `$...$` inline and `$$` on their own lines for display equations. It is rendered with MathJax at build time, so `\label{}` and `\eqref{}` work.

## Files and Metadata

Each post is one folder:

```text
blog/YYYY-MM-DD-<slug>/
├── index.md        # front matter: title, description, publishDate (YYYY-MM-DD), tags
└── ...             # the post's figures, PDFs, and other files
```

- The folder name starts with the post's `publishDate`, so posts sort by date. The build stops if the two differ.
- The post URL is `/blog/<slug>`, without the date.
- Files next to `index.md` are served at `/blog/post/<YYYYMMDD>/<file>`, where `YYYYMMDD` is the post's `publishDate`. Refer to them with that path inside the post, for example `<img src="/blog/post/20260312/fig/cm.png" />`. This matches the old Jekyll URLs.
- A slide deck goes beside `index.md` and is named in the front matter, for example `slides: presentation.pdf` (PDF or HTML). The post then shows a Slides card under its title. Make the card's preview image once with `scripts/slides-thumb.sh blog/<post>/`, run from the repository root, which writes `slides-thumb.png` next to the deck.
- Field types, defaults, and constraints are defined in [src/content.config.ts](../src/content.config.ts).

## Starting Templates

These outlines are starting points. Change headings and order to fit the argument. The comments are drafting prompts; remove them when they are no longer useful. Completed paragraphs do not require claim comments.

### Body-focused post

```markdown
---
title: "Post title"
description: "A short description of the post."
publishDate: 'YYYY-MM-DD'
tags:
  - paper-review
---

## Overview
<!-- Explain the scope and main takeaway. -->

## Background and Question
<!-- Introduce the problem and necessary concepts. -->

## Method
<!-- Explain the mechanism, with equations or figures as needed. -->

## Results and Discussion
<!-- Present evidence, interpretation, and limitations. -->

## References
```

### Slides-focused post

```markdown
---
title: "Post title"
description: "A short description of the post."
publishDate: 'YYYY-MM-DD'
tags:
  - study-note
slides: presentation.pdf
---

## Overview
<!-- Introduce the topic and summarize the main takeaways. -->
```

## Verification

- After editing a post or its assets, run `npm run build` from the repository root and check affected links and assets.
- For layout, figure, or math changes, inspect the rendered post. Run `npx astro preview --port 4329` after building, then open `http://localhost:4329/blog/<slug>`.
- For instructions-only changes, check the diff and local documentation links; a site build is unnecessary.
