# Blog Instructions

## Writing

- Write top-down: state the main point before its explanation or evidence.
- Use sentence case for blog post and series titles, section headings, and captions; preserve the capitalization of proper nouns and acronyms.
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

### Citations and links

Choose the body citation format by source type, while keeping evidence sources in `## References`.

- For papers, use the paper title with a numbered citation at first mention: `<sup class="cite">[<a href="#ref-1">1</a>]</sup>`. Later citations may use the number alone. In References, include authors, title, year, and a link to the paper.
- For blog posts, link the source name or post title directly in the body without a citation number. Also include the source in References with its author or organization, title, and link.
- Use an `<ol class="references">` list with `<li id="ref-1">` entries for References. Order all entries by first mention; paper citation numbers match their entries even when blog entries have no numbered body citation.
- For other sources supporting a claim, number, quote, definition, or reused figure, use numbered citations and matching References entries.
- Link tools, repositories, skills, datasets, talks, and related posts inline when they are only pointers for readers to open or use; these pointers do not go in References. If they serve as evidence, apply the source-type rules above.
- Keep each source's body citation format consistent throughout a post and list it only once in References.

## Figures, Tables, and Mathematics

- Introduce each figure, table, or equation with its purpose.
- Use captions for source attribution and information needed to read the visual independently.
- Keep substantive interpretation and implications in the body.
- Keep terminology and notation consistent throughout the post.
- Check readability at the actual displayed size.
- Math uses `$...$` inline and `$$` on their own lines for display equations. It is rendered with MathJax at build time, so `\label{}` and `\eqref{}` work.

### SVG diagrams

- Create conceptual diagrams as editable SVG assets.
- Prefer angular shapes and neutral colors; reserve color for semantic distinctions and emphasis.
- Use Helvetica with Arial and sans-serif fallbacks. Bold primary labels inside shapes; keep standalone labels regular.
- Use filled equilateral triangle arrowheads. Keep arrow-to-object gaps consistent and use equal spacing between comparable stages.
- Align labels with the objects they describe; split long labels across two lines when needed.
- Render each new or modified SVG at its intended display size and check for clipping, overlap, readability, and consistent spacing. This asset-level check does not require a site build.

## Files and Metadata

Each post is one folder:

```text
blog/YYYY-MM-DD-<slug>/
├── index.md        # front matter: title, publishDate (YYYY-MM-DD), tags
└── ...             # the post's figures, PDFs, and other files
```

- The folder name starts with the post's `publishDate`, so posts sort by date. The build stops if the two differ.
- The post URL is `/blog/<slug>`, without the date.
- Files next to `index.md` are served at `/blog/post/<YYYYMMDD>/<file>`, where `YYYYMMDD` is the post's `publishDate`. Refer to them with that path inside the post, for example `<img src="/blog/post/20260312/fig/cm.png" />`. This matches the old Jekyll URLs.
- Place slide decks (PDF or HTML) beside `index.md` and set `slides: presentation.pdf` or the matching filename. For a series-wide deck, set `slides` only on the post with `seriesOrder: 1`. Later parts link to that deck unless they specify their own. The link appears in the sidebar after the series list or table of contents. Run `scripts/slides-thumb.sh blog/<post>/` from the repository root to generate `slides-thumb.png` beside the deck.
- Posts use optional, free-form `tags` (for example, `[diffusion]` or `[life]`); there is no fixed category list or required post type. Tag names are trimmed, lowercased, and deduplicated.
- Tags name the post's topics. Reuse existing names when they fit, and add a new tag only when at least two posts will use it; a narrower topic stays under its broader tag until then.
- `finance` is for research on financial data and modeling; investment views and market interpretation use `investing`.
- The default blog listing includes every published post, including `life`. Clicking a tag shows posts with that tag; only tags used by published posts appear in production navigation.
- Language is independent of tags; set the optional `language` field per post as needed.
- Field types, defaults, and constraints are defined in [src/content.config.ts](../src/content.config.ts).

## Starting Templates

These outlines are starting points. Change headings and order to fit the argument. The comments are drafting prompts; remove them when they are no longer useful. Completed paragraphs do not require claim comments.

### Body-focused post

```markdown
---
title: "Post title"
publishDate: 'YYYY-MM-DD'
tags: [diffusion]
---

## Overview
<!-- Explain the scope and main takeaway. -->

## Background and question
<!-- Introduce the problem and necessary concepts. -->

## Method
<!-- Explain the mechanism, with equations or figures as needed. -->

## Results and discussion
<!-- Present evidence, interpretation, and limitations. -->

## References
```

### Slides-focused post

```markdown
---
title: "Post title"
publishDate: 'YYYY-MM-DD'
tags: [diffusion]
slides: presentation.pdf
---

## Overview
<!-- Introduce the topic and summarize the main takeaways. -->
```

## Verification

- During drafting, verify only the requested source changes.
- After the post is complete, run `npm run build` from the repository root and check affected links and assets only when the user explicitly requests final verification.
- During that requested verification, inspect the rendered post for layout, figure, or math changes. Use the development server for drafts and the built-site preview for published posts.
- For instructions-only changes, check the diff and local documentation links; a site build is unnecessary.
