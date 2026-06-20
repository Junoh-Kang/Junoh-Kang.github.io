# Project Router

Apply these instructions only at this repository root.

## Role

- This file routes work and defines edit boundaries; it is not a general wiki or project-specific workflow spec.
- Prefer the nearest subfolder `AGENTS.md` for code, paper, data, artifact, or domain-specific work.
- If no nearer instructions exist, inspect the relevant files before deciding how to work.

## Never Edit Without Explicit User Approval

- Style files are protected. Do not modify `_sass/**`, `assets/css/**`, or any `*.css`, `*.scss`, or `*.sass` file without the user's explicit approval for that specific style change.
- If a requested task appears to require a protected style-file change, stop and ask for approval before editing those files.
- Do not edit generated or dependency output such as `_site/**`, `.jekyll-cache/**`, `.bundle/**`, `vendor/**`, or `Gemfile.lock` unless the user explicitly asks for build, preview, deploy, or dependency output changes.
- Treat `_site/**` as generated preview/deploy output. When output changes are explicitly requested, make source changes first, then rebuild instead of editing `_site/**` directly.
- Do not broaden a content edit into layout, theme, or style changes unless the user explicitly approves that broader scope.

## Default Task-Specific Edit Boundaries

- For these common tasks, edit only the listed source files unless the user explicitly expands scope.
- About-me edits should normally modify only `_pages/about.md`.
  - If the request explicitly changes the profile image, also allow the relevant `assets/img/*` profile image.
  - If the request explicitly changes site identity, contact, or social links, also allow the relevant fields in `_config.yml`.
  - Do not edit `_layouts/about.html`, `_includes/**`, or style files for an about-me text change without explicit approval.
- Bib/publication-data edits should normally modify only `_bibliography/papers.bib`.
  - For publication badges or coauthor links, also allow `_data/venues.yml` or `_data/coauthors.yml` when directly needed.
  - For publication preview images or local PDFs, also allow `assets/img/publication_preview/*` or `assets/pdf/*` when directly needed.
  - Do not edit `_layouts/bib.html` or `_includes/selected_papers.html` unless the requested change is about publication rendering behavior.
- Publications-page copy or metadata edits should normally modify only `_pages/publications.md`.
  - Do not change the bibliography source, grouping, or rendering unless the user explicitly asks for a publications-page behavior change.
- Blog-post edits should normally modify only the target `_posts/*.md` file.
  - For post-local images, PDFs, or other attachments, also allow the matching `blog/post/YYYYMMDD/*` directory.
  - For Distill citations, also allow the post's declared bibliography under `assets/bibliography/*` when directly needed.
  - Use `_posts/_template.md` and `_posts/ref/*.md` as references, not as edit targets, unless the user asks to change templates or examples.
  - Do not edit `blog/index.html`, `_layouts/post.html`, `_layouts/distill.html`, or blog config for a single-post content change without explicit approval.
- News edits should normally modify only the target `_news/*.md` file.
- CV edits should normally modify only `_pages/cv.md` and `_data/cv.yml`.
- Repository-list edits should normally modify only `_pages/repositories.md` and `_data/repositories.yml`.
