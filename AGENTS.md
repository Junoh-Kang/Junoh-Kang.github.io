# Project Router

Apply these instructions only at this repository root.

## Role

- This file routes work; it is not a wiki or project-specific workflow spec.
- Prefer the nearest subfolder `AGENTS.md` for code, paper, data, artifact, or domain-specific work.
- If no nearer instructions exist, inspect the relevant files before deciding how to work.

## Edit Boundaries

- <hard-gate> Style files are protected. Do not modify `_sass/**`, `assets/css/**`, or any `*.css`, `*.scss`, or `*.sass` file without the user's explicit approval for that specific style change.
- <hard-gate> If a requested task appears to require a protected style-file change, stop and ask for approval before editing those files.
- Do not edit generated or dependency output such as `_site/**`, `.jekyll-cache/**`, `.bundle/**`, `vendor/**`, or `Gemfile.lock` unless the user explicitly asks for build or dependency output changes.
- Treat `_site/**` as generated preview/deploy output. Make source changes in `_posts/**`, `_news/**`, `_pages/**`, `_includes/**`, `_layouts/**`, `_data/**`, assets, or config files as appropriate, then rebuild if needed.
- Content surfaces such as `_posts/**`, `_news/**`, `_pages/**`, and post-local assets under `blog/post/**` are looser editing areas. Inspect the relevant files first, then make narrowly scoped edits that fit the existing format.
- Do not broaden a content edit into layout, theme, or style changes unless the user explicitly approves that broader scope.

## Skills

- Use project-mounted or generic skills only when the user names them or a nearer `AGENTS.md` enables them for the task.
