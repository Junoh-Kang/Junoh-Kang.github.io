# Site content

Everything you edit by hand lives in three folders:

- `content/` holds the home page.
- `cv/` holds the CV source (education, publications, and the rest).
- `blog/` holds the posts.

Code under `src/` normally does not need to change.

## Home page (`content/`)

| File | What it controls |
| --- | --- |
| `1-profile.yaml` | Left sidebar: name, role, one-line pitch, status badge, links, photo, CV link |
| `photos/` | Square profile photo candidates; `photo:` in `1-profile.yaml` picks one. Crop new photos to 1:1 before adding them. |
| `2-about.md` | The About paragraph, in Markdown (`[text](url)` for links) |
| `3-research-interests.yaml` | Research Interests: topics, and one line per backing paper |
| `4-news.yaml` | News, newest first. See the format at the top of the file. |

`3-research-interests.yaml` refers to papers by their CV id, for example `kim2024fifo`. The build fails with a message that names the file if an id is not in the CV, or if a news `linkText` does not appear in its `text`.

## CV: education, experience, publications, honors, and service

These come from the CV, not from this folder. The source of truth is `cv/` at the repository root (see `cv/README.md`). Items marked `visibility: private` or `archive` stay out of the site and the PDF, but they are still readable in this public repository.

- Papers are in `sections/publications.yml`, including `id`, venue, links, and an optional `award` shown next to the venue.
- Education, experience, and honors are in their own `sections/*.yml`.
- Academic Services come from `sections/service.yml`. Each item is `role` plus a `venues` list. The site groups them by role: `Organizer`, `Invited Talk`, and `Journal Reviewer` / `Conference Reviewer` (shown under Reviewer, years dropped). A group with no public items is hidden.

After editing, regenerate the site data and the CV PDF:

```bash
make -C cv publish-site
```

This writes `src/data/cv.json` and `public/assets/pdf/Junoh_Kang_CV.pdf`. Do not edit those two files by hand.

The pre-commit hook in `.githooks/` does this for you: when a commit stages changes under `cv/`, it regenerates both files and adds them to the same commit. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

## Build locally

```bash
scripts/build.sh            # regenerate the CV outputs, then build the site into dist/
scripts/build.sh --skip-cv  # build the site only
```

This only builds. The live site updates when `master` is pushed and the GitHub Actions deploy runs. The social preview image (`/images/social-card.png`) is drawn from `1-profile.yaml` during every build, and the footer shows the date of the latest commit.

## Posts (`blog/`)

Each post is one folder:

```text
blog/YYYY-MM-DD-<slug>/
├── index.md        # front matter: title, description, publishDate (YYYY-MM-DD), tags
└── ...             # the post's figures, PDFs, and other files
```

- The folder name starts with the post's `publishDate`, so posts sort by date. The build stops if the two differ.
- The post URL is `/blog/<slug>`, without the date.
- Files next to `index.md` are served at `/blog/post/<YYYYMMDD>/<file>`, where `YYYYMMDD` is the post's `publishDate`. Refer to them with that path inside the post, for example `<img src="/blog/post/20260312/fig/cm.png" />`. This matches the old Jekyll URLs.
- A slide deck goes beside `index.md` and is named in the front matter, for example `slides: presentation.pdf` (PDF or HTML). The post then shows a Slides card under its title. Make the card's preview image once with `scripts/slides-thumb.sh blog/<post>/`, which writes `slides-thumb.png` next to the deck.
- Math uses `$...$` inline and `$$` on their own lines for display equations. It is rendered with MathJax at build time, so `\label{}` and `\eqref{}` work.

## Preview

```bash
npm run build && npx astro preview --port 4329
```

Then open http://localhost:4329.
