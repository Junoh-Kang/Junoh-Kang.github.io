# Site content

Everything you edit by hand lives in two folders:

- `content/` holds the home page.
- `blog/` holds the posts.

Code under `src/` normally does not need to change.

## Home page (`content/`)

| File | What it controls |
| --- | --- |
| `1-profile.yaml` | Left sidebar: name, role, one-line pitch, status badge, links, photo, CV link |
| `1-photo.png` | Profile photo (the file named by `photo:` in `1-profile.yaml`) |
| `2-about.md` | The About paragraph, in Markdown (`[text](url)` for links) |
| `3-research-interests.yaml` | Research Interests: topics, and one line per backing paper |
| `4-news.yaml` | News, newest first. See the format at the top of the file. |
| `5-cv/` | Shortcut to the CV source: Education, Experience, Honors, and Publications (see below) |

`3-research-interests.yaml` refers to papers by their CV id, for example `kim2024fifo`. The build fails with a message that names the file if an id is not in the CV, or if a news `linkText` does not appear in its `text`.

## CV, education, experience, honors, and the publication list

These come from the CV, not from this folder. The source of truth is `brain/docs/refs/cv/`. On this Mac, `content/5-cv` is a shortcut (symlink) to it, so you can open the CV files from here. The shortcut is git-ignored because the CV source has private items; it does not exist in other checkouts.

- Papers are in `sections/publications.yml`, including `id`, venue, links, and an optional `award` shown next to the venue.
- Education, experience, and honors are in their own `sections/*.yml`.

After editing, regenerate the site data and the CV PDF:

```bash
make -C /Users/junoh/brain/docs/refs/cv publish-site
```

This writes `src/data/cv.json` and `public/assets/pdf/Junoh_Kang_CV.pdf`. Do not edit those two files by hand.

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
- Math uses `$...$` inline and `$$` on their own lines for display equations. It is rendered with MathJax at build time, so `\label{}` and `\eqref{}` work.

## Preview

```bash
npm run build && npx astro preview --port 4329
```

Then open http://localhost:4329.
