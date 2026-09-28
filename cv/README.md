# CV Canonical Source

This folder is the canonical source tree for Junoh Kang's CV.

The source is split by section so publication, service, teaching, and private reference updates do not require editing one large file. Generated blog data and generated PDF files are outputs, not sources.

## Editing Rule

- Edit canonical CV content only under `cv/`.
- Use `visibility: public`, `visibility: private`, or `visibility: archive` on each item.
- Edit section content in `sections/*.yml`.
- Edit profile/contact content in `profile.yml`.
- Use `cv.yml` for section order and output paths.
- Do not hand-edit generated blog CV data or generated PDF output after the exporter exists.
- Do not generate or maintain a direct HTML file from this folder. The blog builds HTML from the generated blog data YAML.

## Source Model

`cv.yml` assembles the public CV. It names the sections and declares target outputs.

`profile.yml` stores person-level data such as name, affiliation, and links.

Each `sections/*.yml` file stores one logical CV section. Entries should use simple structured fields instead of renderer-specific markup. For example, store a link in `url` instead of embedding an HTML anchor or LaTeX `\href` command.

Public export is item-level. An item with `visibility: public` appears in generated public outputs. Items with `visibility: private` or `visibility: archive` stay in the canonical source but are excluded from public outputs.

## Export Flow

```text
cv/cv.yml
+ cv/profile.yml
+ cv/sections/*.yml
  -> normalized CV model
  -> generated RenderCV YAML
  -> generated RenderCV PDF
  -> generated blog data YAML
  -> generated site data JSON
```

The site data JSON (`build/site.json`) feeds the Astro site. It holds the public profile and public section items in `cv.yml` order, with `visibility` removed. Publication items may carry an `id`, which the site uses to reference papers, `links` (`label` and `url` pairs) for per-paper links such as Paper, arXiv, Code, or Project, and `award`, a short award line the site shows next to the venue. A paper has no top-level `url`: the PDF and blog YAML link its title to alphaXiv, built from the `arXiv` link because it opens fast. Without an arXiv link the title links to `Paper`, the published PDF. The homepage uses the same rule. The PDF and blog YAML ignore `id` and `award`.

The PDF uses RenderCV with the `engineeringresumes` theme. The blog still builds its own HTML from the generated blog data YAML.

The exporter uses the `rendercv` command when it exists. If `rendercv` is not installed but `uv` is available, it runs RenderCV with `uv run --with rendercv[full] rendercv`.

The default build writes outputs under `cv/build/`. Publishing to the Astro site is a separate step that copies the generated site data JSON and PDF to the `publish_site` paths declared in `cv.yml`.

## Commands

Run checks:

```bash
make -C cv test
```

Generate all outputs (blog data YAML, site data JSON, and PDF) under `cv/build/`:

```bash
make -C cv all
```

Generate only the RenderCV input YAML:

```bash
make -C cv rendercv-data
```

Publish the site data JSON and PDF into the Astro site:

```bash
make -C cv publish-site
```

`publish-site` writes to `src/data/cv.json` and `public/assets/pdf/Junoh_Kang_CV.pdf` in this repository. Commit both with the source change.

## Acceptance Check

- A CV content change happens once in a canonical YAML file.
- The generated blog YAML and generated PDF contain the same `visibility: public` entries.
- `visibility: private` and `visibility: archive` items stay out of public blog and PDF outputs.
- The PDF is generated from `build/rendercv.yml`, not from a hand-edited LaTeX file.
- The blog repository treats CV output files as generated artifacts.

## Seed Sources

The initial files were seeded from the public blog CV data and the Overleaf/Awesome-CV export supplied in the current task.

- Blog data: `/Users/junoh/junoh-kang.github.io/_data/cv.yml`
- Overleaf export: `/Users/junoh/.codex/attachments/a4dfcff6-b9f8-470c-99a4-725f8c7080de/Junoh_Kang_CV.zip`
