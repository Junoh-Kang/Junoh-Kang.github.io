#!/usr/bin/env python3
"""Convert al-folio Jekyll posts into Astro Pure blog entries.

Usage: migrate_post.py <old-repo>

For every _posts/YYYY-MM-DD-*.md in <old-repo>, writes
src/content/blog/<slug>/index.md, copies every /blog/post/YYYYMMDD/ folder the
post references into public/ under the same path, and writes
scripts/redirects.json mapping the old post URL (taken from the old repo's
built _site/) to the new one. Exits non-zero if Jekyll-only syntax survives.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MATH_BLOCK = "\x00MATH{}\x00"


def split_front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise SystemExit("no front matter")
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


def jekyll_slug(stem):
    # Matches the old site's /blog/:year/:title/ (Jekyll default slugify keeps case and commas).
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9,._~!$&'()+;=@]+", "-", stem)).strip("-")


def new_slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def protect_math(body):
    """Swap math and code out so text rewrites never touch them."""
    saved = []

    def keep(m):
        saved.append(m.group(0))
        return MATH_BLOCK.format(len(saved) - 1)

    pattern = re.compile(r"```.*?```|\$\$.*?\$\$|(?<!\\)\$[^$\n]+?\$", re.S)
    return pattern.sub(keep, body), saved


def restore_math(body, saved):
    return re.sub(r"\x00MATH(\d+)\x00", lambda m: saved[int(m.group(1))], body)


def liquid_path(expr):
    m = re.search(r"""['"]([^'"]+)['"]""", expr)
    return m.group(1) if m else expr.strip()


def convert_liquid(body):
    # {{ '/path' | relative_url }} -> /path
    body = re.sub(r"\{\{\s*(['\"][^}]*?)\|\s*relative_url\s*\}\}", lambda m: liquid_path(m.group(1)), body)

    # ![alt](/path){:style="...width:80%..."} -> <img>
    def img(m):
        alt, path, attrs = m.group(1), m.group(2), m.group(3) or ""
        width = re.search(r"width:\s*([0-9.]+%)", attrs)
        style = f' style="width:{width.group(1)}"' if width else ""
        return f'<img src="{path}" alt="{alt}" class="zoomable"{style} />'

    body = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)(\{:[^}]*\})?", img, body)

    # {% include figure.html path="blog/post/..." ... %} -> <img>
    def figure(m):
        path = re.search(r'path="([^"]+)"', m.group(0)).group(1)
        return f'<img src="/{path.lstrip("/")}" alt="" class="zoomable" />'

    body = re.sub(r"\{%-?\s*include\s+figure\.(?:html|liquid)[^%]*%\}", figure, body)

    # {% details X %} ... {% enddetails %} -> <details>
    def summary(m):
        label = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", m.group(1).strip())
        return f"<details>\n<summary>{label}</summary>\n\n"

    body = re.sub(r"\{%-?\s*details\s+(.*?)\s*-?%\}\n?", summary, body)
    body = re.sub(r"\{%-?\s*enddetails\s*-?%\}", "\n</details>", body)

    # {% highlight lang %} ... {% endhighlight %} -> fenced code
    body = re.sub(r"\{%-?\s*highlight\s+(\w+)[^%]*%\}", r"```\1", body)
    body = re.sub(r"\{%-?\s*endhighlight\s*-?%\}", "```", body)

    body = re.sub(r"\{%-?\s*(raw|endraw)\s*-?%\}", "", body)
    body = re.sub(r"\{::nomarkdown\}|\{:/nomarkdown\}|\{:/\}", "", body)
    body = re.sub(r"\{:\s*target=\"?_blank\"?\s*\}", "", body)

    # PDF embeds: fill the column instead of a fixed 800px
    body = re.sub(r'<iframe ([^>]*?)width="\d+"', r'<iframe \1width="100%"', body)
    return body


def convert_refs(body):
    # al-folio typeset \ref/\eqref anywhere on the page; remark-math only inside $...$
    return re.sub(r"(?<!\$)\\(eq)?ref\{([^}]+)\}", lambda m: f"${m.group(0)}$", body)


def parse_bib(path):
    entries = {}
    for chunk in re.split(r"\n(?=@)", path.read_text()):
        m = re.match(r"@\w+\{([^,\s]+),(.*)", chunk, re.S)
        if not m:
            continue
        fields = dict(
            (k.lower(), re.sub(r"[{}]", "", v).strip())
            for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(2))
        )
        entries[m.group(1)] = fields
    return entries


def person(name):
    # "Ho, Jonathan" -> "Jonathan Ho"
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}" if first else name.strip()


def format_ref(e):
    authors = [person(a) for a in re.split(r"\s+and\s+", e.get("author", "")) if a.strip()]
    who = authors[0] + (" et al." if len(authors) > 2 else f" and {authors[1]}" if len(authors) == 2 else "") if authors else ""
    venue = e.get("booktitle") or e.get("journal") or e.get("publisher") or ""
    parts = [p for p in (who, f"*{e.get('title', '')}*", venue, e.get("year", "")) if p]
    text = ", ".join(parts[:2]) + (f" ({', '.join(parts[2:])})" if parts[2:] else "")
    return f"[{text}]({e['url']})" if e.get("url") else text


def convert_citations(body, bib):
    order = []

    def cite(m):
        nums = []
        for key in [k.strip() for k in m.group(1).split(",")]:
            if key not in bib:
                raise SystemExit(f"unknown citation key: {key}")
            if key not in order:
                order.append(key)
            n = order.index(key) + 1
            nums.append(f'<a href="#ref-{n}">{n}</a>')
        return f'<sup class="cite">[{", ".join(nums)}]</sup>'

    body = re.sub(r'<d-cite key="([^"]+)"></d-cite>', cite, body)
    if not order:
        return body
    refs = "\n".join(f'<li id="ref-{i}">\n\n{format_ref(bib[k])}\n\n</li>' for i, k in enumerate(order, 1))
    return body.rstrip() + f"\n\n## References\n\n<ol class=\"references\">\n{refs}\n</ol>\n"


def normalize_math(body):
    """Give every display equation its own `$$` fence lines, as remark-math requires."""
    inline_pair = re.compile(r"(?<!\$)\$\$([^$\n]+?)\$\$(?!\$)")
    out = []
    in_display = in_code = False

    def open_block(content=""):
        if out and out[-1].strip() != "":
            out.append("")
        out.append("$$")
        if content.strip():
            out.append(content)

    def close_block(content=""):
        if content.strip():
            out.append(content)
        out.extend(["$$", ""])

    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        if in_code:
            out.append(line)
            continue
        if in_display:
            if stripped == "$$":
                close_block()
                in_display = False
            elif stripped.endswith("$$"):
                close_block(stripped[:-2])
                in_display = False
            else:
                out.append(line)
            continue
        if stripped == "$$":
            open_block()
            in_display = True
            continue
        # A line that is only `$$ ... $$` is a display equation.
        whole = re.fullmatch(r"\$\$(.+?)\$\$", stripped)
        if whole and "$$" not in whole.group(1):
            open_block(whole.group(1).strip())
            close_block()
            continue
        # kramdown inline $$x$$ -> $x$
        line = inline_pair.sub(r"$\1$", line)
        # A `$$` left over opens a display block mid-line: split it off.
        if "$$" in line:
            before, _, after = line.partition("$$")
            if before.strip():
                out.append(before.rstrip())
            if after.rstrip().endswith("$$"):
                open_block(after.rstrip()[:-2])
                close_block()
            else:
                open_block(after)
                in_display = True
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def migrate(old_repo, post_path, redirects):
    fm, body = split_front_matter(post_path.read_text())
    date = str(fm["date"])[:10]
    slug = new_slug(fm["title"])
    tags = list(dict.fromkeys((fm.get("categories") or []) + (fm.get("tags") or [])))
    new_fm = {"title": fm["title"], "description": fm.get("description") or "", "publishDate": date, "tags": tags}

    body = normalize_math(body)
    # A heading glued to the end of a math span ("...$##### Definition") starts its own line.
    body = re.sub(r"\$(#{2,6} )", r"$\n\n\1", body)
    body, saved = protect_math(body)
    body = convert_liquid(body)
    body = convert_refs(body)
    if fm.get("bibliography"):
        body = convert_citations(body, parse_bib(old_repo / "assets/bibliography" / fm["bibliography"]))
    body = restore_math(body, saved)

    if fm.get("attachments"):
        body = f"**Slides:** [PDF]({fm['attachments']})\n\n" + body.lstrip("\n")

    dest = ROOT / "src/content/blog" / slug / "index.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + yaml.safe_dump(new_fm, sort_keys=False, allow_unicode=True) + "---\n\n" + body.lstrip("\n")
    dest.write_text(text)

    for stamp in sorted(set(re.findall(r"/blog/post/(\d{8})/", text))):
        src = old_repo / "blog/post" / stamp
        if src.is_dir():
            shutil.copytree(src, ROOT / "public/blog/post" / stamp, dirs_exist_ok=True)
        else:
            print(f"warning: {post_path.name} references missing /blog/post/{stamp}/", file=sys.stderr)

    old_url = f"/blog/{date[:4]}/{jekyll_slug(post_path.stem[11:])}/"
    if not (old_repo / "_site" / old_url.strip("/")).is_dir():
        raise SystemExit(f"old URL not found in _site: {old_url}")
    redirects[old_url] = f"/blog/{slug}"

    leftovers = re.findall(r"\{%.*?%\}|\{:[^}]*\}|\{\{.*?\}\}|<d-cite", text)
    if leftovers:
        raise SystemExit(f"{post_path.name}: Jekyll syntax left: {leftovers}")
    return slug


def main():
    old_repo = Path(sys.argv[1])
    redirects = {}
    for post in sorted((old_repo / "_posts").glob("[0-9]*.md")):
        print(migrate(old_repo, post, redirects))
    (ROOT / "scripts/redirects.json").write_text(json.dumps(redirects, indent=2) + "\n")


if __name__ == "__main__":
    main()
