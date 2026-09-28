// Serve each post's attachments from its own folder at the old Jekyll URLs.
//
// blog/YYYY-MM-DD-<slug>/index.md sits next to its figures and PDFs. Every non-index file in that
// folder is published at /blog/post/<YYYYMMDD>/<relative path>, where YYYYMMDD comes from
// the post's `publishDate`. This keeps old links (and the paths inside posts) working.
import {
  copyFileSync,
  createReadStream,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync
} from 'node:fs'
import { dirname, extname, join, relative, sep } from 'node:path'
import type { AstroIntegration } from 'astro'

const BLOG_DIR = join(process.cwd(), 'blog')

const MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
  '.html': 'text/html; charset=utf-8',
  '.mp4': 'video/mp4'
}

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name)
    return statSync(p).isDirectory() ? walk(p) : [p]
  })
}

/** URL path -> absolute file path, for every attachment of every post. */
export function blogAssetMap(): Map<string, string> {
  const map = new Map<string, string>()
  for (const slug of readdirSync(BLOG_DIR)) {
    const postDir = join(BLOG_DIR, slug)
    if (!statSync(postDir).isDirectory()) continue
    const index = ['index.md', 'index.mdx']
      .map((f) => join(postDir, f))
      .find((f) => {
        try {
          return statSync(f).isFile()
        } catch {
          return false
        }
      })
    if (!index) continue
    const date = /^publishDate:\s*'?(\d{4})-(\d{2})-(\d{2})/m.exec(readFileSync(index, 'utf8'))
    if (!date) throw new Error(`blog-assets: no publishDate in ${index}`)
    const stamp = `${date[1]}${date[2]}${date[3]}`
    // Folder names are YYYY-MM-DD-<slug>; the prefix must match publishDate so URLs stay put.
    const prefix = /^(\d{4})-(\d{2})-(\d{2})-/.exec(slug)
    if (!prefix || `${prefix[1]}${prefix[2]}${prefix[3]}` !== stamp)
      throw new Error(
        `blog-assets: folder "${slug}" must start with its publishDate (${date[1]}-${date[2]}-${date[3]}-)`
      )
    for (const file of walk(postDir)) {
      if (file === index || file.endsWith('.DS_Store')) continue
      const url = `/blog/post/${stamp}/${relative(postDir, file).split(sep).join('/')}`
      if (map.has(url)) throw new Error(`blog-assets: two posts publish ${url}`)
      map.set(url, file)
    }
  }
  return map
}

export default function blogAssets(): AstroIntegration {
  return {
    name: 'blog-assets',
    hooks: {
      'astro:server:setup': ({ server }) => {
        server.middlewares.use((req, res, next) => {
          const file = blogAssetMap().get(decodeURIComponent((req.url ?? '').split('?')[0]))
          if (!file) return next()
          res.setHeader(
            'Content-Type',
            MIME[extname(file).toLowerCase()] ?? 'application/octet-stream'
          )
          createReadStream(file).pipe(res)
        })
      },
      'astro:build:done': ({ dir, logger }) => {
        const out = dir.pathname
        const map = blogAssetMap()
        for (const [url, file] of map) {
          const target = join(out, url)
          mkdirSync(dirname(target), { recursive: true })
          copyFileSync(file, target)
        }
        logger.info(`copied ${map.size} post attachments`)
      }
    }
  }
}
