// Loads the hand-edited home content from /content (YAML), validating it at build time.
// Edit the files in /content, not this module. The CV itself comes from ./cv.ts.
import { parse } from 'yaml'

import newsRaw from '../../content/news.yaml?raw'
import profileRaw from '../../content/profile.yaml?raw'
import publicationsRaw from '../../content/publications.yaml?raw'
import interestsRaw from '../../content/research-interests.yaml?raw'
import { publications } from './cv'

type Profile = {
  name: string
  role: string
  pitch: string
  status?: string
  photo: string
  cv: string
  links: { label: string; icon: string; href: string }[]
}
type NewsItem = { date: string; text: string; links?: { text: string; url: string }[] }
type Interest = { title: string; papers: { paper: string; note: string }[] }

function fail(file: string, msg: string): never {
  throw new Error(`content/${file}: ${msg}`)
}

export const profile = parse(profileRaw) as Profile

export const news = (parse(newsRaw) as NewsItem[]).map((n) => {
  const date = String(n.date)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) fail('news.yaml', `bad date "${date}"`)
  for (const l of n.links ?? []) {
    if (!l.url) fail('news.yaml', `link "${l.text}" on ${date} has no url`)
    if (!n.text.includes(l.text)) fail('news.yaml', `link text "${l.text}" is not in "${n.text}"`)
  }
  return { ...n, date }
})

const paperIds = new Set(publications.map((p) => p.id))
function checkPaper(file: string, id: string) {
  if (!paperIds.has(id)) fail(file, `unknown paper id "${id}" (not in the CV)`)
}

export const researchInterests = parse(interestsRaw) as Interest[]
researchInterests.forEach((r) =>
  r.papers.forEach((p) => checkPaper('research-interests.yaml', p.paper))
)

export const pubBadges: Record<string, string> = parse(publicationsRaw)?.badges ?? {}
Object.keys(pubBadges).forEach((id) => checkPaper('publications.yaml', id))

/** Split a news sentence into plain and linked pieces, in the order the links appear. */
export function newsParts(n: NewsItem): { text: string; url?: string }[] {
  const links = [...(n.links ?? [])].sort((a, b) => n.text.indexOf(a.text) - n.text.indexOf(b.text))
  const parts: { text: string; url?: string }[] = []
  let pos = 0
  for (const l of links) {
    const i = n.text.indexOf(l.text, pos)
    if (i < 0) fail('news.yaml', `links overlap in "${n.text}"`)
    if (i > pos) parts.push({ text: n.text.slice(pos, i) })
    parts.push({ text: l.text, url: l.url })
    pos = i + l.text.length
  }
  if (pos < n.text.length) parts.push({ text: n.text.slice(pos) })
  return parts
}
