// Typed access to cv.json, which is generated from brain/docs/refs/cv
// (`make -C docs/refs/cv publish-site`). Edit the CV there, not here.
import raw from './cv.json'

export type Detail = string | { label: string; url?: string }

export type CvItem = {
  id?: string
  title: string
  authors?: string[]
  venue?: string
  institution?: string
  organization?: string
  location?: string
  date?: string
  year?: string | number
  url?: string
  links?: { label: string; url: string }[]
  award?: string
  details?: Detail[]
}

type Section = { id: string; title: string; note?: string; items: CvItem[] }

const sections = raw.sections as Section[]

export function cvSection(id: string): CvItem[] {
  return sections.find((s) => s.id === id)?.items ?? []
}

export const detailText = (d: Detail) => (typeof d === 'string' ? d : d.label)

// 'Mar 2022 - present' -> 'Mar 2022 – present'
export const dash = (s?: string | number) => String(s ?? '').replace(/\s-\s/g, ' – ')

export const publications = cvSection('publications')
export const education = cvSection('education')
export const experience = cvSection('experience')
export const honors = cvSection('honors')
export const currentResearch = cvSection('current_research')
export const service = cvSection('service') as (CvItem & { role?: string; venues?: string[] })[]

const ME = 'Junoh Kang'

// Lead-authorship label for interest bullets: '1st author' when listed first (equal
// contribution or not), 'Co-1st author' when a later-listed equal first author, '' otherwise.
// Other positions stay unlabeled; the publication list shows the full author order and `*`.
export function authorRole(pub: CvItem): string {
  const authors = pub.authors ?? []
  const i = authors.findIndex((a) => a.replace(/\*$/, '') === ME)
  if (i === 0) return '1st author'
  if (i > 0 && authors.slice(0, i + 1).every((a) => a.endsWith('*'))) return 'Co-1st author'
  return ''
}
