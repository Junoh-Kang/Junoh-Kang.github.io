// Renders the default social preview image (1200x630) from content/1-profile.yaml at build time,
// so the card follows the profile's name, role, pitch, and photo. Edit the profile, not this file.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { APIRoute } from 'astro'
import satori from 'satori'
import sharp from 'sharp'

import { profile } from '../../data/content'

const root = process.cwd()
const NAVY = '#1f4e79'

const font = (file: string) => readFileSync(join(root, 'src/assets/fonts', file))

// Minimal element builder for satori, which accepts React-like objects without JSX.
const el = (type: string, style: Record<string, unknown>, children?: unknown) => ({
  type,
  props: { style, children }
})

export const GET: APIRoute = async () => {
  const photo = await sharp(join(root, 'content', profile.photo))
    .resize(560, 560, { fit: 'cover' })
    .jpeg({ quality: 85 })
    .toBuffer()

  const card = el(
    'div',
    {
      width: 1200,
      height: 630,
      display: 'flex',
      alignItems: 'center',
      gap: 60,
      padding: '0 80px',
      background: '#fff',
      borderTop: `14px solid ${NAVY}`,
      fontFamily: 'Inter',
      color: '#1e293b'
    },
    [
      {
        type: 'img',
        props: {
          src: `data:image/jpeg;base64,${photo.toString('base64')}`,
          width: 280,
          height: 280,
          style: { borderRadius: '50%', border: '6px solid #e8eef5' }
        }
      },
      el('div', { display: 'flex', flexDirection: 'column' }, [
        el(
          'div',
          { fontFamily: 'Source Serif 4', fontWeight: 600, fontSize: 84, lineHeight: 1.05 },
          profile.name
        ),
        el('div', { marginTop: 20, fontSize: 29, color: '#475569' }, profile.role),
        el('div', { marginTop: 12, fontSize: 31, fontWeight: 600, color: NAVY }, profile.pitch),
        el(
          'div',
          { marginTop: 44, fontSize: 26, fontWeight: 500, color: '#94a3b8' },
          new URL(import.meta.env.SITE).host
        )
      ])
    ]
  )

  const svg = await satori(card as unknown as Parameters<typeof satori>[0], {
    width: 1200,
    height: 630,
    fonts: [
      { name: 'Inter', data: font('inter-latin-400-normal.woff'), weight: 400, style: 'normal' },
      { name: 'Inter', data: font('inter-latin-500-normal.woff'), weight: 500, style: 'normal' },
      { name: 'Inter', data: font('inter-latin-600-normal.woff'), weight: 600, style: 'normal' },
      {
        name: 'Source Serif 4',
        data: font('source-serif-4-latin-600-normal.woff'),
        weight: 600,
        style: 'normal'
      }
    ]
  })

  const png = await sharp(Buffer.from(svg)).png().toBuffer()
  return new Response(new Uint8Array(png), { headers: { 'Content-Type': 'image/png' } })
}
