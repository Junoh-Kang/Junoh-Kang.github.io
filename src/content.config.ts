import { defineCollection } from 'astro:content'
import { glob } from 'astro/loaders'
import { z } from 'astro/zod'

function removeDupsAndLowerCase(array: string[]) {
  if (!array.length) return array
  const lowercaseItems = array.map((str) => str.toLowerCase())
  const distinctItems = new Set(lowercaseItems)
  return Array.from(distinctItems)
}

// Define blog collection
const blog = defineCollection({
  // Load Markdown and MDX files in the `src/content/blog/` directory.
  // One folder per post: blog/YYYY-MM-DD-<slug>/index.md, with its attachments beside it.
  // The date prefix only orders the folders; the post URL is /blog/<slug>.
  loader: glob({
    base: './blog',
    pattern: '*/index.{md,mdx}',
    generateId: ({ entry }) =>
      entry.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\/index\.mdx?$/, '')
  }),
  // Required
  schema: ({ image }) =>
    z.object({
      // Required
      title: z.string().max(120),
      description: z.string().max(1000).default(''),
      publishDate: z.coerce.date(),
      // Optional
      updatedDate: z.coerce.date().optional(),
      heroImage: z
        .object({
          src: image(),
          alt: z.string().optional(),
          inferSize: z.boolean().optional(),
          width: z.number().optional(),
          height: z.number().optional(),
          color: z.string().optional()
        })
        .optional(),
      tags: z.array(z.string()).default([]).transform(removeDupsAndLowerCase),
      language: z.string().optional(),
      draft: z.boolean().default(false),
      pin: z.boolean().optional(),
      // Slide deck file next to index.md (e.g. presentation.pdf); shown above the TOC
      slides: z.string().optional(),
      // Special fields
      comment: z.boolean().default(true)
    })
})

// Define docs collection
const docs = defineCollection({
  loader: glob({ base: './src/content/docs', pattern: '**/*.{md,mdx}' }),
  schema: () =>
    z.object({
      title: z.string().max(60),
      description: z.string().max(160),
      publishDate: z.coerce.date().optional(),
      updatedDate: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]).transform(removeDupsAndLowerCase),
      draft: z.boolean().default(false),
      // Special fields
      order: z.number().default(999)
    })
})

export const collections = { blog, docs }
