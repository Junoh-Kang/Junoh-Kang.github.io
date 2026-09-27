import { readFileSync } from 'node:fs'

import { rehypeHeadingIds } from '@astrojs/markdown-remark'
import AstroPureIntegration from 'astro-pure'
import { defineConfig, fontProviders, svgoOptimizer } from 'astro/config'
import remarkMath from 'remark-math'

// Local integrations
import rehypeAutolinkHeadings from './src/plugins/rehype-auto-link-headings.ts'
import rehypeMathjax4 from './src/plugins/rehype-mathjax4.ts'
import blogAssets from './src/integrations/blog-assets.ts'
// Shiki
import {
  addCollapse,
  addCopyButton,
  addLanguage,
  addTitle,
  updateStyle
} from './src/plugins/shiki-custom-transformers.ts'
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerRemoveNotationEscape
} from './src/plugins/shiki-official/transformers.ts'
import config from './src/site.config.ts'

// Old Jekyll URLs -> new routes. Post URLs come from scripts/migrate_post.py.
const postRedirects: Record<string, string> = JSON.parse(
  readFileSync(new URL('./scripts/redirects.json', import.meta.url), 'utf8')
)
const oldTags = ['agents', 'finance', 'generative', 'llm', 'rl', 'statistics', 'test-time-scaling', 'time-series', 'video']
const oldNews = ['20230622', '20231009', '20240227', '20240521', '20240617', '20250123', '20250124', '20250527', '20250701', '20251127', '20260616', '20260925']
const oldCategories = ['paper-review', 'study-note', 'survey']
const redirects: Record<string, string> = {
  ...postRedirects,
  ...Object.fromEntries(oldTags.map((t) => [`/blog/tag/${t}/`, `/tags/${t}`])),
  ...Object.fromEntries(oldCategories.map((c) => [`/blog/category/${c}/`, `/tags/${c}`])),
  ...Object.fromEntries(['2023', '2024', '2025', '2026'].map((y) => [`/blog/${y}/`, '/archives'])),
  '/blog/page/2/': '/blog/2',
  '/publications': '/#publications',
  ...Object.fromEntries(oldNews.map((d) => [`/news/${d}/`, '/#news'])),
  '/news': '/#news',
  '/cv/': '/assets/pdf/Junoh_Kang_CV.pdf',
  '/repositories/': 'https://github.com/Junoh-Kang'
}

// https://astro.build/config
export default defineConfig({
  // [Basic]
  site: 'https://junoh-kang.github.io',
  // Deploy to a sub path
  // https://astro-pure.js.org/docs/setup/deployment#platform-with-base-path
  // base: '/astro-pure/',
  trailingSlash: 'never',
  redirects,
  // root: './my-project-directory',
  server: { host: true },
  // https://docs.astro.build/en/guides/prefetch/
  prefetch: {
    // prefetchAll: true,
    defaultStrategy: 'viewport'
  },

  // [Adapter]
  // https://docs.astro.build/en/guides/deploy/
  // GitHub Pages serves static files only
  output: 'static',
  // Local (standalone)
  // adapter: node({ mode: 'standalone' }),
  // output: 'server',

  // [Assets]
  image: {
    responsiveStyles: true,
    service: { entrypoint: 'astro/assets/services/sharp' },
    // domains: ['ghchart.rshah.org'],
    remotePatterns: [{ protocol: 'https' }]
  },
  // Enable font preloading and optimization
  // https://docs.astro.build/en/guides/fonts/
  fonts: [
    {
      provider: fontProviders.google(),
      name: 'Inter',
      cssVariable: '--font-inter',
      styles: ['normal', 'italic'],
      weights: [400, 500, 600],
      subsets: ['latin']
    },
    {
      provider: fontProviders.google(),
      name: 'Source Serif 4',
      cssVariable: '--font-source-serif',
      styles: ['normal', 'italic'],
      weights: [400, 600],
      subsets: ['latin']
    }
  ],

  // [Markdown]
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeMathjax4,
      rehypeHeadingIds,
      [
        rehypeAutolinkHeadings,
        {
          behavior: 'append',
          // data-pagefind-ignore keeps the '#' out of search excerpts
          properties: { className: ['anchor'], 'data-pagefind-ignore': '' },
          content: { type: 'text', value: '#' }
        }
      ]
    ],
    // https://docs.astro.build/en/guides/syntax-highlighting/
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark'
      },
      transformers: [
        // Two copies of @shikijs/types (one under node_modules
        // and another nested under @astrojs/markdown-remark → shiki).
        // Official transformers
        // @ts-ignore this happens due to multiple versions of shiki types
        transformerNotationDiff(),
        // @ts-ignore this happens due to multiple versions of shiki types
        transformerNotationHighlight(),
        // @ts-ignore this happens due to multiple versions of shiki types
        transformerRemoveNotationEscape(),
        // Custom transformers
        // @ts-ignore this happens due to multiple versions of shiki types
        updateStyle(),
        // @ts-ignore this happens due to multiple versions of shiki types
        addTitle(),
        // @ts-ignore this happens due to multiple versions of shiki types
        addLanguage(),
        // @ts-ignore this happens due to multiple versions of shiki types
        addCopyButton(2000), // timeout in ms
        // @ts-ignore this happens due to multiple versions of shiki types
        addCollapse(15) // max lines that needs to collapse
      ]
    }
  },

  // [Integrations]
  integrations: [
    // astro-pure will automatically add sitemap, mdx & unocss
    // sitemap(),
    // mdx(),
    AstroPureIntegration(config),
    blogAssets()
  ],

  // [Experimental]
  experimental: {
    // Allow compatible editors to support intellisense features for content collection entries
    // https://docs.astro.build/en/reference/experimental-flags/content-intellisense/
    contentIntellisense: true,
    // Enable SVGO optimization for SVG assets
    // https://docs.astro.build/en/reference/experimental-flags/svg-optimization/
    svgOptimizer: svgoOptimizer(),
    // Enables pre-rendering your prefetched pages on the client in supported browsers.
    // https://docs.astro.build/en/reference/experimental-flags/client-prerender/
    clientPrerender: true,
    // https://docs.astro.build/en/reference/experimental-flags/queued-rendering/
    queuedRendering: {
      enabled: true
    }
  }
})
