// Build-time MathJax 4 (SVG output, New Computer Modern font) for remark-math nodes.
// Equation numbers and \label/\eqref are scoped to one page (texReset per file).
import MathJaxInit from '@mathjax/src'
import type { Element, Root } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { visit } from 'unist-util-visit'

type MathJaxApi = {
  tex2svgPromise: (tex: string, options: { display: boolean }) => Promise<unknown>
  texReset: () => void
  svgStylesheet: () => unknown
  startup: {
    adaptor: { outerHTML: (node: unknown) => string; textContent: (node: unknown) => string }
    document: { outputJax: { font: { loadDynamicFiles: () => Promise<void> } } }
  }
}

let mathjaxPromise: Promise<MathJaxApi> | undefined

function getMathJax(): Promise<MathJaxApi> {
  mathjaxPromise ??= (async () => {
    const MathJax = (await MathJaxInit.init({
      loader: { load: ['input/tex', 'output/svg'] },
      tex: { tags: 'ams', inlineMath: [], displayMath: [] },
      svg: { fontCache: 'local' },
      output: { font: 'mathjax-newcm' }
    })) as MathJaxApi
    // Preload every glyph range so a render never retries (retries re-register \label).
    await MathJax.startup.document.outputJax.font.loadDynamicFiles()
    return MathJax
  })()
  return mathjaxPromise
}

function classList(node: Element): string[] {
  const cls = node.properties?.className as unknown
  return Array.isArray(cls) ? cls.map(String) : typeof cls === 'string' ? cls.split(' ') : []
}

function textOf(node: Element): string {
  let out = ''
  visit(node, 'text', (t: { value: string }) => {
    out += t.value
  })
  return out
}

export default function rehypeMathjax4() {
  return async (tree: Root, file: { path?: string }) => {
    const targets: { node: Element; parent: Element | Root; index: number; display: boolean }[] =
      []

    visit(tree, 'element', (node: Element, index, parent) => {
      if (parent == null || index == null) return
      const cls = classList(node)
      if (node.tagName === 'pre') {
        const code = node.children.find(
          (c): c is Element => c.type === 'element' && c.tagName === 'code'
        )
        if (code && classList(code).includes('math-display')) {
          targets.push({ node: code, parent: parent as Element, index, display: true })
          return 'skip'
        }
      }
      if (node.tagName === 'code' && cls.includes('math-inline')) {
        targets.push({ node, parent: parent as Element, index, display: false })
        return 'skip'
      }
    })

    if (!targets.length) return

    const MathJax = await getMathJax()
    MathJax.texReset()

    // Render in document order so equation numbers follow the page.
    const rendered: Element[] = []
    for (const t of targets) {
      const tex = textOf(t.node).trim()
      const out = MathJax.startup.adaptor.outerHTML(
        await MathJax.tex2svgPromise(tex, { display: t.display })
      )
      const err = /data-mjx-error="([^"]*)"/.exec(out)
      if (err) console.warn(`[mathjax] ${file.path ?? ''}: ${err[1]} in: ${tex.slice(0, 80)}`)
      const frag = fromHtml(out, { fragment: true }).children[0] as Element
      rendered.push(
        t.display
          ? {
              type: 'element',
              tagName: 'div',
              properties: { className: ['math-display'] },
              children: [frag]
            }
          : frag
      )
    }

    // Replace from the back so earlier indices stay valid within a parent.
    for (let i = targets.length - 1; i >= 0; i--) {
      const t = targets[i]
      t.parent.children.splice(t.index, 1, rendered[i])
    }

    const css = MathJax.startup.adaptor.textContent(MathJax.svgStylesheet())
    tree.children.push({
      type: 'element',
      tagName: 'style',
      properties: {},
      children: [{ type: 'text', value: css }]
    })
  }
}
