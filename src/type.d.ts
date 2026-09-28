declare module 'virtual:config' {
  const Config: import('astro-pure/types').ConfigOutput
  export default Config
}

declare module '@mathjax/src' {
  const MathJax: { init: (config: object) => Promise<unknown> }
  export default MathJax
}
