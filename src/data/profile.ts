// Homepage data that is not part of the CV. CV sections come from ./cv.ts.
// News source (old Jekyll site): _news/*.md.

export const profile = {
  name: 'Junoh Kang',
  role: 'Ph.D. student, Seoul National University',
  pitch: 'Generative models for images, video, and financial markets',
  status: 'Open to quant research roles',
  email: 'junoh.kang@snu.ac.kr',
  links: [
    { label: 'Email', icon: 'email', href: 'mailto:junoh.kang@snu.ac.kr' },
    {
      label: 'Scholar',
      icon: 'link',
      href: 'https://scholar.google.com/citations?user=TLGqhucAAAAJ&hl'
    },
    { label: 'alphaXiv', icon: 'link', href: 'https://www.alphaxiv.org/@junoh-kang' },
    { label: 'GitHub', icon: 'github', href: 'https://github.com/Junoh-Kang' },
    { label: 'LinkedIn', icon: 'link', href: 'https://www.linkedin.com/in/junohkang' },
    { label: 'X', icon: 'x', href: 'https://x.com/junoh__kang' }
  ],
  cv: '/assets/pdf/Junoh_Kang_CV.pdf'
}

// Award badges shown next to a paper's venue in the home publication list (ids from cv.json).
export const pubBadges: Record<string, string> = {
  kim2024fifo: 'Gold medal, Samsung Humantech Paper Award'
}

// Home "Research Interests": current directions, each backed by papers (ids from cv.json).
export const researchInterests: {
  title: string
  papers: { id: string; note: string }[]
}[] = [
  {
    title: 'Generative modeling for financial markets',
    papers: [
      {
        id: 'kang2025probsiginv',
        note: 'a probabilistic reframing of truncated signature inversion that quantifies its inherent ambiguity'
      },
      {
        id: 'kang2026relobgen',

        note: 'an LOB message generator whose messages are replayable by construction'
      }
    ]
  },
  {
    title: 'Efficient sampling for diffusion models',
    papers: [
      {
        id: 'kang2024ogdm',

        note: 'a training method for few-step diffusion sampling'
      },
      {
        id: 'choi2025rx',

        note: 'a few-step diffusion sampler inspired by Richardson extrapolation'
      }
    ]
  }
]

// Unused on the home page; kept in case a paper-grouped research overview comes back.
// Draft grouping for the research overview; titles and grouping need the author's review.
export const researchThemes: { title: string; draft?: boolean; note?: string; papers: string[] }[] =
  [
    {
      title: 'Generative models for financial time series',
      draft: true,
      papers: ['kang2026relobgen', 'kang2025probsiginv']
    },
    {
      title: 'Fast and accurate diffusion sampling',
      draft: true,
      papers: ['kang2024ogdm', 'choi2025rx']
    },
    {
      title: 'Video generation, editing, and restoration',
      draft: true,
      papers: ['kim2024fifo', 'lee2025strmatch', 'kang2025icmsr']
    }
  ]

// Newest first. HTML is trusted, copied from _news/*.md.
export const news = [
  {
    date: '2026-09-25',
    html: 'Our paper, <a href="https://arxiv.org/abs/2606.15332">Probabilistic Signature Inversion</a>, has been accepted to NeurIPS 2026!'
  },
  {
    date: '2026-06-16',
    html: 'Our paper, <a href="https://arxiv.org/abs/2606.15332">Probabilistic Signature Inversion: Learning Conditional Distributions from Truncated Signatures</a>, is now available on arXiv.'
  },
  {
    date: '2025-11-27',
    html: 'Our paper, <a href="https://arxiv.org/abs/2511.22048">ICM-SR: Image-Conditioned Manifold Regularization for Image Super-Resolution</a>, is archived.'
  },
  {
    date: '2025-07-01',
    html: 'Our paper, <a href="https://arxiv.org/pdf/2506.22868">STR-Match: Matching SpatioTemporal Relevence Score for Training-Free Video Editing</a>, is archived.'
  },
  {
    date: '2025-05-27',
    html: 'We release a new project page for flexible video editing technique. <a href="https://jslee525.github.io/str-match/">STR-Match: Matching SpatioTemporal Relevence Score for Training-Free Video Editing</a> can even change cat into dragon, basketball, ... <em>etc</em>,.!! Feel free to explore.'
  },
  {
    date: '2025-01-24',
    html: 'Our paper, <a href="https://arxiv.org/abs/2405.11473">FIFO-Diffusion: Generating Infinite Videos from Text without Training</a>, awards gold medal on 30th <a href="https://humantech.samsung.com/saitext/board.do">Samsung Humantech paper award</a>.'
  },
  {
    date: '2025-01-23',
    html: 'Our paper, <a href="https://openreview.net/forum?id=rCGleSgNBK">Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions</a>, is accepted to ICLR 2025.'
  },
  {
    date: '2024-06-17',
    html: 'I join Video AI lab at <a href="https://research.adobe.com/">Adobe Research</a> as a summer intern.'
  },
  {
    date: '2024-05-21',
    html: 'Our paper, <a href="https://arxiv.org/abs/2405.11473">FIFO-Diffusion: Generating Infinite Videos from Text without Training</a>, is archived.'
  },
  {
    date: '2024-02-27',
    html: 'Our paper, <a href="https://arxiv.org/abs/2310.04041">Observation-Guided Diffusion Probabilistic Models</a>, is accepted to CVPR 2024.'
  },
  {
    date: '2023-10-09',
    html: 'Our paper, <a href="https://arxiv.org/abs/2310.04041">Observation-Guided Diffusion Probabilistic Models</a>, is archived.'
  },
  {
    date: '2023-06-22',
    html: 'Our paper, <a href="https://arxiv.org/abs/2106.15278">Open-Set Representation Learning Through Combinatorial Embedding</a>, is presented at CVPR 2023.'
  }
]
