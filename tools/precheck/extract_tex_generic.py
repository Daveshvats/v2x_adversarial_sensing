#!/usr/bin/env python3
"""Extract readable plain text from any LaTeX file for similarity analysis.
Usage: python3 extract_tex_generic.py <src.tex> <out.txt>
Same pipeline as extract_tex_text.py (comments, preamble, bibliography,
floats, math stripped; captions and section titles kept).
"""
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"

with open(SRC, errors="replace") as f:
    tex = f.read()

# 1. drop comments
tex = re.sub(r'(?<!\\)%.*', '', tex)

# 2. locate body (after \begin{document}), else keep all
m = re.search(r'\\begin\{document\}', tex)
if m:
    tex = tex[m.end():]

# 3. drop bibliography
for pat in (r'\\bibliography\{', r'\\begin\{thebibliography\}', r'\\bibliographystyle'):
    m = re.search(pat, tex)
    if m:
        tex = tex[:m.start()]

# 4. cite/ref/label/url placeholders
tex = re.sub(r'\\cite[tp]?\*?(\[[^\]]*\])?\{[^}]*\}', '[cite]', tex)
tex = re.sub(r'\\(ref|eqref|autoref)\{[^}]*\}', '[ref]', tex)
tex = re.sub(r'\\label\{[^}]*\}', '', tex)
tex = re.sub(r'\\url\{[^}]*\}', '[url]', tex)
tex = re.sub(r'\\href\{[^}]*\}\{[^}]*\}', '[link]', tex)

# 5. floats out (captions would be kept via step 7 fallback)
tex = re.sub(r'\\begin\{(tabular|table\*|figure\*|verbatim|lstlisting|equation\*?|align\*?|algorithm)\*?\}.*?\\end\{\1\*?\}', ' [float] ', tex, flags=re.S)

# 6. sectioning -> markers
tex = re.sub(r'\\(sub)*section\*?\{([^}]*)\}', r'\n\n## \2\n', tex)
tex = re.sub(r'\\(paragraph|subsubsection)\*?\{([^}]*)\}', r'\n\n\2\n', tex)
tex = re.sub(r'\\title\{([^}]*)\}', r'\n\nTITLE: \1\n', tex)
tex = re.sub(r'\\abstract', '\n\nABSTRACT\n', tex)

# 7. simple one-arg commands unwrapped
for cmd in ['textbf', 'textit', 'emph', 'texttt', 'textsc', 'underline', 'textrm', 'textsf', 'mbox', 'fbox', 'centering', 'caption']:
    tex = re.sub(r'\\' + cmd + r'\{([^{}]*)\}', r'\1', tex)

# 8. inline math
tex = re.sub(r'\$[^$]*\$', ' [math] ', tex)
tex = re.sub(r'\\\(', ' [math] ', tex)
tex = re.sub(r'\\\)', ' [math] ', tex)

# 9. leftovers
tex = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', tex)
tex = re.sub(r'[{}]', '', tex)

# 10. tidy
tex = re.sub(r'[ \t]+', ' ', tex)
tex = re.sub(r'\n\s*\n+', '\n\n', tex)
tex = re.sub(r' ?\n ?', '\n', tex)

text = tex.strip()
with open(OUT, 'w') as f:
    f.write(text)
words = re.findall(r'\b\w+\b', text)
print(f"Extracted {len(words)} words, {len(text)} chars -> {OUT}")
