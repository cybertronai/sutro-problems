# Building the report

The four supplied report files are preserved byte for byte. The original Markdown
is stored here and published as `../backprop_memory_wall.md.txt`, with a download
filename of `backprop_memory_wall.md`, to avoid Jekyll interpreting its front matter.
The PDF, Python script, and PNG remain in the parent directory.
The companion Python file is a download; this web build does not execute it.

With Node.js 22.12 or newer, run `npm ci --ignore-scripts` and `npm run build`
from this directory. The build converts Markdown using markdown-it and KaTeX,
then writes `../index.html` and the local KaTeX assets. Equations are rendered
as HTML with accessible MathML at build time; the page needs no CDN or math
JavaScript. Jekyll excludes this `_build` directory from the published site.

The chart's three panels are displayed through CSS windows into the original
image and stack vertically on phones. All original text, table values, and
math expressions are retained. `../style.css` and `../reader.js` provide the
responsive layout, contained table/equation scrolling, and chapter navigation.
