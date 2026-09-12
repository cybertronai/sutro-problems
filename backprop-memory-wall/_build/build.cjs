// Format the supplied report; never execute its companion calculation script.
const fs = require('node:fs');
const path = require('node:path');
const MarkdownIt = require('markdown-it');
const katex = require('katex');
const texmath = require('markdown-it-texmath');
const root = path.resolve(__dirname, '..');
const original = fs.readFileSync(path.join(__dirname, 'backprop_memory_wall.md'), 'utf8');
const source = original.replace(/^---\n[\s\S]*?\n---\n/, '').trim();
const md = new MarkdownIt({ html: false, typographer: false }).use(texmath, {
  engine: katex, delimiters: 'dollars', katexOptions: { throwOnError: true, trust: false, output: 'htmlAndMathml' }
});
const esc = md.utils.escapeHtml;
let inlineCount = 0;
let displayCount = 0;
function renderMath(content, displayMode) {
  return katex.renderToString(content, { displayMode, throwOnError: true, trust: false, output: 'htmlAndMathml', strict: 'error' });
}
md.renderer.rules.math_inline = (tokens, i) => {
  inlineCount++;
  return `<span class="math-inline">${renderMath(tokens[i].content, false)}</span>`;
};
md.renderer.rules.math_block = (tokens, i) => {
  displayCount++;
  return `<div class="equation-block"><p class="scroll-hint">Swipe or scroll to read the equation <span aria-hidden="true">↔</span></p><div class="equation" tabindex="0" role="region" aria-label="Equation ${displayCount}; scroll horizontally if needed">${renderMath(tokens[i].content, true)}</div></div>\n`;
};
const sections = [];
let tableCount = 0;
md.renderer.rules.heading_open = (tokens, i) => {
  const title = tokens[i + 1].content;
  const id = /^\d+\./.test(title) ? `section-${title.match(/^\d+/)[0]}` : 'appendix';
  sections.push({ id, title });
  return `${sections.length > 1 ? '</section>\n' : ''}<section aria-labelledby="${id}"><h2 id="${id}">`;
};
md.renderer.rules.heading_close = () => '</h2>\n';
md.renderer.rules.table_open = () => {
  tableCount++;
  return `<div class="table-block"><p class="scroll-hint">Swipe or scroll to explore the table <span aria-hidden="true">↔</span></p><div class="table-scroll" tabindex="0" role="region" aria-label="Data table ${tableCount}; scroll horizontally"><table>`;
};
md.renderer.rules.table_close = () => '</table></div></div>\n';
md.renderer.rules.th_open = (tokens, i, options, env, renderer) => {
  tokens[i].attrSet('scope', 'col');
  return renderer.renderToken(tokens, i, options);
};
md.renderer.rules.code_inline = (tokens, i) => {
  const text = tokens[i].content;
  const code = `<code>${esc(text)}</code>`;
  return ['backprop_memory_wall.py', 'fig_backprop_memory_wall.png'].includes(text) ? `<a href="${text}">${code}</a>` : code;
};
// The source figure is displayed through three equal CSS windows on mobile.
// Its pixels and the downloadable original remain unchanged.
const parsed = md.parse(source, {});
for (let i = 0; i < parsed.length; i++) {
  const token = parsed[i];
  if (token.type !== 'inline' || token.children?.length !== 1 || token.children[0].type !== 'image') continue;
  const caption = token.children[0].content;
  const descriptions = [
    'Panel (a): training-step HBM intensity versus tokens per weight visit, with A100 and H100 balance lines.',
    'Panel (b): minimum tokens per weight visit versus hidden size for A100, H100, and B200.',
    'Panel (c): energy shares for on-chip dynamic power, HBM weights and gradients, HBM activations, and static power.'
  ];
  const panelTitles = ['(a) Training-step intensity vs. batch', '(b) Batch needed to leave the HBM wall', '(c) Where the joules go'];
  const panels = descriptions.map((alt, n) => `<a class="figure-panel panel-${n}" href="fig_backprop_memory_wall.png" aria-label="${esc(alt)} Open full-resolution figure."><span class="panel-title">${panelTitles[n]}</span><span class="figure-crop"><img src="fig_backprop_memory_wall.png" width="2805" height="833" loading="lazy" decoding="async" alt="${esc(alt)}"></span></a>`).join('');
  const html = `<figure id="memory-wall-figure"><div class="figure-panels">${panels}</div><figcaption>${md.renderInline(caption)} <a class="figure-link" href="fig_backprop_memory_wall.png">Open full-resolution figure <span aria-hidden="true">↗</span></a></figcaption></figure>`;
  parsed.splice(i - 1, 3, { type: 'html_block', content: html, block: true });
  i--;
}
// html_block here comes only from our figure template; source HTML is disabled.
let article = md.renderer.render(parsed, md.options, {}) + '</section>';
article = article.replace('<ol>', '<ol class="heuristics">');
const toc = sections.map(({ id, title }) => {
  const match = title.match(/^(\d+)\. (.*)$/);
  return `<li><a href="#${id}"><span class="toc-number">${match ? match[1].padStart(2, '0') : 'A'}</span><span>${esc(match ? match[2] : 'Parameters & provenance')}</span></a></li>`;
}).join('\n');
const template = fs.readFileSync(path.join(__dirname, 'template.html'), 'utf8');
const words = source.split(/\s+/).length;
const result = template.replace('<!-- CONTENTS -->', toc).replace('<!-- ARTICLE -->', article).replace('<!-- READING_TIME -->', `${Math.ceil(words / 210)} min read`);
fs.writeFileSync(path.join(root, 'index.html'), result);
const katexRoot = path.dirname(require.resolve('katex/package.json'));
const vendor = path.join(root, 'vendor', 'katex');
fs.mkdirSync(path.join(vendor, 'fonts'), { recursive: true });
let css = fs.readFileSync(path.join(katexRoot, 'dist/katex.min.css'), 'utf8');
// WOFF2 is sufficient for the browsers this responsive page targets.
css = css.replace(/,url\([^)]*\.woff\) format\("woff"\),url\([^)]*\.ttf\) format\("truetype"\)/g, '');
fs.writeFileSync(path.join(vendor, 'katex.min.css'), css);
for (const file of fs.readdirSync(path.join(katexRoot, 'dist/fonts')).filter(f => f.endsWith('.woff2'))) {
  fs.copyFileSync(path.join(katexRoot, 'dist/fonts', file), path.join(vendor, 'fonts', file));
}
fs.copyFileSync(path.join(katexRoot, 'LICENSE'), path.join(vendor, 'LICENSE'));
if (sections.length !== 10 || tableCount !== 9 || displayCount !== 6 || /katex-error/.test(result)) {
  throw new Error(`Unexpected report structure: ${sections.length} sections, ${tableCount} tables, ${displayCount} equations`);
}
console.log(`Built ${sections.length} sections, ${tableCount} tables, ${displayCount} display equations, ${inlineCount} inline equations.`);
