// Build a navigable HTML reading draft and rasterize code-native SVGs for visual QA.
// No browser session, experiment, training process, or GPU is used.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const root=path.resolve(import.meta.dirname,'..');
const bundle=process.env.CODEX_DEPENDENCY_NODE_MODULES || 'C:/Users/leoni/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require=createRequire(path.join(bundle,'_resolver.cjs'));
const {marked}=await import(pathToFileURL(require.resolve('marked')).href);
const sharp=require('sharp');
const katexPath=process.env.DISSERTATION_KATEX_ROOT;
const vendor=path.join(root,'docs/dissertation/refactor_v2/vendor/katex.min.js');
const katex=katexPath ? require(path.join(katexPath,'dist/katex.js')) : fs.existsSync(vendor) ? require(vendor) : null;
const mathErrors=[];
let mathCount=0;
const man=path.join(root,'docs/dissertation/condensed_translated_humanized_ukrainian');
const edit=path.join(root,'docs/dissertation/refactor_v2');
const figs=path.join(root,'figures/dissertation_v2');
const preview=path.join(edit,'preview');
fs.mkdirSync(preview,{recursive:true});
const names=fs.readdirSync(man).filter(n=>/^0[0-8]_.*\.md$/.test(n)).sort();
const sections=[];
for(const name of names){
  let md=fs.readFileSync(path.join(man,name),'utf8');
  md=md.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,(_,alt,p)=>`![${alt}](${path.relative(preview,path.resolve(man,p)).replaceAll('\\','/')})`);
  md=md.replace(/(?<!!)\[([^\]]+)\]\((0[0-8]_[^#)]+\.md)(#[^)]+)?\)/g,(_,label,file,anchor)=>`[${label}](${anchor || '#'+file})`);
  const maths=[];
  const render=(math,display)=>{
    if(!katex) return display ? `<pre>${math.replaceAll('<','&lt;')}</pre>` : `<code>${math.replaceAll('<','&lt;')}</code>`;
    try {mathCount++; return katex.renderToString(math,{displayMode:display,output:'mathml',throwOnError:true,strict:'ignore',trust:false,macros:{'\\bm':'\\boldsymbol{#1}'}});}
    catch(error){mathErrors.push({file:name,math,error:error.message});return `<pre class="math-error">${math.replaceAll('<','&lt;')}</pre>`;}
  };
  md=md.replace(/\$\$([\s\S]*?)\$\$/g,(_,math)=>{const token=`MATHBLOCKTOKEN${maths.length}END`;maths.push([token,render(math.trim(),true)]);return token;});
  md=md.replace(/(?<!\$)\$(?!\$)([^\n$]+)\$(?!\$)/g,(_,math)=>{const token=`MATHINLINETOKEN${maths.length}END`;maths.push([token,render(math,false)]);return token;});
  let rendered=marked.parse(md);
  for(const [token,html] of maths) rendered=rendered.replace(token,html);
  sections.push(`<section id="${name}">${rendered}</section>`);
}
const nav=names.map(n=>`<a href="#${n}">${n}</a>`).join(' · ');
const style=`body{font:18px/1.6 Georgia,serif;color:#202b36;margin:0 auto;max-width:1080px;padding:30px}nav{font:13px Arial;background:#f4f6f9;padding:15px;position:sticky;top:0}h1,h2,h3{font-family:Arial;line-height:1.25}h1{margin-top:70px}h2{margin-top:45px}img{display:block;max-width:100%;height:auto;margin:25px auto}table{border-collapse:collapse;display:block;overflow:auto;font:14px/1.5 Arial;margin:25px 0}th,td{border:1px solid #c6cdd6;padding:9px}th{background:#f0f3f6}blockquote{border-left:4px solid #ca771d;padding:10px 20px;background:#fff7e9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f8;padding:16px;font:14px/1.5 monospace}a{color:#226ba3}section{border-top:1px solid #bbc4ce} @media print{nav{position:static}h1{break-before:page}img,table{break-inside:avoid}}`;
fs.writeFileSync(path.join(preview,'dissertation_reading_draft.html'),`<!doctype html><html lang="en"><meta charset="utf-8"><title>Dissertation refactor — evidence-gated reading draft</title><style>${style} math[display="block"]{display:block;max-width:100%;overflow-x:auto;margin:22px 0}.math-error{background:#ffdddd}</style><body><h1>Dissertation reading draft</h1><p>English changes on the preserved Ukrainian foundation. Pending evidence is visible. ${katex?'Mathematics is typeset as native MathML using pinned KaTeX 0.16.22.':'Mathematics is source-form; set DISSERTATION_KATEX_ROOT for typesetting.'} This is a complete reading draft, not institutional Word/PDF pagination.</p><nav>${nav}</nav>${sections.join('\n')}</body></html>`);
fs.writeFileSync(path.join(edit,'registries/math_render.json'),JSON.stringify({engine:katex?'KaTeX 0.16.22 MathML':'source fallback',mathCount,errors:mathErrors},null,2)+'\n');
const gallery=[];
for(const name of fs.readdirSync(figs).filter(n=>n.endsWith('.svg'))){
  const output=path.join(preview,name.replace('.svg','.png'));
  await sharp(path.join(figs,name)).png().toFile(output);
  gallery.push(`<h2>${name}</h2><img src="${path.basename(output)}" alt="${name}">`);
}
fs.writeFileSync(path.join(preview,'figure_gallery.html'),`<!doctype html><html><meta charset="utf-8"><title>Figure QA gallery</title><style>${style}</style><body>${gallery.join('\n')}</body></html>`);
console.log(JSON.stringify({preview:path.join(preview,'dissertation_reading_draft.html'),rendered_figures:gallery.length}));
