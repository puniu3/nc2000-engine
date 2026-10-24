"""Render the authoritative README cache section for deployment or the shared preview hub."""
from pathlib import Path
import html
import re
import shutil
import sys
import zipfile

repo = Path(__file__).resolve().parent.parent
dest = Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)
text = (repo / 'README.md').read_text().split('## Server-side preview cache interface\n', 1)[1].split('\n## Party evaluator', 1)[0].strip()
inline = lambda s: re.sub(r'`([^`]+)`', r'<code>\1</code>', re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html.escape(s)))
lines = text.splitlines()
blocks = []
i = 0
while i < len(lines):
    if not lines[i].strip():
        i += 1
        continue
    if lines[i].startswith('```'):
        i += 1
        code = []
        while i < len(lines) and not lines[i].startswith('```'):
            code.append(lines[i])
            i += 1
        blocks.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
        i += 1
    elif lines[i].startswith('|'):
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            cells = [c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[- :]+', c) for c in cells):
                tag = 'th' if not rows else 'td'
                rows.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells) + '</tr>')
            i += 1
        blocks.append('<div class="scroll"><table>' + ''.join(rows) + '</table></div>')
    else:
        paragraph = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('```', '|')):
            paragraph.append(lines[i])
            i += 1
        blocks.append('<p>' + inline(' '.join(paragraph)) + '</p>')

page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NC2000 Preview Cache API</title>
<style>body{background:#f6f3ec;color:#282b27;font:16px/1.8 system-ui,sans-serif;max-width:1000px;margin:auto;padding:26px}a{color:#385c42}h1{font-size:1.7rem}pre{padding:18px;background:#eaece2;overflow:auto;font-size:14px;line-height:1.6}code{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #c9c9bd}.scroll{overflow:auto}</style>
<h1>NC2000 Preview Cache API</h1><p lang="ja">フォーク向けの選出キャッシュ生成・読み込みインターフェースです。保存、利用者間共有、同時リクエストの重複計算防止は利用側が実装します。以下はREADMEから生成した仕様書です。</p>
<p><a href="preview-cache.js">Node.js helper</a> · <a href="preview-cache-node.zip">Helper + Node WASM bindings</a> · <a href="preview-cache-doc.txt">Documentation source</a> · <a href="index.html">NC2000</a></p>
''' + '\n'.join(blocks) + '</html>'
(dest / 'preview-cache.html').write_text(page)
(dest / 'preview-cache-doc.txt').write_text('Server-side preview cache interface\n\n' + text + '\n')
shutil.copy2(repo / 'tools/preview-cache.js', dest / 'preview-cache.js')
with zipfile.ZipFile(dest / 'preview-cache-node.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name in ['LICENSE', 'THIRD-PARTY-NOTICES.md']:
        z.write(repo / name, name)
    z.write(repo / 'tools/preview-cache.js', 'tools/preview-cache.js')
    z.write(dest / 'preview-cache-doc.txt', 'README-cache.txt')
    for name in ['nc2000_wasm.js', 'nc2000_wasm_bg.wasm', 'nc2000_wasm.d.ts', 'nc2000_wasm_bg.wasm.d.ts', 'package.json']:
        path = repo / 'crates/wasm/pkg-node' / name
        z.write(path, str(path.relative_to(repo)))
print(dest / 'preview-cache.html')
