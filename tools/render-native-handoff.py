#!/usr/bin/env python3
"""Publish the authoritative fork handoff as readable HTML and Markdown."""
import hashlib
import html
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent.parent
source = (root / 'docs/NATIVE-FORK-HANDOFF.md').read_text()
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)', r'<a href="\2">\1</a>', text)


blocks = []
paragraph = []
ordered = False


def flush():
    if paragraph:
        tag = 'li' if ordered else 'p'
        blocks.append(f'<{tag}>' + inline(' '.join(paragraph)) + f'</{tag}>')
        paragraph.clear()


for line in source.splitlines() + ['']:
    heading = re.match(r'^(#{1,3}) (.*)', line)
    item = re.match(r'^\d+\. (.*)', line)
    if not line.strip() or heading:
        flush()
        if ordered:
            blocks.append('</ol>')
            ordered = False
        if heading:
            level = len(heading[1])
            blocks.append(f'<h{level}>' + inline(heading[2]) + f'</h{level}>')
    elif item:
        flush()
        if not ordered:
            blocks.append('<ol>')
            ordered = True
        paragraph.append(item[1])
    else:
        paragraph.append(line.strip())

digest = hashlib.sha256(source.encode()).hexdigest()
page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NC2000 Native Fork Handoff</title>
<style>body{max-width:900px;margin:auto;padding:24px;background:#f6f3ec;color:#282b27;font:16px/1.75 system-ui,sans-serif}a{color:#385c42}h1{font-size:1.7rem}h2{font-size:1.25rem;margin-top:2em}code{font-size:.9em;overflow-wrap:anywhere}li{margin-bottom:.7em}</style>
<nav><a href="native-fork-handoff.md">Markdown for maintainers and AI agents</a> · <a href="preview-cache.html">Preview cache API</a> · <a href="index.html">NC2000</a></nav>
''' + '\n'.join(blocks) + f'\n<!-- source-sha256: {digest} -->\n</html>\n'
(destination / 'native-fork-handoff.md').write_text(source)
(destination / 'native-fork-handoff.html').write_text(page)
print(destination / 'native-fork-handoff.html')
