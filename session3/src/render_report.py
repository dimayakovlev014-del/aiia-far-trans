"""Standalone stdlib Markdown renderer adapted from the preserved Session 1 renderer."""
import hashlib
import html
import json
import re
from pathlib import Path

from common import ROOT,dump


def inline(source):
    slots=[]
    def reserve(s):
        slots.append(s)
        return f'\x00{len(slots)-1}\x00'
    source=re.sub(r'`([^`]+)`',lambda m:reserve('<code>'+html.escape(m[1])+'</code>'),source)
    source=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:reserve('<a href="'+html.escape(m[2],quote=True)+'">'+html.escape(m[1])+'</a>'),source)
    source=html.escape(source)
    source=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',source)
    source=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',source)
    return re.sub(r'\x00(\d+)\x00',lambda m:slots[int(m[1])],source)


def render(source):
    lines=source.splitlines()
    output=['<section class="report-page" data-part="body">']
    index=0
    while index<len(lines):
        line=lines[index]
        if not line.strip():
            index+=1
        elif line in ('<!-- pagebreak -->','<!-- references -->'):
            output.append('</section><section class="report-page" data-part="'+('references' if 'references' in line else 'body')+'">')
            index+=1
        elif re.match(r'^#{1,3} ',line):
            level=len(line.split(' ',1)[0])
            output.append(f'<h{level}>'+inline(line[level+1:])+f'</h{level}>')
            index+=1
        elif line.startswith('|'):
            rows=[]
            while index<len(lines) and lines[index].startswith('|'):
                cells=[c.strip() for c in lines[index].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):
                    rows.append(cells)
                index+=1
            assert all(len(r)==len(rows[0]) for r in rows)
            output.append('<table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>')
            for row in rows[1:]:
                output.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
            output.append('</tbody></table>')
        else:
            block=[]
            while index<len(lines) and lines[index].strip() and not lines[index].startswith(('#','|','<!--')):
                block.append(lines[index])
                index+=1
            if not block:
                raise ValueError(line)
            output.append('<p>'+inline(' '.join(block))+'</p>')
    output.append('</section>')
    return '\n'.join(output)


CSS='''
@page { size:A4; margin:16mm 16mm 15mm; }
* { box-sizing:border-box; }
body { font-family:"Times New Roman",serif; color:#161616; font-size:10pt; line-height:1.18; margin:0; }
h1 { font-size:17pt; text-align:center; line-height:1.15; margin:0 0 8pt; }
h2 { font-size:12.5pt; margin:10pt 0 5pt; border-bottom:1px solid #aaa; padding-bottom:2pt; }
h3 { font-size:11pt; margin:8pt 0 4pt; }
p { text-align:justify; margin:0 0 6pt; orphans:2; widows:2; }
table { width:100%; border-collapse:collapse; font-size:9pt; line-height:1.18; margin:6pt 0 8pt; }
td,th { border:1px solid #aaa; padding:4pt; vertical-align:top; overflow-wrap:anywhere; }
th { background:#eee; text-align:left; }
tr { break-inside:avoid; }
code { font-family:Menlo,monospace; font-size:8.4pt; overflow-wrap:anywhere; }
a { color:inherit; overflow-wrap:anywhere; }
h2,h3 { break-after:avoid; }
.report-page { break-after:page; }
.report-page:last-child { break-after:auto; }
[data-part=references] p { text-align:left; font-size:10pt; }
@media screen { body { width:178mm; margin:15mm auto; } .report-page { margin-bottom:20mm; } }
'''


def main():
    folder=ROOT/'session3/report'
    source=(folder/'report.md').read_text()
    body=render(source)
    html_text='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>HW9 — Dmitry Yakovlev — DRAFT</title><style>'+CSS+'</style></head><body>'+body+'</body></html>'
    (folder/'report.html').write_text(html_text)
    dump(folder/'render_manifest.json',dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
         html_sha256=hashlib.sha256(html_text.encode()).hexdigest(),
         intended_body_pages=body.count('data-part="body"'),references_pages=body.count('data-part="references"'),
         renderer='Reused pinned HW8 Markdown renderer; paths/title adapted for HW9'))
    print('Rendered session3/report/report.html')


if __name__=='__main__':
    main()
