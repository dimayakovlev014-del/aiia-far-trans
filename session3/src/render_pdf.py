"""Render the derived HTML locally and check its PDF text/page boundaries."""
import subprocess
from pypdf import PdfReader
from common import ROOT, S3, dump, read, sha


def main():
    source=S3/'report/report.html'
    pdf=S3/'submission/yakovlev_dmitry_session3_report_draft.pdf'
    pdf.parent.mkdir(parents=True,exist_ok=True)
    command=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '--headless=new','--disable-gpu','--no-pdf-header-footer','--no-first-run',
        '--disable-background-networking','--disable-component-update','--disable-sync',
        '--user-data-dir=/private/tmp/aiia_hw9_render_20261126',
        '--print-to-pdf='+str(pdf),source.as_uri()]
    subprocess.run(command,check=True,timeout=60)
    reader=PdfReader(str(pdf))
    pages=[page.extract_text() for page in reader.pages]
    text='\n'.join(pages)
    (S3/'evidence/pdf_text.txt').write_text(text)
    references_page=next(i+1 for i,page in enumerate(pages) if page.lstrip().startswith('References'))
    provenance=read(S3/'report/report_provenance.json')
    normalized=''.join(text.split())
    assert provenance['experiment_commit'] in normalized
    assert 'PENDING' in text
    assert references_page-1<=5, f'Body exceeds five pages: {references_page-1}'
    result=read(S3/'configs/results.json')
    for value in result['metrics'].values():
        for key in ('ndcg_at_10','ndcg_at_10_nonempty'):
            assert f'{value[key]:.4f}' in text
    dump(S3/'evidence/pdf_check.json',dict(pages=len(pages),body_pages=references_page-1,
        references_start_page=references_page,pages_characters=[len(p) for p in pages],
        report_sha256=sha(S3/'report/report.md'),html_sha256=sha(source),pdf_sha256=sha(pdf),
        actual_commit_present=True,all_metric_values_present=True,personal_status='PENDING',
        visual_review='PENDING until raster pages inspected',command=command))
    print('PDF pages:',len(pages),'body pages:',references_page-1)


if __name__=='__main__':
    main()
