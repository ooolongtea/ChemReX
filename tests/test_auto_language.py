"""Auto language selection across real PDF text, parsed scans, queue and HTTP."""
import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from batch.api import ApiConfig, create_app
from batch.language import detect_text_language, detect_pdf_language
from batch.scheduler import BatchScheduler
from batch.store import TaskStore, _SCHEMA

ZH = '实施例一：将起始原料加入反应瓶，加入溶剂后加热搅拌三小时，冷却过滤得到目标化合物。'
EN = 'Example one: The starting material was dissolved in ethanol and heated for three hours to give the title compound. '


def pdf(path, language=None):
    document = canvas.Canvas(str(path))
    if language == 'zh':
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        document.setFont('STSong-Light', 10)
        for i in range(5):document.drawString(30, 750-i*20, ZH)
    elif language == 'en':
        for i in range(8):document.drawString(30, 750-i*20, EN)
    else:
        # No text layer: the parser must supply the language evidence.
        document.rect(30, 30, 300, 300, fill=1)
    document.save()


@pytest.mark.parametrize('text,language', [
    (ZH*3+'2-(methylamino)-phenyl acetate hydrochloride', 'zh'),
    (EN*3+'作者张三', 'en'),
    ('1234 80℃ 3h 93% CCO>>CC=O', None),
])
def test_prose_language_ignores_short_labels(text, language):
    assert detect_text_language(text) == language


@pytest.mark.parametrize('language', ['zh','en',None])
def test_real_pdf_text_layer_detection(tmp_path, language):
    path = tmp_path/'sample.pdf'
    pdf(path, language)
    assert detect_pdf_language(str(path)) == language


class Parser:
    def __init__(self, text, pause=None):
        self.text, self.pause, self.languages = text, pause, []
    def wait_healthy(self, attempts=1):return {'status':'ok'}
    def parse(self, pdf_path, output_dir, backend, language='en'):
        self.languages.append(language)
        output = Path(output_dir)
        output.mkdir(parents=True,exist_ok=True)
        content = output/'content.json'
        content.write_text(json.dumps([{'type':'text','text':self.text,'page_idx':0}],ensure_ascii=False))
        (output/'images').mkdir(exist_ok=True)
        if self.pause:self.pause()
        return {'content_list_file':str(content),'image_base':str(output/'images'),'parse_dir':str(output)}


def scheduler(store, parser):
    return BatchScheduler(store,parser,None,task_workers=1,chunk_workers=1,poll_interval=.01,
                          auto_retry=False,no_image=True,no_delivery=True)


@pytest.mark.parametrize('requested,pdf_language,body,resolved,source,parser_language', [
    ('auto','zh',ZH,'zh','pdf_text','zh'),
    ('auto','en',EN,'en','pdf_text','en'),
    ('auto',None,ZH,'zh','parsed_text','zh'),
    ('auto',None,EN,'en','parsed_text','zh'),
    ('auto',None,'1234','en','fallback_en','zh'),
    ('en','zh',ZH,'en','explicit','en'),
    ('zh','en',EN,'zh','explicit','zh'),
])
def test_worker_resolves_language_before_model_routing(tmp_path,requested,pdf_language,body,resolved,source,parser_language):
    from test_batch_scheduler import FakeProcessor
    from batch import config
    path = tmp_path/'sample.pdf'
    pdf(path,pdf_language)
    store = TaskStore(tmp_path/'tasks.db')
    try:
        store.add_task('sample',str(path),str(tmp_path/'out'),requested)
        parser = Parser(body)
        FakeProcessor.calls=[]
        with mock.patch('batch.scheduler.DocumentProcessor',FakeProcessor):
            scheduler(store,parser).run()
        row = store.get('sample')
        assert row['status']=='DONE',row
        assert row['requested_language']==requested
        assert row['language']==resolved and row['language_source']==source
        assert parser.languages==[parser_language]
        assert FakeProcessor.calls[0].language==resolved
        assert FakeProcessor.calls[0].preset==config.BATCH_EXTRACTION_PRESETS[resolved]
    finally:store.close()


def test_scan_pause_and_resume_reuses_parse_before_resolving(tmp_path):
    from test_batch_scheduler import FakeProcessor
    path=tmp_path/'scan.pdf'
    pdf(path)
    store=TaskStore(tmp_path/'tasks.db')
    try:
        store.add_task('scan',str(path),str(tmp_path/'out'),'auto')
        parser=Parser(ZH,pause=lambda:store.request_stop('scan','pause'))
        with mock.patch('batch.scheduler.DocumentProcessor',FakeProcessor):
            scheduler(store,parser).run()
            assert store.get('scan')['status']=='PAUSED'
            assert store.get('scan')['language']=='auto'
            store.resume('scan')
            scheduler(store,parser).run()
        row=store.get('scan')
        assert row['status']=='DONE' and row['language']=='zh'
        assert row['requested_language']=='auto'
        assert row['language_source']=='parsed_text'
        assert len(parser.languages)==1
    finally:store.close()


def test_resolved_auto_upload_remains_idempotent(tmp_path):
    config=ApiConfig(tmp_path,poll_interval=.01)
    app=create_app(config)
    path=tmp_path/'sample.pdf'
    pdf(path,'zh')
    with TestClient(app) as client:
        def upload(language=None):
            return client.post('/v1/pdf?wait=false',files={'file':('sample.pdf',path.read_bytes())},
                headers={'Idempotency-Key':'language-test'},data={} if language is None else {'language':language})
        response=upload()
        assert response.status_code==202
        assert response.json()['requested_language']=='auto'
        assert response.json()['language']=='auto'
        task=app.state.store.claim_next('worker')
        app.state.store.resolve_language(task['id'],'zh','pdf_text','worker')
        app.state.store.mark_done(task['id'],1,run_id='worker')
        for language in [None,'auto']:
            response=upload(language)
            assert response.status_code==200
            assert response.json()['language']=='zh'
            assert response.json()['requested_language']=='auto'
            assert response.json()['language_source']=='pdf_text'
        assert upload('zh').status_code==409
        assert app.state.store.counts()['TOTAL']==1
        schema=client.get('/openapi.json').json()
        form=next(s for s in schema['components']['schemas'].values() if 'file' in s.get('properties',{}) and 'language' in s.get('properties',{}))
        assert form['properties']['language']['default']=='auto'


def test_legacy_queue_language_is_preserved(tmp_path):
    import sqlite3
    path=tmp_path/'legacy.db'
    old=_SCHEMA.replace("    requested_language TEXT NOT NULL DEFAULT 'en',\n",'').replace("    language_source TEXT NOT NULL DEFAULT 'explicit',\n",'')
    with sqlite3.connect(path) as connection:
        connection.executescript(old)
        connection.execute("INSERT INTO pdf_task (uuid,pdf_path,output_dir,language,created_at,updated_at) VALUES ('legacy','old.pdf','out','zh',1,1)")
    store=TaskStore(path)
    try:
        row=store.get('legacy')
        assert row['language']==row['requested_language']=='zh'
        assert row['language_source']=='explicit'
    finally:store.close()


def test_batch_cli_defaults_to_shared_auto_detection():
    from batch.cli import build_parser
    assert build_parser().parse_args(['add','sample.pdf']).language=='auto'


def test_parsed_lists_supply_language_evidence(tmp_path):
    from batch.language import detect_parsed_language
    path=tmp_path/'blocks.json'
    path.write_text(json.dumps([{'type':'list','list_items':[ZH]}],ensure_ascii=False))
    assert detect_parsed_language(str(path))=='zh'


def test_batch_readding_resolved_auto_task_keeps_request_identity(tmp_path, capsys):
    from batch.cli import main
    path=tmp_path/'sample.pdf'
    pdf(path,'zh')
    database=tmp_path/'tasks.db'
    args=['--db',str(database),'add',str(path),'--output-root',str(tmp_path/'out')]
    assert main(args)==0
    store=TaskStore(database)
    try:
        task=store.claim_next('worker')
        store.resolve_language(task['id'],'zh','pdf_text','worker')
        store.mark_done(task['id'],1,run_id='worker')
    finally:store.close()
    capsys.readouterr()
    assert main(args)==0
    assert 'warning:' not in capsys.readouterr().out
