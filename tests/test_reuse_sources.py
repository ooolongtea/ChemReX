import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from batch.store import TaskStore
from pipeline.document.reuse_sources import load_document_reuse


class ReuseSourcesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'tasks.db'
        self.store = TaskStore(self.db)
        self.addCleanup(self.store.close)
        self.pdf = self.root / 'uploads' / 'patent.pdf'
        self.pdf.parent.mkdir()
        self.pdf.write_bytes(b'%PDF-1.4 original patent')
        self.out = self.root / 'current'
        self.out.mkdir()
        (self.out / 'source_identity.json').write_text(json.dumps({'pdf_path': str(self.pdf)}))
        env = patch.dict(os.environ, {'CHEMREX_REUSE_TASK_DATABASES': str(self.db)})
        env.start()
        self.addCleanup(env.stop)

    def add_source(self, label, content=b'%PDF-1.4 original patent', status='DONE', language='en'):
        pdf = self.root / label / 'patent.pdf'
        pdf.parent.mkdir()
        pdf.write_bytes(content)
        output = pdf.parent / 'result'
        output.mkdir()
        chunk = {'id': 'example_1', 'chunk_fingerprint': 'current-source'}
        (output / 'chunks.jsonl').write_text(json.dumps(chunk) + '\n')
        record = {**chunk, 'status': 'ok', 'reactions': [{'segment_index': 1, 'record_index': 1,
            'text': 'Methanol was stirred for 2 h.', 'extraction': {'language': language,
                'raw': {'ner': '{"SOLVENT":["Methanol"]}', 'action': '["stirred"]'},
                'extraction_configuration': {'model': 'test'}}, 'coref': {'ner': {'SOLVENT': [
                    {'name': 'Methanol', 'standard_name': 'methanol', 'smiles': 'CO',
                     'structure_source': 'name_resolution', 'naming_raw': 'Methanol|methanol'}]}}}]}
        (output / 'results.jsonl').write_text(json.dumps(record) + '\n')
        task_id, _ = self.store.add_task(label, str(pdf), str(output), language)
        with sqlite3.connect(self.db) as conn:
            conn.execute('UPDATE pdf_task SET status=? WHERE id=?', (status, task_id))
        return output

    def test_loads_historical_names_with_provenance_and_current_checks(self):
        source = self.add_source('historical')
        reuse = load_document_reuse(self.out, 'en')
        step = {'text': 'Methanol was stirred for 2 h.', 'local_text': 'Methanol was stirred for 2 h.',
                'procedure_references': []}
        enriched = reuse.enrich_known_names({'SOLVENT': ['Methanol']}, step, 'en', 1)
        compound = enriched['SOLVENT'][0]
        self.assertEqual(compound['smiles'], 'CO')
        self.assertEqual(compound['resolution_status'], 'reused_candidate')
        self.assertEqual(compound['normalization_reuse']['source'], str(source / 'results.jsonl'))
        self.assertTrue(reuse.assess(step, 'en', {'model': 'changed'})['needs_ner'])

    def test_new_api_results_are_visible_without_reimport_or_restart(self):
        self.assertIsNone(load_document_reuse(self.out, 'en'))
        self.add_source('new_api_result')
        # A separate loader/worker instance uses persistent task data directly.
        self.assertTrue(load_document_reuse(self.out, 'en').inputs)

    def test_same_filename_different_bytes_language_and_running_are_excluded(self):
        self.add_source('different', content=b'%PDF-1.4 different patent')
        self.add_source('running', status='RUNNING')
        self.add_source('chinese', language='zh')
        self.assertIsNone(load_document_reuse(self.out, 'en'))

    def test_newest_extraction_is_used_but_old_successful_structure_survives(self):
        self.add_source('first')
        second = self.add_source('second')
        path = second / 'results.jsonl'
        record = json.loads(path.read_text())
        record['reactions'][0]['extraction']['raw']['ner'] = '{"SOLVENT":["Methanol"],"TIME":"2 h"}'
        record['reactions'][0]['coref']['ner']['SOLVENT'][0].update(smiles=None, standard_name='UNKNOWN', naming_raw=None)
        path.write_text(json.dumps(record) + '\n')
        reuse = load_document_reuse(self.out, 'en')
        step = {'text': 'Methanol was stirred for 2 h.', 'local_text': 'Methanol was stirred for 2 h.',
                'procedure_references': []}
        decision = reuse.assess(step, 'en', {'model': 'test'})
        self.assertNotIn('ambiguous_prior_responses', decision['reasons'])
        self.assertEqual(decision['unit']['source'], str(second / 'results.jsonl'))
        enriched = reuse.enrich_known_names({'SOLVENT': ['Methanol']}, step, 'en', 1)
        self.assertEqual(enriched['SOLVENT'][0]['smiles'], 'CO')

    def test_conflicting_historical_structures_are_still_not_reused(self):
        self.add_source('first')
        second = self.add_source('second')
        path = second / 'results.jsonl'
        record = json.loads(path.read_text())
        record['reactions'][0]['coref']['ner']['SOLVENT'][0]['smiles'] = 'CCO'
        path.write_text(json.dumps(record) + '\n')
        reuse = load_document_reuse(self.out, 'en')
        step = {'text': 'Methanol was stirred for 2 h.', 'local_text': 'Methanol was stirred for 2 h.',
                'procedure_references': []}
        enriched = reuse.enrich_known_names({'SOLVENT': ['Methanol']}, step, 'en', 1)
        self.assertEqual(enriched['SOLVENT'], ['Methanol'])

    def test_missing_or_bad_history_falls_back_to_fresh_work(self):
        self.add_source('bad')
        (self.root / 'bad/result/chunks.jsonl').write_text('invalid json')
        self.assertIsNone(load_document_reuse(self.out, 'en'))


if __name__ == '__main__':
    unittest.main()
