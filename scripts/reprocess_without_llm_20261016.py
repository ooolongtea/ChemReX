"""Recompute repair candidates from saved responses with zero new LLM calls.

Writes a separate candidate tree and deferred work list. Production task rows,
old outputs and deliveries are retained until the candidate comparison is ready.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Pin the production MaaS routing before any pipeline import: the .env
# COREF_MODEL default is a stale chatgpt value, while production ran with
# the explicit --llm-model deepseek-v4.1-flash override (batch/cli.py).
# dotenv never overrides variables that already exist, so an explicit
# shell environment still wins over these defaults.
for _name,_value in {'COREF_MODEL':'deepseek-v4.1-flash','COREF_PROVIDER':'maas',
                     'FINAL_PRODUCT_MODEL':'deepseek-v4.1-flash','FINAL_PRODUCT_PROVIDER':'maas',
                     'TMPDIR':'/data/patents/chemrex_batch/tmp',
                     'MAAS_RATE_LIMIT_FILE':'/data/patents/chemrex_batch/tmp/chemrex_maas_rate_%d'%os.getuid()}.items():
    os.environ.setdefault(_name,_value)
from scripts import repair_quality_20261013 as driver
from scripts.audit_llm_reuse_20261016 import configuration
from pipeline.document.extraction_reuse import ExtractionReuse
from pipeline.extraction.parser import parse_json_output
from pipeline.coref.coref_resolver import CorefResolver, _looks_like_identifier
from pipeline.standardize.compound_standardizer import CompoundStandardizer
from pipeline.standardize.standardizer import Standardizer, COMPOUND_FIELDS


def no_new_llm(*args,**kwargs):
    raise RuntimeError('New LLM calls are disabled by user instruction')


def disable_llm_calls():
    from pipeline.gpt_api import GPTAPI
    from pipeline.extraction.inference import InferenceClient
    GPTAPI.chat=no_new_llm
    InferenceClient.chat=no_new_llm


class DeferredReuse(ExtractionReuse):
    def __init__(self, records, source, deferred):
        super().__init__(records,source,allow_unversioned=True)
        self.deferred=deferred

    def extract(self, extractor, step, language):
        decision=self.assess(step,language,extractor.configuration(language))
        unit=decision['unit']
        tasks=[task for task in ('ner','action') if decision['needs_'+task]]
        if tasks:
            self.deferred.append(dict(kind='extraction',step_id=step['step_id'],tasks=tasks,
                                      reasons=decision['reasons']))
        result=dict(language=language,task='both',ner_records=deepcopy(decision['parsed']),
            ner=deepcopy(decision['parsed'][0]) if len(decision['parsed'])==1 else {},
            action=deepcopy(decision['actions']),raw=deepcopy(unit['raw']) if unit else {},
            extraction_configuration=deepcopy(unit.get('configuration')) if unit else None,
            ner_parse_diagnostics=decision['ner_parse_diagnostics'],
            action_parse_details=decision['action_parse_details'],
            parse_status=dict(ner='ok' if decision['parsed'] else 'deferred_no_compatible_ner',
                              action=decision['action_parse_details']['status']),
            deferred_llm=dict(tasks=tasks,reasons=decision['reasons']))
        if unit:
            result['reuse_provenance']=dict(source=unit.get('source') or self.source,
                chunk_id=unit['chunk_id'],chunk_fingerprint=unit['chunk_fingerprint'],
                same_step_input=not unit.get('body_fallback'),ner_reused=True,action_reused=True,
                original_configuration=deepcopy(unit.get('configuration')),
                compatibility_basis='same_procedure_body_and_current_source_checks' if unit.get('body_fallback')
                    else 'exact_input_and_current_source_checks',
                original_prompt_configuration_known=unit.get('configuration') is not None)
        return result


class OfflineExtractor:
    def __init__(self,records,deferred):
        self.deferred=deferred
        self.plans=defaultdict(set)
        self.configs={lang:configuration(lang) for lang in ('en','zh')}
        for record in records:
            parents=[s for s in (record.get('source') or {}).get('steps',[]) if s.get('needs_step_planning')]
            raw={rx.get('step_source',{}).get('step_plan_raw') for rx in record.get('reactions') or []}
            raw.discard(None)
            if len(parents)==1 and len(raw)==1:
                self.plans[(parents[0]['local_text'],parents[0]['title'])].update(raw)

    def configuration(self,language):
        return self.configs[language]

    def plan_steps(self,text,title):
        raw=self.plans.get((text,title),set())
        if len(raw)==1:
            value=next(iter(raw));parsed=parse_json_output(value)
            return dict(raw=value,ranges=parsed if isinstance(parsed,list) else [])
        self.deferred.append(dict(kind='step_plan',title=title,reason='no_unique_saved_plan_for_exact_input'))
        return dict(raw=None,ranges=[])

    extract=no_new_llm


class OfflineNames(CompoundStandardizer):
    def __init__(self):
        from types import SimpleNamespace
        super().__init__(naming_client=SimpleNamespace(chat=no_new_llm))

    def standardize(self,names,prior_naming=None,language=None):
        self.missing_names=set()
        result=super().standardize(names,prior_naming=prior_naming,language=language)
        for name in self.missing_names:
            if not result.get(name,{}).get('smiles'):
                self.deferred.append(dict(kind='naming',name=name,reason='no_unique_saved_naming_response_or_literal_structure'))
        return result

    def _standardize_names(self,names):
        result={}
        for name in names:
            prior=self.saved_names.get(name,{})
            if len(prior)==1:
                result[name]=deepcopy(next(iter(prior.values())))
            elif _looks_like_identifier(name):
                result[name]=dict(standard_name='ID')
            else:
                # The literal name may be resolved by dictionaries/OPSIN. No
                # translated name is invented when a saved response is absent.
                result[name]=dict(standard_name=name if name.isascii() else 'UNKNOWN')
                self.missing_names.add(name)
        return result


def literal_mapping(text_ids, dictionary):
    def label(value):
        value=re.sub(r'^(?:compound|intermediate|化合物|中间体|式)\s*','',value.strip(),flags=re.I)
        return value[1:-1] if value.startswith('(') and value.endswith(')') else value
    mapping={}
    for name in text_ids:
        if name in dictionary:
            mapping[name]=name
        else:
            matches=[key for key in dictionary if label(str(key))==label(name)]
            if len(matches)==1:
                mapping[name]=matches[0]
    return mapping


class OfflineCoref(CorefResolver):
    def _build_llm_mapping(self,text_identifiers, dictionary):
        mapping=literal_mapping(text_identifiers,dictionary)
        missing=[i for i in text_identifiers if i not in mapping]
        if missing:
            self.deferred.append(dict(kind='coreference',identifiers=missing,
                                      reason='no_unique_literal_identifier_match'))
        return mapping


class HybridCoref(OfflineCoref):
    """Literal identifier mapping first; one live LLM call only for the rest.

    Used by the --fill-store integration pass: coreference gaps are filled
    online (they need document-level dictionary context), while extraction,
    naming and step planning come from the shared fill-store.
    """
    def __init__(self,deferred,calls):
        from pipeline.prompts.prompt_loader import PromptLoader
        from pipeline.gpt_api import GPTAPI
        loader=PromptLoader();loader.load_prompt('coref_resolution')
        self.loader=loader;self.prompt_name='coref_resolution'
        self.deferred=deferred;self.llm_calls=calls
        self.api=GPTAPI(model=os.getenv('COREF_MODEL') or None,
                        provider=os.getenv('COREF_PROVIDER') or None)

    def _build_llm_mapping(self,text_identifiers, dictionary):
        mapping=literal_mapping(text_identifiers,dictionary)
        missing=[i for i in text_identifiers if i not in mapping]
        if missing:
            prompt=self.loader.format_prompt(self.prompt_name,
                text_identifiers=json.dumps(sorted(missing),ensure_ascii=False),
                dict_keys=json.dumps(sorted(str(k) for k in dictionary),ensure_ascii=False))
            response=self.api.chat(prompt)
            self.llm_calls['coreference']+=1
            start=response.find('{');end=response.rfind('}')+1
            if 0<=start<end:
                try: data=json.loads(response[start:end])
                except json.JSONDecodeError: data={}
                mapping.update({str(k):str(v) for k,v in (data.get('mapping') or {}).items()
                                if k in text_identifiers and v in dictionary})
        return mapping


def load_fill_store(store):
    """Index every shard store file: extraction raws per (text, language),
    naming batches, step plans by (text, title)."""
    extraction={};naming=[];plans={}
    for pattern in ('extraction.*.jsonl','naming.*.jsonl','step_plan.*.jsonl'):
        for path in sorted(store.glob(pattern)):
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                try: entry=json.loads(line)
                except json.JSONDecodeError:continue
                kind=entry.get('kind')
                if kind=='extraction':
                    slot=extraction.setdefault((entry['text'],entry['language']),
                        {'raws':{},'configuration':entry.get('configuration')})
                    slot['raws'].update(entry.get('raws') or {})
                elif kind=='naming':
                    naming.append(entry)
                elif kind=='step_plan':
                    plans[(entry['text'],entry['title'])]=entry
    return extraction,naming,plans


def overlay_fill_store(reuse,names,extractor,store):
    """Merge fill-store responses into the reuse pool with top priority.

    A merged unit keeps the pool unit's prior compounds so enrich_known_names
    still works, replaces the key's candidates so assess() sees one
    unambiguous response, and carries the stored current configuration so the
    configuration gate accepts it.
    """
    extraction,naming_batches,plans=load_fill_store(store)
    for (text,language),slot in extraction.items():
        units=reuse.inputs.get((text,language))
        base=deepcopy(units[0]) if units else {'raw':{},'configuration':None,'source':None,
            'chunk_id':None,'chunk_fingerprint':None,'reactions':[]}
        base['raw'].update(slot['raws'])
        base['configuration']=slot['configuration']
        base['source']='fill-store'
        base['fill_tasks']=sorted(slot['raws'])
        reuse.inputs[(text,language)]=[base]
    for entry in naming_batches:
        parsed=names._parse_text_response(entry['raw'])
        for name in entry.get('names') or []:
            info=parsed.get(name)
            if info and info.get('standard_name'):
                names.saved_names[name][info['standard_name']]=info
    for (text,title),entry in plans.items():
        extractor.plans[(text,title)]={entry['raw']}
    return len(extraction),len(naming_batches),len(plans)


@lru_cache(maxsize=1)
def local_mapping_scorer():
    try:
        from pipeline.validation.map_score import MappingScorer
        return MappingScorer(),None
    except Exception as exc:
        return None,f'{type(exc).__name__}: {exc}'


class CachedScorer:
    def __init__(self,records,deferred):
        self.cache={}
        self.deferred=deferred
        self.fallback=None
        for record in records:
            for rx in record.get('reactions') or []:
                for candidate in ((rx.get('validation') or {}).get('rxn_validation') or {}).get('candidates',[]):
                    if candidate.get('mapped_rxn') and candidate.get('confidence') is not None:
                        self.cache[candidate['expression']]=candidate

    def score(self,candidates):
        missing=[]
        for candidate in candidates:
            saved=self.cache.get(candidate.expression)
            if saved:
                candidate.mapped_rxn=saved['mapped_rxn'];candidate.confidence=float(saved['confidence'])
            else:
                missing.append(candidate)
        if missing:
            try:
                if self.fallback is None:
                    self.fallback,error=local_mapping_scorer()
                    if self.fallback is None:
                        raise RuntimeError(error)
                self.fallback.score(missing)
            except Exception as exc:
                for candidate in missing:
                    candidate.confidence=0.;candidate.mapped_rxn='';candidate.error='local_mapping_unavailable'
                self.deferred.append(dict(kind='local_mapping',reason=f'{type(exc).__name__}: {exc}'))
        return candidates


def process(task,work,output,supplemental_root=None,fill_store=None,online=True):
    from batch.document_processor import DocumentProcessor
    from pipeline.pipeline import ChemReXPipeline
    from pipeline.validation.validator import ReactionValidator
    deferred=[]
    llm_calls=Counter()
    original=work/'before'/Path(task['db']).stem/task['uuid']
    old_chunks,old=driver.records(original)
    _,recent=driver.records(task['output_dir'])
    supplement=(supplemental_root/Path(task['db']).stem/task['uuid']) if supplemental_root else None
    _,extra=driver.records(supplement) if supplement and supplement.is_dir() else ([],{})
    records=[*old.values(),*extra.values(),*recent.values()]
    names=OfflineNames()
    names.saved_names=defaultdict(dict);names.deferred=deferred
    for record in records:
        for rx in record.get('reactions') or []:
            for field in COMPOUND_FIELDS:
                for compound in ((rx.get('coref') or {}).get('ner') or {}).get(field) or []:
                    if not isinstance(compound,dict):continue
                    name=compound.get('name');parsed=names._parse_text_response(compound.get('naming_raw') or '').get(name)
                    if parsed and parsed['standard_name']==compound.get('standard_name'):
                        names.saved_names[name][parsed['standard_name']]=parsed
    standardizer=object.__new__(Standardizer);standardizer.compound_std=names
    if fill_store is not None and online:
        coref=HybridCoref(deferred,llm_calls)
    else:
        coref=object.__new__(OfflineCoref);coref.deferred=deferred
    coref.auto_standardize=True;coref._standardizer=standardizer
    pipe=ChemReXPipeline(enable_image=False)
    pipe._extraction=OfflineExtractor(records,deferred);pipe._coref=coref
    pipe._validation=ReactionValidator(mapping_scorer=CachedScorer(records,deferred))
    processor=object.__new__(DocumentProcessor)
    processor.language=task['language'];processor.chunk_workers=1
    processor.skip_final_product=not (fill_store is not None and online)
    processor.global_dict_enabled=True;processor.no_delivery=True;processor.pipe=pipe
    target=output/Path(task['db']).stem/task['uuid']
    src=driver.source(task,old_chunks)
    prepared=processor.prepare(src,target)
    reuse=DeferredReuse(list(old.values()),str(original/'results.jsonl'),deferred)
    if extra:
        reuse.overlay_recent(ExtractionReuse(list(extra.values()),str(supplement/'results.jsonl'),allow_unversioned=True))
    from pipeline.document.step_context import CONTEXT_VERSION
    current={key:r for key,r in recent.items() if (r.get('source') or {}).get('context_version')==CONTEXT_VERSION}
    reuse.overlay_recent(ExtractionReuse(list(current.values()),str(Path(task['output_dir'])/'results.jsonl'),allow_unversioned=True))
    prepared.extraction_reuse=reuse
    if fill_store is not None:
        overlay_fill_store(reuse,names,pipe._extraction,fill_store)
    image_path=Path(task['output_dir'])/'image_result.json'
    image=json.loads(image_path.read_text()) if image_path.exists() else {}
    from batch.scheduler import _content_fingerprint
    fingerprint=_content_fingerprint(src)
    if image and image.get('parse_fingerprint') != fingerprint:
        deferred.append(dict(kind='image_cache',reason='cached_image_dictionary_does_not_match_current_parse'))
        image={}
    from pipeline.document.image_reuse import reject_confirmed_candidates,known_image_defects
    image,_=reject_confirmed_candidates(image,known_image_defects(task['uuid']))
    driver.save(target/'image_result.json',image)
    final=Path(task['output_dir'])/'final_product.json'
    from pipeline.document.target_extractor import FINAL_PRODUCT_VERSION
    final_data=json.loads(final.read_text()) if final.exists() else None
    final_cached=isinstance(final_data,dict) and final_data.get('_version')==FINAL_PRODUCT_VERSION
    if final_cached:
        shutil.copy2(final,target/final.name)
    elif fill_store is not None and online:
        pass  # filled online by the processor (cached file written there)
    elif any(c.get('process_kind')=='synthesis' for c in prepared.chunks):
        deferred.append(dict(kind='final_product',reason='no_compatible_saved_document_target'))
    def progress(done,total):
        driver.save(target/'chunk-progress.json',dict(done=done,total=total,new_llm_calls=0,updated_at=time.time()))
    summary=processor.extract(prepared,target,image,retry_errors=True,reextract_reclassified=True,progress_cb=progress)
    if fill_store is not None and online and not final_cached:
        filled=target/'final_product.json'
        if filled.exists():
            data=json.loads(filled.read_text()) if filled.stat().st_size else None
            if isinstance(data,dict) and data.get('_version')==FINAL_PRODUCT_VERSION:
                llm_calls['final_product']+=1
            elif isinstance(data,dict) and data.get('error'):
                llm_calls['final_product']+=1
                deferred.append(dict(kind='final_product',reason='online_target_extraction_failed'))
            elif data is None:
                llm_calls['final_product']+=1  # model answered: no target
    _,current_results=driver.records(target)
    for record in current_results.values():
        if record.get('status')=='error':
            deferred.append(dict(kind='execution_error',record_id=record['id'],reason=record.get('error')))
        for reaction in record.get('reactions') or []:
            if reaction.get('status')=='needs_review':
                deferred.append(dict(kind='quality_review',record_id=record['id'],
                    step_id=reaction.get('step_id'),reaction_index=reaction.get('index'),
                    reasons=(reaction.get('quality') or {}).get('blocking_issues') or []))
    result=dict(uuid=task['uuid'],db=task['db'],output=str(target),new_llm_calls=sum(llm_calls.values()),
                deferred=dict(Counter(d['kind'] for d in deferred)),items=deferred,
                llm_calls=dict(llm_calls) if fill_store is not None else None,
                result_status=summary.get('result_status'),candidate_only=True)
    driver.save(target/'deferred-llm.json',result)
    return result


def main(args):
    if args.fill_store is None:
        disable_llm_calls()
    else:
        # Integration pass: extraction and step planning stay offline
        # (fill-store only, missing responses defer again); GPTAPI stays
        # live for the online coreference and final-product fills unless
        # --no-online requests a pure overlay test.
        from pipeline.extraction.inference import InferenceClient
        InferenceClient.chat=no_new_llm
        if args.no_online:
            from pipeline.gpt_api import GPTAPI
            GPTAPI.chat=no_new_llm
    os.environ.update(STANDARDIZE_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      CHEMREX_LEGACY_CODE_REVISION='66523a5',CHEMREX_REUSE_SOURCE_CHECKED_LEGACY='1',
                      CHEMREX_SOURCE_DEFECTS_FILE=str(ROOT/'doc/reaction-quality-repair-20261013/confirmed-source-defects.json'))
    tasks=json.loads((args.work_dir/'affected.json').read_text())['tasks']
    if args.uuids_file:
        wanted={line.strip() for line in args.uuids_file.read_text().splitlines() if line.strip()}
        tasks=[t for t in tasks if t['uuid'] in wanted]
    if args.uuids:tasks=[t for t in tasks if t['uuid'] in args.uuids]
    if args.limit:tasks=tasks[:args.limit]
    tasks=tasks[args.shard_index::args.shards]
    counts,errors=Counter(),[]
    started=time.time()
    args.output.mkdir(parents=True,exist_ok=True)
    driver.save(args.output/'status.json',dict(phase='running',scope=len(tasks),scanned=0,
                counts={},errors=[],new_llm_calls=0,started_at=started,
                fill_store=str(args.fill_store) if args.fill_store else None,
                online=(args.fill_store is not None and not args.no_online)))
    with (args.output/'documents.jsonl').open('a') as stream:
        for n,task in enumerate(tasks,1):
            try:
                result=process(task,args.work_dir,args.output,args.supplemental_root,
                               args.fill_store,not args.no_online)
                counts['documents']+=1
                counts.update({'deferred_'+k:v for k,v in result['deferred'].items()})
                counts['documents_with_deferred_llm']+=any(k in result['deferred'] for k in
                    ('extraction','naming','coreference','step_plan','final_product'))
                counts['new_llm_calls']+=result['new_llm_calls']
                for k,v in (result.get('llm_calls') or {}).items():
                    counts['llm_calls_'+k]+=v
                stream.write(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False)+'\n');stream.flush()
            except Exception as exc:
                errors.append(dict(uuid=task['uuid'],error=f'{type(exc).__name__}: {exc}'))
                import traceback
                traceback.print_exc()
            progress=dict(phase='running' if n<len(tasks) else 'complete',scope=len(tasks),scanned=n,
                counts=dict(counts),errors=errors,new_llm_calls=counts['new_llm_calls'],elapsed_seconds=round(time.time()-started,1))
            driver.save(args.output/'status.json',progress)
            print(json.dumps(progress,ensure_ascii=False),flush=True)
    return bool(errors or counts.get('deferred_execution_error'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--uuids',nargs='+')
    parser.add_argument('--uuids-file',type=Path)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--supplemental-root',type=Path,default=Path('/data/patents/chemrex_batch/quality-repair-20261016/supplemental-pilot'))
    parser.add_argument('--fill-store',type=Path,help='Overlay shared fill-store responses; enable online coref/final-product fills')
    parser.add_argument('--no-online',action='store_true',help='With --fill-store: keep all LLM calls disabled (overlay-only test)')
    raise SystemExit(main(parser.parse_args()))
