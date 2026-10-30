"""Recover prompt provenance from the existing cache identity and Git version.

No new identity or baseline is stored. This reconstructs the fingerprint that
the old attach_context already wrote, using its own code and prompt files.
Unknown identities remain unknown and are re-extracted.
"""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=32)
def revision_configuration(revision):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)
    revision = git('rev-parse', '--verify', revision+'^{commit}').decode().strip()
    # Read one Git snapshot, rather than spawning a Git process per file.
    with tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', revision, 'pipeline'))) as archive:
        contents = {m.name: archive.extractfile(m).read() for m in archive.getmembers()
                    if m.isfile() and (m.name.endswith('.py') or
                       (m.name.startswith('pipeline/extraction/prompts/') and Path(m.name).suffix in ('.md','.json')))}
    if 'pipeline/document/step_context.py' not in contents:
        return None
    digest = hashlib.sha256()
    for p in sorted(p for p in contents if p.startswith('pipeline/extraction/prompts/') and Path(p).suffix in ('.md','.json')):
        digest.update(p.removeprefix('pipeline/extraction/prompts/').encode());digest.update(contents[p])
    for p in sorted(p for p in contents if p.endswith('.py')):
        digest.update(p.removeprefix('pipeline/').encode());digest.update(contents[p])
    context = re.search(r"CONTEXT_VERSION = ['\"]([^'\"]+)", contents['pipeline/document/step_context.py'].decode())[1]
    segmentation = re.search(r"SEGMENTATION_VERSION = ['\"]([^'\"]+)", contents['pipeline/document/segmenter.py'].decode())[1]
    versions = json.loads(contents['pipeline/extraction/prompts/versions.json'])
    prompts = {}
    for language in ('zh','en'):
        prompts[language] = {}
        for task in ('ner','action'):
            name = f'{task}_{language}';version = versions.get(name,'v1')
            raw = contents[f'pipeline/extraction/prompts/{name}/{version}.md'].decode()
            try:
                import frontmatter
                content = frontmatter.loads(raw).content.strip()
            except ImportError:
                content = raw.split('---',2)[2].strip() if raw.startswith('---') else raw
            prompts[language][task] = dict(version=version,content=content)
    return dict(revision=revision,context=context,segmentation=segmentation,
                code_digest=digest.hexdigest(),prompts=prompts)


def configuration_for_record(record, language, revision):
    config = revision_configuration(revision)
    chunk = record.get('source') or {}
    if not config or chunk.get('context_version') != config['context']:
        return None
    required = ('text','block_start','block_end','page_start','page_end','steps')
    if any(k not in chunk for k in required):
        return None
    payload = (config['segmentation'],re.sub(r'\s+',' ',chunk['text']).strip(),
               chunk['block_start'],chunk['block_end'],chunk['page_start'],chunk['page_end'],
               chunk.get('block_indices',[]),chunk.get('images',[]),chunk.get('tables',[]))
    base = hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps([base,config['context'],config['code_digest'],chunk['steps']],
                                  sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if fingerprint != record.get('chunk_fingerprint') or fingerprint != chunk.get('chunk_fingerprint'):
        return None
    return dict(model=None,preset=None,prompts=config['prompts'][language],
                prompt_provenance_verified=dict(git_revision=config['revision'],cache_field='chunk_fingerprint'))


@lru_cache(maxsize=4)
def historical_revisions(revision):
    history = subprocess.check_output(['git','log','--format=%H',revision,'--','pipeline'],cwd=ROOT)
    return tuple(dict.fromkeys([revision, *history.decode().splitlines()]))


def historical_configuration_for_record(record, language, revision):
    """Try actual ancestors of the configured legacy version, keeping proof.

    A different revision is eligible only when it reproduces both saved input
    fingerprints. Unknown/uncommitted code and changed prompts remain ineligible.
    """
    for candidate in historical_revisions(revision):
        config = configuration_for_record(record, language, candidate)
        if config:
            return config
    return None
