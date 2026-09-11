"""Build low-memory reusable embedding caches for Dr.Emb searches.

Fingerprint caches use compact uint8 arrays.  Continuous embeddings use
float32 arrays.  In both cases values and insertion order are copied from the
deployed pickle files so cached searches can reproduce the deployed ranking.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import itertools
import json
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
LIBRARY_ROOT = Path(os.environ.get('DREMB_LIBRARY_ROOT', APP_ROOT / 'Library'))
CACHE_ROOT = Path(os.environ.get(
    'DREMB_FINGERPRINT_CACHE_ROOT',
    LIBRARY_ROOT / '.dremb_fingerprint_cache',
))


LIBRARIES = [
    'selleck', 'mce', 'zinc', 'kcb_cn', 'pubchem_cn',
    'chembl_cn', 'kcb', 'chembl', 'pubchem',
]
EMBEDDINGS = ['ReSimNet', 'MoAble', 'ECFP', 'MACCSKeys', 'Mol2vec', 'MACAW']
EMBEDDING_SIZES = {
    'ReSimNet': 300,
    'MoAble': 256,
    'ECFP': 2048,
    'MACCSKeys': 167,
    'Mol2vec': 300,
    'MACAW': 256,
}


def fingerprint_size(embed_method):
    if embed_method not in {'ECFP', 'MACCSKeys'}:
        raise ValueError(f'{embed_method} is not a fingerprint embedding')
    return EMBEDDING_SIZES[embed_method]


def embedding_size(embed_method):
    return EMBEDDING_SIZES[embed_method]


def embedding_dtype(embed_method):
    return np.uint8 if embed_method in {'ECFP', 'MACCSKeys'} else np.float32


def table_specs(input_db):
    if input_db == 'pubchem':
        return [(LIBRARY_ROOT / f'PubChem_chunk_{number}.tsv', '\t') for number in range(1, 5)]
    specs = {
        'kcb': ('kcb.csv', ','),
        'kcb_cn': ('kcb_common_name.tsv', '\t'),
        'zinc': ('ZINC_named+waited.tsv', '\t'),
        'mce': ('MCE_library.tsv', '\t'),
        'selleck': ('Selleckchem_library.tsv', '\t'),
        'pubchem_cn': ('pubchem_common_name.tsv', '\t'),
        'chembl': ('chembl.tsv', '\t'),
        'chembl_cn': ('chembl_common_name.tsv', '\t'),
    }
    filename, separator = specs[input_db]
    return [(LIBRARY_ROOT / filename, separator)]


def legacy_pickle_paths(input_db, embed_method):
    """Return deployed embedding pickle files in their search order."""
    if input_db == 'pubchem':
        if embed_method == 'ReSimNet':
            return [
                LIBRARY_ROOT / f'ReSimNet_pubchem_{number}'
                / f'ReSimNet_pubchem_{number}_7.pkl'
                for number in range(1, 5)
            ]
        directory = LIBRARY_ROOT / f'{embed_method}_{input_db}'
        return [
            directory / f'{embed_method}_{input_db}_{number}.pkl'
            for number in range(1, 5)
        ]
    directory = LIBRARY_ROOT / f'{embed_method}_{input_db}'
    suffix = '_7' if embed_method == 'ReSimNet' else ''
    return [directory / f'{embed_method}_{input_db}{suffix}.pkl']


def cache_paths(input_db, embed_method):
    prefix = CACHE_ROOT / f'{embed_method}_{input_db}'
    return {
        'vectors': str(prefix) + '.npy',
        'names': str(prefix) + '.names.txt',
        'metadata': str(prefix) + '.json',
    }


def portable_library_source_path(path):
    normalized = os.path.normpath(str(path)).replace('\\', '/')
    marker = '/Library/'
    if marker in normalized:
        return normalized.rsplit(marker, 1)[1]
    return normalized[2:] if normalized.startswith('./') else normalized


def source_metadata_matches(stored_sources, current_sources):
    if not isinstance(stored_sources, list) or len(stored_sources) != len(current_sources):
        return False
    for stored, current in zip(stored_sources, current_sources):
        if not isinstance(stored, dict):
            return False
        if portable_library_source_path(stored.get('path', '')) != portable_library_source_path(
            current.get('path', '')
        ):
            return False
        for key in ('separator', 'size', 'mtime_ns'):
            if stored.get(key) != current.get(key):
                return False
    return True


def source_metadata(input_db):
    sources = []
    for file_path, separator in table_specs(input_db):
        stat = file_path.stat()
        sources.append({
            'path': str(file_path.resolve().relative_to(LIBRARY_ROOT.resolve())),
            'separator': separator,
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    return sources


def file_metadata(file_paths):
    sources = []
    for file_path in file_paths:
        stat = file_path.stat()
        sources.append({
            'path': str(file_path.resolve().relative_to(LIBRARY_ROOT.resolve())),
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    return sources


def valid_cache(input_db, embed_method, requested_source='auto'):
    paths = cache_paths(input_db, embed_method)
    try:
        with open(paths['metadata'], encoding='utf-8') as stream:
            metadata = json.load(stream)
        vectors = np.load(paths['vectors'], mmap_mode='r')
        valid = (
            metadata.get('format_version') in (1, 2)
            and metadata.get('input_db') == input_db
            and metadata.get('embed_method') == embed_method
            and source_metadata_matches(metadata.get('sources'), source_metadata(input_db))
            and vectors.shape == (int(metadata['rows']), embedding_size(embed_method))
            and vectors.dtype == embedding_dtype(embed_method)
            and Path(paths['names']).is_file()
        )
        if not valid:
            return False
        if metadata.get('format_version') == 1:
            return requested_source == 'auto'
        vector_source = metadata.get('vector_source')
        if requested_source != 'auto' and vector_source != requested_source:
            return False
        if vector_source == 'pickle':
            return source_metadata_matches(
                metadata.get('vector_sources'),
                file_metadata(legacy_pickle_paths(input_db, embed_method)),
            )
        return vector_source == 'smiles'
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def fingerprint_batch_task(task):
    smiles, embed_method = task
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, MACCSkeys

    RDLogger.DisableLog('rdApp.warning')
    matrix = np.zeros((len(smiles), fingerprint_size(embed_method)), dtype=np.uint8)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        if embed_method == 'ECFP':
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048, useChirality=True
            )
        else:
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix


def fingerprint_matrix(smiles, embed_method, executor, workers):
    if executor is None or workers == 1 or len(smiles) < workers * 2:
        return fingerprint_batch_task((smiles, embed_method))
    chunk_size = (len(smiles) + workers - 1) // workers
    tasks = [
        (smiles[start:start + chunk_size], embed_method)
        for start in range(0, len(smiles), chunk_size)
    ]
    return np.concatenate(list(executor.map(fingerprint_batch_task, tasks)), axis=0)


def parse_csv(value, allowed):
    if value == 'all':
        return list(allowed)
    requested = [part.strip() for part in value.split(',') if part.strip()]
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise SystemExit(f'Unknown values: {unknown}')
    return requested


def count_table_rows(file_path):
    with open(file_path, 'rb') as stream:
        rows = sum(1 for _ in stream) - 1
    if rows < 0:
        raise ValueError(f'Invalid empty table: {file_path}')
    return rows


def write_legacy_pickle_cache(
    input_db,
    embed_method,
    vectors,
    names_file,
    source_rows,
    batch_size,
):
    """Convert deployed pickle vectors without changing values or insertion order.

    Recomputing ECFP from SMILES can change a small number of chiral fingerprints
    when RDKit versions differ.  Copying the deployed vectors is therefore the
    only deterministic way to preserve the old exact ranking.
    """
    offset = 0
    for pickle_path, expected_rows in zip(
        legacy_pickle_paths(input_db, embed_method), source_rows
    ):
        with open(pickle_path, 'rb') as stream:
            mapping = pickle.load(stream)
        if not hasattr(mapping, 'items'):
            raise TypeError(f'{pickle_path} does not contain a mapping')
        if len(mapping) != expected_rows:
            raise ValueError(
                f'{pickle_path} contains {len(mapping)} rows; table contains {expected_rows}'
            )
        item_iterator = iter(mapping.items())
        while True:
            items = list(itertools.islice(item_iterator, batch_size))
            if not items:
                break
            matrix = np.empty(
                (len(items), embedding_size(embed_method)),
                dtype=embedding_dtype(embed_method),
            )
            for row_index, (name, value) in enumerate(items):
                flattened = np.asarray(value).squeeze().reshape(-1)
                if flattened.size != matrix.shape[1]:
                    raise ValueError(
                        f'{pickle_path}: {name!r} has {flattened.size} values; '
                        f'expected {matrix.shape[1]}'
                    )
                matrix[row_index] = flattened
                names_file.write(
                    str(name).replace('\r', ' ').replace('\n', ' ') + '\n'
                )
            stop = offset + len(matrix)
            vectors[offset:stop] = matrix
            offset = stop
            print(
                f'{embed_method}/{input_db}: {offset:,}/{sum(source_rows):,} '
                f'({100 * offset / max(sum(source_rows), 1):.1f}%)',
                flush=True,
            )
        del mapping
        gc.collect()
    return offset


def build_one(input_db, embed_method, batch_size, workers, force=False, source='auto'):
    paths = cache_paths(input_db, embed_method)
    table_sources = table_specs(input_db)
    sources = source_metadata(input_db)
    pickle_paths = legacy_pickle_paths(input_db, embed_method)
    if source == 'auto':
        vector_source = 'pickle' if all(path.is_file() for path in pickle_paths) else 'smiles'
    else:
        vector_source = source
    if vector_source == 'pickle' and not all(path.is_file() for path in pickle_paths):
        missing = [str(path) for path in pickle_paths if not path.is_file()]
        raise FileNotFoundError(f'Missing legacy pickle files: {missing}')
    if vector_source == 'smiles' and embed_method not in {'ECFP', 'MACCSKeys'}:
        raise ValueError(f'--source smiles is not supported for {embed_method}')

    if valid_cache(input_db, embed_method, requested_source=source) and not force:
        print(f'SKIP {embed_method}/{input_db}: valid cache already exists', flush=True)
        return

    Path(paths['vectors']).parent.mkdir(parents=True, exist_ok=True)
    suffix = f'.tmp.{os.getpid()}'
    temporary = {key: value + suffix for key, value in paths.items()}
    source_rows = [count_table_rows(file_path) for file_path, _ in table_sources]
    total_rows = sum(source_rows)
    vector_size = embedding_size(embed_method)
    vector_dtype = embedding_dtype(embed_method)
    vectors = np.lib.format.open_memmap(
        temporary['vectors'],
        mode='w+',
        dtype=vector_dtype,
        shape=(total_rows, vector_size),
    )

    started = time.perf_counter()
    offset = 0
    executor = None
    try:
        if vector_source == 'smiles' and workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'),
            )
        with open(temporary['names'], 'w', encoding='utf-8', newline='\n') as names_file:
            if vector_source == 'pickle':
                offset = write_legacy_pickle_cache(
                    input_db,
                    embed_method,
                    vectors,
                    names_file,
                    source_rows,
                    batch_size,
                )
            else:
                for file_path, sep in table_specs(input_db):
                    for chunk in pd.read_csv(file_path, sep=sep, chunksize=batch_size):
                        if 'Name' not in chunk.columns or 'SMILES' not in chunk.columns:
                            raise ValueError(f'{file_path} must contain Name and SMILES columns')
                        names = chunk['Name'].astype(str).tolist()
                        smiles = chunk['SMILES'].astype(str).tolist()
                        matrix = fingerprint_matrix(smiles, embed_method, executor, workers)
                        stop = offset + len(matrix)
                        vectors[offset:stop] = matrix
                        for name in names:
                            names_file.write(name.replace('\r', ' ').replace('\n', ' ') + '\n')
                        offset = stop
                        print(
                            f'{embed_method}/{input_db}: {offset:,}/{total_rows:,} '
                            f'({100 * offset / max(total_rows, 1):.1f}%)',
                            flush=True,
                        )
        if offset != total_rows:
            raise ValueError(f'Expected {total_rows} rows but generated {offset}')
        vectors.flush()
        del vectors
        metadata = {
            'format_version': 2,
            'input_db': input_db,
            'embed_method': embed_method,
            'rows': total_rows,
            'vector_size': vector_size,
            'dtype': np.dtype(vector_dtype).name,
            'sources': sources,
            'vector_source': vector_source,
            'vector_sources': (
                file_metadata(pickle_paths) if vector_source == 'pickle' else sources
            ),
            'source_rows': source_rows,
            'created_at_unix': time.time(),
            'build_seconds': time.perf_counter() - started,
        }
        with open(temporary['metadata'], 'w', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        for key in ('vectors', 'names', 'metadata'):
            os.replace(temporary[key], paths[key])
        print(
            f'DONE {embed_method}/{input_db}: {total_rows:,} rows in '
            f'{metadata["build_seconds"]:.2f}s',
            flush=True,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for path in temporary.values():
            try:
                os.unlink(path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--libraries', default='all')
    parser.add_argument('--embeddings', default='all')
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        '--source',
        choices=('auto', 'pickle', 'smiles'),
        default='auto',
        help='auto/pickle preserves deployed vectors; smiles recomputes them with current RDKit',
    )
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    libraries = parse_csv(args.libraries, LIBRARIES)
    embeddings = parse_csv(args.embeddings, EMBEDDINGS)
    if args.workers < 1:
        raise SystemExit('--workers must be at least 1')
    for embedding in embeddings:
        for library in libraries:
            build_one(
                library,
                embedding,
                args.batch_size,
                workers=args.workers,
                force=args.force,
                source=args.source,
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
