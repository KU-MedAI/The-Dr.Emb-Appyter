"""Runtime utilities for Dr.Emb Appyter 0.2."""

import pandas as pd
import re
import logging
import numpy as np
from rdkit import Chem, DataStructs
from concurrent.futures import ProcessPoolExecutor
from rdkit.Chem import AllChem, MACCSkeys
# DeepChem imports optional TensorFlow, PyG, Lightning, and JAX integrations at
# module load time.  Dr.Emb only uses its molecular featurizers, so suppress
# those irrelevant missing-optional-dependency warnings in notebook output.
logging.getLogger('deepchem').setLevel(logging.ERROR)
import deepchem as dc
import os
import sys
import torch
import methods.moable.model
import pickle
import faiss
import atexit
import hashlib
import heapq
import itertools
import json
import shlex
import subprocess
import time
import traceback
from tqdm import tqdm
from itertools import combinations
from IPython.display import HTML, display, Markdown, IFrame, FileLink, Image, HTML
from scipy.spatial import distance

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_BATCH_SIZE = int(os.environ.get('DREMB_BATCH_SIZE', '10000'))
FINGERPRINT_SEARCH_BATCH_SIZE = int(
    os.environ.get('DREMB_FINGERPRINT_SEARCH_BATCH_SIZE', '0')
)
CONTINUOUS_SEARCH_BATCH_SIZE = int(
    os.environ.get('DREMB_CONTINUOUS_SEARCH_BATCH_SIZE', '0')
)
APP_ROOT = os.environ.get(
    'DREMB_APP_ROOT',
    os.path.dirname(os.path.abspath(__file__)),
)
LIBRARY_ROOT = os.environ.get('DREMB_LIBRARY_ROOT', os.path.join(APP_ROOT, 'Library'))
METHODS_ROOT = os.environ.get('DREMB_METHODS_ROOT', os.path.join(APP_ROOT, 'methods'))
FINGERPRINT_CACHE_ROOT = os.environ.get(
    'DREMB_FINGERPRINT_CACHE_ROOT',
    os.path.join(LIBRARY_ROOT, '.dremb_fingerprint_cache'),
)
UMAP_CACHE_ROOT = os.environ.get(
    'DREMB_UMAP_CACHE_ROOT',
    os.path.join(LIBRARY_ROOT, '.dremb_umap_cache'),
)


def library_path(*parts):
    return os.path.join(LIBRARY_ROOT, *parts)


def methods_path(*parts):
    return os.path.join(METHODS_ROOT, *parts)


def _update_array_digest(digest, values):
    """Hash a numeric matrix without making one full-size contiguous copy."""
    array = np.asarray(values)
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.dtype.str.encode('ascii'))
    if array.dtype.hasobject:
        for value in array.flat:
            encoded = str(value).encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
        return

    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
        return
    row_bytes = max(int(array[0:1].nbytes), 1)
    rows_per_chunk = max(1, (64 * 1024 * 1024) // row_bytes)
    for start in range(0, len(array), rows_per_chunk):
        block = np.ascontiguousarray(array[start:start + rows_per_chunk])
        digest.update(block.view(np.uint8))


def umap_projection_cache_key(
    library_features,
    query_features,
    input_db,
    embed_method,
    sim_method,
    model_identity,
):
    """Identify the exact UMAP inputs used by one Appyter result."""
    digest = hashlib.sha256()
    for value in (
        'dremb-umap-projection-v1',
        input_db,
        embed_method,
        sim_method,
        model_identity,
    ):
        encoded = str(value).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    _update_array_digest(digest, library_features)
    _update_array_digest(digest, query_features)
    return digest.hexdigest()


def umap_library_projection_cache_key(
    library_features,
    input_db,
    embed_method,
    model_identity,
):
    """Identify a pretrained UMAP library projection independent of queries."""
    digest = hashlib.sha256()
    for value in (
        'dremb-umap-library-projection-v1',
        input_db,
        embed_method,
        model_identity,
    ):
        encoded = str(value).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    _update_array_digest(digest, library_features)
    return digest.hexdigest()


def umap_projection_cache_path(input_db, embed_method, sim_method, cache_key):
    safe_parts = [
        re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))
        for value in (input_db, embed_method, sim_method)
    ]
    return os.path.join(
        UMAP_CACHE_ROOT,
        '_'.join(safe_parts) + '_' + str(cache_key) + '.npz',
    )


def load_umap_projection_cache(
    input_db,
    embed_method,
    sim_method,
    cache_key,
    library_rows,
    query_rows,
):
    path = umap_projection_cache_path(
        input_db, embed_method, sim_method, cache_key
    )
    try:
        with np.load(path, allow_pickle=False) as cached:
            library_projection = np.asarray(cached['library'], dtype=np.float32)
            query_projection = np.asarray(cached['query'], dtype=np.float32)
        if library_projection.shape != (int(library_rows), 3):
            return None
        if query_projection.shape != (int(query_rows), 3):
            return None
        return library_projection, query_projection
    except (OSError, ValueError, KeyError):
        return None


def save_umap_projection_cache(
    input_db,
    embed_method,
    sim_method,
    cache_key,
    library_projection,
    query_projection,
):
    """Atomically save a compact projection cache; failure is non-fatal."""
    path = umap_projection_cache_path(
        input_db, embed_method, sim_method, cache_key
    )
    temporary_path = (
        path + f'.tmp.{os.getpid()}.{time.time_ns()}.npz'
    )
    try:
        os.makedirs(UMAP_CACHE_ROOT, exist_ok=True)
        np.savez_compressed(
            temporary_path,
            library=np.asarray(library_projection, dtype=np.float32),
            query=np.asarray(query_projection, dtype=np.float32),
        )
        os.replace(temporary_path, path)
        return True
    except OSError:
        return False
    finally:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass


def _format_elapsed(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def cleanup_runtime():
    """Release large Python, FAISS, and CUDA allocations before a process exits."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


atexit.register(cleanup_runtime)


def install_fail_fast_exit(exit_code=1):
    """Terminate the current Appyter/Jupyter kernel on uncaught exceptions.

    Appyter jobs can otherwise leave a failed kernel process alive and block the
    queue. This function is opt-in so importing the module in ordinary Python code
    keeps normal exception behavior.
    """
    def _exit_after_print(etype, evalue, tb):
        traceback.print_exception(etype, evalue, tb)
        cleanup_runtime()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(exit_code)

    sys.excepthook = _exit_after_print

    try:
        ipython = get_ipython()
    except NameError:
        ipython = None

    if ipython is not None:
        def _ipython_exception_handler(shell, etype, evalue, tb, tb_offset=None):
            _exit_after_print(etype, evalue, tb)

        ipython.set_custom_exc((Exception,), _ipython_exception_handler)


def run_checked(cmd, progress_desc=None, **kwargs):
    """Run an external command quietly and raise with its command if it fails."""
    command = [str(part) for part in cmd]
    # Subprocess commands are implementation details and should not clutter the
    # Appyter result. They can still be enabled explicitly when debugging.
    if os.environ.get('DREMB_LOG_COMMANDS', '0') == '1':
        print('+ ' + shlex.join(command))
    if progress_desc:
        with tqdm(
            total=1,
            desc=progress_desc,
            unit='step',
            dynamic_ncols=True,
        ) as progress:
            completed = subprocess.run(command, check=False, **kwargs)
            if completed.returncode == 0:
                progress.update(1)
    else:
        completed = subprocess.run(command, check=False, **kwargs)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{shlex.join(command)}"
        )
    return completed


def _flatten_vector(value):
    arr = np.asarray(value)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    return np.asarray(arr).reshape(-1)


def _normalize_rows(matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    # Match the deployed FAISS normalization while avoiding NumPy's full-size
    # ``matrix * matrix`` temporary (several GB for ECFP libraries).
    faiss.normalize_L2(matrix)
    return matrix


def _iter_item_batches(mapping, batch_size=None):
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    batch = []
    for item in mapping.items():
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _items_to_matrix(items, dtype=np.float32):
    keys = []
    vectors = []
    for key, value in items:
        vec = _flatten_vector(value)
        if vec.size == 0:
            continue
        keys.append(str(key))
        vectors.append(vec)
    if not vectors:
        return [], np.empty((0, 0), dtype=dtype)
    return keys, np.ascontiguousarray(np.asarray(vectors, dtype=dtype))


class _TopK:
    def __init__(self, k, larger_is_better=True):
        self.k = int(k)
        self.larger_is_better = larger_is_better
        self.heap = []
        self._counter = itertools.count()

    def push_many(self, keys, scores):
        if self.k <= 0:
            return
        for key, score in zip(keys, scores):
            score = float(score)
            if not np.isfinite(score):
                continue
            rank_score = score if self.larger_is_better else -score
            # Earlier candidates win exact ties, matching Python's stable sort
            # used by the deployed Jaccard implementation.  The negative
            # counter also makes a later tied item the worst heap entry.
            entry = (rank_score, -next(self._counter), str(key), score)
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, entry)
            elif entry[:2] > self.heap[0][:2]:
                heapq.heapreplace(self.heap, entry)

    def ordered(self):
        # Preserve the incoming order for exact ties.  FAISS supplies that
        # order for cosine/L2 batches; table order supplies it for Jaccard.
        rows = sorted(self.heap, key=lambda item: item[:2], reverse=True)
        return [(key, score) for _, _, key, score in rows]


def _load_query_matrix(output_embed_filename, embed_method=None):
    with open(output_embed_filename, "rb") as f:
        query_embedding_vectors = pickle.load(f)

    if isinstance(query_embedding_vectors, dict):
        items = query_embedding_vectors.items()
    else:
        merged = {}
        for obj in query_embedding_vectors:
            if isinstance(obj, dict):
                merged.update(obj)
        items = merged.items()

    query_names, query_vectors = _items_to_matrix(items, dtype=np.float32)
    if query_vectors.size == 0:
        raise ValueError(f"No query embedding vectors found in {output_embed_filename}")
    return query_names, query_vectors


def _score_batch(query_vectors, library_vectors, sim_method):
    query_vectors = _normalize_rows(query_vectors.copy())
    library_vectors = _normalize_rows(library_vectors)
    if sim_method == 'Cosine':
        return np.dot(query_vectors, library_vectors.T)
    if sim_method == 'Euclidean':
        q_norm = np.sum(query_vectors * query_vectors, axis=1, keepdims=True)
        l_norm = np.sum(library_vectors * library_vectors, axis=1)
        return q_norm + l_norm - 2.0 * np.dot(query_vectors, library_vectors.T)
    raise ValueError("Invalid similarity method. Use 'Cosine' or 'Euclidean'.")


def _faiss_search_batch(query_vectors, library_vectors, sim_method, topk):
    """Run an exact FAISS search without copying vectors into an index."""
    query_vectors = _normalize_rows(query_vectors.copy())
    library_vectors = _normalize_rows(library_vectors)
    if library_vectors.size == 0:
        return (
            np.empty((len(query_vectors), 0), dtype=np.float32),
            np.empty((len(query_vectors), 0), dtype=np.int64),
        )

    if sim_method == 'Cosine':
        metric = faiss.METRIC_INNER_PRODUCT
    elif sim_method == 'Euclidean':
        metric = faiss.METRIC_L2
    else:
        raise ValueError("Invalid similarity method. Use 'Cosine' or 'Euclidean'.")

    # faiss.knn uses the same exhaustive Flat kernels and tie ordering as
    # IndexFlat.search, but avoids retaining a second float32 copy in an index.
    return faiss.knn(
        query_vectors,
        library_vectors,
        min(int(topk), len(library_vectors)),
        metric,
    )


def _push_faiss_results(accumulators, keys, scores, indices):
    for query_idx, accumulator in enumerate(accumulators):
        valid = indices[query_idx] >= 0
        batch_indices = indices[query_idx][valid]
        batch_keys = [keys[int(index)] for index in batch_indices]
        accumulator.push_many(batch_keys, scores[query_idx][valid])


def _fingerprint_size(embed_method):
    if embed_method == 'ECFP':
        return 2048
    if embed_method == 'MACCSKeys':
        return 167
    raise ValueError(f"Unsupported fingerprint embedding method: {embed_method}")


def _embedding_size(embed_method):
    sizes = {
        'ReSimNet': 300,
        'MoAble': 256,
        'ECFP': 2048,
        'MACCSKeys': 167,
        'Mol2vec': 300,
        'MACAW': 256,
    }
    try:
        return sizes[embed_method]
    except KeyError as exc:
        raise ValueError(f'Unsupported embedding method: {embed_method}') from exc


def _fingerprint_from_smiles(smiles, embed_method):
    mol = Chem.MolFromSmiles(str(smiles))
    size = _fingerprint_size(embed_method)
    arr = np.zeros((size,), dtype=np.uint8)
    if mol is None:
        return arr
    if embed_method == 'ECFP':
        fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
    elif embed_method == 'MACCSKeys':
        fp_obj = MACCSkeys.GenMACCSKeys(mol)
    else:
        raise ValueError(f"Unsupported fingerprint embedding method: {embed_method}")
    DataStructs.ConvertToNumpyArray(fp_obj, arr)
    return arr


def _fingerprint_matrix_from_smiles(smiles_values, embed_method):
    matrix = np.zeros((len(smiles_values), _fingerprint_size(embed_method)), dtype=np.uint8)
    for idx, smiles in enumerate(smiles_values):
        matrix[idx] = _fingerprint_from_smiles(smiles, embed_method)
    return matrix


def _library_table_specs(input_db):
    if input_db == 'pubchem':
        return [(library_path(f'PubChem_chunk_{n}.tsv'), '\t') for n in range(1, 5)]
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
    if input_db not in specs:
        raise ValueError(f"Unsupported input_db for library table streaming: {input_db}")
    filename, sep = specs[input_db]
    return [(library_path(filename), sep)]


def fingerprint_cache_paths(input_db, embed_method):
    prefix = os.path.join(FINGERPRINT_CACHE_ROOT, f'{embed_method}_{input_db}')
    return {
        'vectors': prefix + '.npy',
        'names': prefix + '.names.txt',
        'metadata': prefix + '.json',
    }


def _portable_library_source_path(path):
    """Return a Library-relative identity for host/container cache reuse."""
    normalized = os.path.normpath(str(path)).replace('\\', '/')
    marker = '/Library/'
    if marker in normalized:
        return normalized.rsplit(marker, 1)[1]
    return normalized[2:] if normalized.startswith('./') else normalized


def _source_metadata_matches(stored_sources, current_sources):
    if not isinstance(stored_sources, list) or len(stored_sources) != len(current_sources):
        return False
    for stored, current in zip(stored_sources, current_sources):
        if not isinstance(stored, dict):
            return False
        if _portable_library_source_path(stored.get('path', '')) != _portable_library_source_path(
            current.get('path', '')
        ):
            return False
        for key in ('separator', 'size', 'mtime_ns'):
            if stored.get(key) != current.get(key):
                return False
    return True


def fingerprint_source_metadata(input_db):
    sources = []
    for file_path, sep in _library_table_specs(input_db):
        stat = os.stat(file_path)
        sources.append({
            'path': os.path.relpath(os.path.abspath(file_path), LIBRARY_ROOT),
            'separator': sep,
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    return sources


def fingerprint_pickle_source_metadata(input_db, embed_method):
    sources = []
    for file_path in _embedding_files_for_input(input_db, embed_method):
        stat = os.stat(file_path)
        sources.append({
            'path': os.path.relpath(os.path.abspath(file_path), LIBRARY_ROOT),
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    return sources


def _load_fingerprint_cache(input_db, embed_method):
    paths = fingerprint_cache_paths(input_db, embed_method)
    try:
        with open(paths['metadata'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if metadata.get('format_version') not in (1, 2):
            return None
        if metadata.get('input_db') != input_db or metadata.get('embed_method') != embed_method:
            return None
        if not _source_metadata_matches(
            metadata.get('sources'), fingerprint_source_metadata(input_db)
        ):
            return None
        if metadata.get('format_version') == 2:
            vector_source = metadata.get('vector_source')
            if vector_source == 'pickle':
                if not _source_metadata_matches(
                    metadata.get('vector_sources'),
                    fingerprint_pickle_source_metadata(input_db, embed_method),
                ):
                    return None
            elif vector_source != 'smiles':
                return None
        vectors = np.load(paths['vectors'], mmap_mode='r')
        expected_shape = (int(metadata['rows']), _fingerprint_size(embed_method))
        if vectors.shape != expected_shape or vectors.dtype != np.uint8:
            return None
        return paths, vectors, metadata
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _load_continuous_embedding_cache(input_db, embed_method):
    """Load a current float32 mmap cache copied from deployed pickle data."""
    if embed_method in {'ECFP', 'MACCSKeys'}:
        return None
    paths = fingerprint_cache_paths(input_db, embed_method)
    try:
        with open(paths['metadata'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if (
            metadata.get('format_version') != 2
            or metadata.get('input_db') != input_db
            or metadata.get('embed_method') != embed_method
            or not _source_metadata_matches(
                metadata.get('sources'), fingerprint_source_metadata(input_db)
            )
            or metadata.get('vector_source') != 'pickle'
            or not _source_metadata_matches(
                metadata.get('vector_sources'),
                fingerprint_pickle_source_metadata(input_db, embed_method),
            )
        ):
            return None
        vectors = np.load(paths['vectors'], mmap_mode='r')
        if (
            vectors.shape != (int(metadata['rows']), _embedding_size(embed_method))
            or vectors.dtype != np.float32
        ):
            return None
        source_rows = metadata.get('source_rows')
        if (
            not isinstance(source_rows, list)
            or not source_rows
            or sum(int(rows) for rows in source_rows) != len(vectors)
        ):
            return None
        return paths, vectors, metadata
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _iter_continuous_embedding_cache_batches(
    input_db,
    embed_method,
    batch_size=None,
    with_source_index=False,
):
    cached = _load_continuous_embedding_cache(input_db, embed_method)
    if cached is None:
        return
    paths, vectors, metadata = cached
    source_rows = metadata['source_rows']
    with open(paths['names'], 'r', encoding='utf-8') as names_file:
        source_start = 0
        for source_index, row_count in enumerate(source_rows):
            source_stop = source_start + int(row_count)
            # Match the deployed Flat-index call (and its exact tie ordering)
            # by searching one original pickle source at a time.  An explicit
            # override can reduce the ceiling further on smaller hosts.
            step = int(
                batch_size
                or CONTINUOUS_SEARCH_BATCH_SIZE
                or max(int(row_count), 1)
            )
            for start in range(source_start, source_stop, step):
                stop = min(start + step, source_stop)
                names = []
                for _ in range(stop - start):
                    line = names_file.readline()
                    if not line:
                        raise ValueError(
                            f'Embedding cache name count is shorter than {len(vectors)}'
                        )
                    names.append(line.rstrip('\n'))
                # The FAISS normalization is in-place.  Copy each bounded batch
                # so the read-only mmap remains byte-for-byte unchanged.
                matrix = np.array(vectors[start:stop], dtype=np.float32, copy=True)
                result = (names, matrix)
                yield (source_index, *result) if with_source_index else result
            source_start = source_stop


def _search_continuous_embedding_cache(
    input_db,
    embed_method,
    output_embed_filename,
    sim_method,
    topk_candidate,
    name,
    per_source=False,
):
    cached = _load_continuous_embedding_cache(input_db, embed_method)
    if cached is None:
        return None
    _, cached_vectors, _ = cached
    expected_total = len(cached_vectors)
    query_names, query_vectors = _load_query_matrix(output_embed_filename)
    larger_is_better = sim_method == 'Cosine'
    accumulators = [
        _TopK(topk_candidate, larger_is_better=larger_is_better)
        for _ in query_names
    ]
    ordered_by_query = [[] for _ in query_names] if per_source else None
    current_source = None
    total = 0
    batches = _iter_continuous_embedding_cache_batches(
        input_db,
        embed_method,
        with_source_index=per_source,
    )
    with tqdm(
        total=expected_total,
        desc=f'{name}: {embed_method} similarity search',
        unit='compound',
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for batch in batches:
            if per_source:
                source_index, keys, library_vectors = batch
                if current_source is not None and source_index != current_source:
                    for query_idx, accumulator in enumerate(accumulators):
                        ordered_by_query[query_idx].extend(accumulator.ordered())
                    accumulators = [
                        _TopK(topk_candidate, larger_is_better=larger_is_better)
                        for _ in query_names
                    ]
                current_source = source_index
            else:
                keys, library_vectors = batch
            scores, indices = _faiss_search_batch(
                query_vectors,
                library_vectors,
                sim_method,
                topk_candidate,
            )
            total += len(keys)
            _push_faiss_results(accumulators, keys, scores, indices)
            progress.update(len(keys))

    print(f'{name}: Searched from {total} cached candidates...')
    if per_source:
        if current_source is not None:
            for query_idx, accumulator in enumerate(accumulators):
                ordered_by_query[query_idx].extend(accumulator.ordered())
    else:
        ordered_by_query = [accumulator.ordered() for accumulator in accumulators]

    max_len = max((len(rows) for rows in ordered_by_query), default=0)
    similarity = np.full((len(ordered_by_query), max_len), np.nan, dtype=np.float32)
    index = np.full((len(ordered_by_query), max_len), -1, dtype=np.int64)
    result_df_list = []
    for row_idx, rows in enumerate(tqdm(
        ordered_by_query,
        total=len(ordered_by_query),
        desc=f'{name}: assembling search results',
        unit='query',
        dynamic_ncols=True,
    )):
        keys = [key for key, _ in rows]
        scores = [score for _, score in rows]
        similarity[row_idx, :len(scores)] = scores
        index[row_idx, :len(keys)] = np.arange(len(keys), dtype=np.int64)
        result_df_list.append(pd.DataFrame(index=[str(key) for key in keys]))
    return similarity, index, result_df_list, ordered_by_query


def _select_continuous_embedding_cache_vectors(input_db, embed_method, selected_keys):
    cached = _load_continuous_embedding_cache(input_db, embed_method)
    if cached is None:
        return pd.DataFrame()
    paths, vectors, _ = cached
    remaining = set(str(key) for key in selected_keys)
    if not remaining:
        return pd.DataFrame()
    found_names = []
    found_indices = []
    scanned = 0
    pending_update = 0
    with tqdm(
        total=len(vectors),
        desc=f'{embed_method}: selecting result vectors',
        unit='compound',
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        with open(paths['names'], 'r', encoding='utf-8') as names_file:
            for row_index, line in enumerate(names_file):
                scanned += 1
                pending_update += 1
                key = line.rstrip('\n')
                if key in remaining:
                    found_names.append(key)
                    found_indices.append(row_index)
                    remaining.remove(key)
                    if not remaining:
                        break
                if pending_update >= DEFAULT_BATCH_SIZE:
                    progress.update(pending_update)
                    pending_update = 0
        if pending_update:
            progress.update(pending_update)
        # A successful lookup can stop before the end of the cache.  Adjust the
        # completed total to the number that was actually scanned so the final
        # display is truthful and finishes at 100%.
        if not remaining and hasattr(progress, 'total'):
            progress.total = scanned
            if hasattr(progress, 'refresh'):
                progress.refresh()
    if not found_names:
        return pd.DataFrame()
    with tqdm(
        total=1,
        desc=f'{embed_method}: assembling result-vector table',
        unit='step',
        dynamic_ncols=True,
    ) as progress:
        result = pd.DataFrame(
            np.asarray(vectors[np.asarray(found_indices, dtype=np.int64)], dtype=np.float32),
            index=found_names,
        )
        progress.update(1)
    return result


def _iter_library_fingerprint_batches(
    input_db,
    embed_method,
    batch_size=None,
    with_source_index=False,
):
    """Yield cached fingerprints while retaining original source boundaries.

    With a cache, the default is one FAISS call per original source file.  This
    reproduces the deployed Flat-index ranking (including ties) without loading
    the old pickle/DataFrame/index copies.  A positive environment override or
    explicit ``batch_size`` trades exact tie ordering for a lower memory ceiling.
    """
    cached = _load_fingerprint_cache(input_db, embed_method)
    if cached is not None:
        paths, vectors, metadata = cached
        source_rows = metadata.get('source_rows')
        if (
            not isinstance(source_rows, list)
            or not source_rows
            or sum(int(rows) for rows in source_rows) != len(vectors)
        ):
            source_rows = [len(vectors)]
        requested_batch_size = batch_size or FINGERPRINT_SEARCH_BATCH_SIZE
        with open(paths['names'], 'r', encoding='utf-8') as names_file:
            source_start = 0
            for source_index, row_count in enumerate(source_rows):
                source_stop = source_start + int(row_count)
                step = requested_batch_size or max(int(row_count), 1)
                for start in range(source_start, source_stop, step):
                    stop = min(start + step, source_stop)
                    names = []
                    for _ in range(stop - start):
                        line = names_file.readline()
                        if not line:
                            raise ValueError(
                                f"Fingerprint cache name count is shorter than {len(vectors)}"
                            )
                        names.append(line.rstrip('\n'))
                    result = (names, np.asarray(vectors[start:stop], dtype=np.uint8))
                    yield (source_index, *result) if with_source_index else result
                source_start = source_stop
        return

    fallback_batch_size = batch_size or FINGERPRINT_SEARCH_BATCH_SIZE or DEFAULT_BATCH_SIZE
    for source_index, (file_path, sep) in enumerate(_library_table_specs(input_db)):
        for chunk in pd.read_csv(file_path, sep=sep, chunksize=fallback_batch_size):
            if 'Name' not in chunk.columns or 'SMILES' not in chunk.columns:
                raise ValueError(f"{file_path} must contain Name and SMILES columns")
            names = chunk['Name'].astype(str).tolist()
            smiles_values = chunk['SMILES'].astype(str).tolist()
            result = (names, _fingerprint_matrix_from_smiles(smiles_values, embed_method))
            yield (source_index, *result) if with_source_index else result


def _search_fingerprints_from_tables(
    input_db,
    embed_method,
    output_embed_filename,
    sim_method,
    topk_candidate,
    name,
    per_source=False,
):
    cached = _load_fingerprint_cache(input_db, embed_method)
    expected_total = len(cached[1]) if cached is not None else None
    query_names, query_vectors = _load_query_matrix(output_embed_filename)
    larger_is_better = sim_method == 'Cosine'
    accumulators = [_TopK(topk_candidate, larger_is_better=larger_is_better) for _ in query_names]
    ordered_by_query = [[] for _ in query_names] if per_source else None
    current_source = None
    total = 0

    batch_iter = _iter_library_fingerprint_batches(
        input_db,
        embed_method,
        with_source_index=per_source,
    )
    with tqdm(
        total=expected_total,
        desc=f"{name}: {embed_method} similarity search",
        unit='compound',
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for batch in batch_iter:
            if per_source:
                source_index, keys, library_vectors = batch
                if current_source is not None and source_index != current_source:
                    for query_idx, accumulator in enumerate(accumulators):
                        ordered_by_query[query_idx].extend(accumulator.ordered())
                    accumulators = [
                        _TopK(topk_candidate, larger_is_better=larger_is_better)
                        for _ in query_names
                    ]
                current_source = source_index
            else:
                keys, library_vectors = batch
            batch_rows = len(keys)
            total += batch_rows
            if library_vectors.size:
                scores, indices = _faiss_search_batch(
                    query_vectors,
                    library_vectors.astype(np.float32),
                    sim_method,
                    topk_candidate,
                )
                _push_faiss_results(accumulators, keys, scores, indices)
            progress.update(batch_rows)

    print(f"{name}: Searched from {total} candidates...")
    if per_source:
        if current_source is not None:
            for query_idx, accumulator in enumerate(accumulators):
                ordered_by_query[query_idx].extend(accumulator.ordered())
    else:
        ordered_by_query = [accumulator.ordered() for accumulator in accumulators]
    max_len = max((len(rows) for rows in ordered_by_query), default=0)
    similarity = np.full((len(ordered_by_query), max_len), np.nan, dtype=np.float32)
    index = np.full((len(ordered_by_query), max_len), -1, dtype=np.int64)
    result_df_list = []
    for row_idx, rows in enumerate(tqdm(
        ordered_by_query,
        total=len(ordered_by_query),
        desc=f'{name}: assembling search results',
        unit='query',
        dynamic_ncols=True,
    )):
        keys = [key for key, _ in rows]
        scores = [score for _, score in rows]
        similarity[row_idx, :len(scores)] = scores
        index[row_idx, :len(keys)] = np.arange(len(keys), dtype=np.int64)
        result_df_list.append(pd.DataFrame(index=[str(key) for key in keys]))
    return similarity, index, result_df_list, ordered_by_query


def _select_fingerprint_vectors_from_tables(input_db, embed_method, selected_keys):
    selected_keys = set(str(key) for key in selected_keys)
    if not selected_keys:
        return pd.DataFrame()
    found = {}
    cached = _load_fingerprint_cache(input_db, embed_method)
    if cached is not None:
        paths, vectors, _ = cached
        found_names = []
        found_indices = []
        scanned = 0
        pending_update = 0
        with tqdm(
            total=len(vectors),
            desc=f'{embed_method}: selecting result vectors',
            unit='compound',
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            with open(paths['names'], 'r', encoding='utf-8') as names_file:
                for row_index, line in enumerate(names_file):
                    scanned += 1
                    pending_update += 1
                    key = line.rstrip('\n')
                    if key in selected_keys:
                        found_names.append(key)
                        found_indices.append(row_index)
                        selected_keys.remove(key)
                        if not selected_keys:
                            break
                    if pending_update >= DEFAULT_BATCH_SIZE:
                        progress.update(pending_update)
                        pending_update = 0
            if pending_update:
                progress.update(pending_update)
            if not selected_keys and hasattr(progress, 'total'):
                progress.total = scanned
                if hasattr(progress, 'refresh'):
                    progress.refresh()
        if found_names:
            with tqdm(
                total=1,
                desc=f'{embed_method}: assembling result-vector table',
                unit='step',
                dynamic_ncols=True,
            ) as progress:
                result = pd.DataFrame(
                    np.asarray(
                        vectors[np.asarray(found_indices, dtype=np.int64)],
                        dtype=np.uint8,
                    ),
                    index=found_names,
                )
                progress.update(1)
            return result

    for file_path, sep in _library_table_specs(input_db):
        if not selected_keys:
            break
        for chunk in pd.read_csv(file_path, sep=sep, chunksize=DEFAULT_BATCH_SIZE):
            if not selected_keys:
                break
            chunk = chunk[chunk['Name'].astype(str).isin(selected_keys)]
            if chunk.empty:
                continue
            for name, smiles in zip(chunk['Name'].astype(str), chunk['SMILES'].astype(str)):
                if name in selected_keys:
                    found[name] = _fingerprint_from_smiles(smiles, embed_method)
                    selected_keys.remove(name)
    if not found:
        return pd.DataFrame()
    with tqdm(
        total=1,
        desc=f'{embed_method}: assembling result-vector table',
        unit='step',
        dynamic_ncols=True,
    ) as progress:
        result = pd.DataFrame.from_dict(found, orient='index')
        progress.update(1)
    return result


def _jaccard_scores(query_vector, library_vectors):
    query_bool = _flatten_vector(query_vector).astype(bool)
    library_bool = library_vectors.astype(bool, copy=False)
    # Match ``1 - scipy.spatial.distance.jaccard`` used by the deployed code,
    # including its floating-point operation order.  Computing intersection /
    # union is mathematically equivalent but can differ in the final bit and
    # therefore in a downloaded TSV's last decimal place.
    unequal = np.count_nonzero(library_bool != query_bool, axis=1).astype(np.float64)
    union = np.count_nonzero(library_bool | query_bool, axis=1).astype(np.float64)
    jaccard_distance = np.divide(
        unequal,
        union,
        out=np.zeros_like(unequal),
        where=union != 0,
    )
    return 1.0 - jaccard_distance


def _embedding_files_for_pubchem(embed_method):
    embedding_vectors_directory = library_path(f"{embed_method}_pubchem")
    return [
        os.path.join(embedding_vectors_directory, f'{embed_method}_pubchem_1.pkl'),
        os.path.join(embedding_vectors_directory, f'{embed_method}_pubchem_2.pkl'),
        os.path.join(embedding_vectors_directory, f'{embed_method}_pubchem_3.pkl'),
        os.path.join(embedding_vectors_directory, f'{embed_method}_pubchem_4.pkl'),
    ]


def _embedding_files_for_input(input_db, embed_method):
    if input_db == 'pubchem':
        if embed_method == 'ReSimNet':
            return [
                library_path(f'ReSimNet_{input_db}_{n}', f'ReSimNet_{input_db}_{n}_7.pkl')
                for n in range(1, 5)
            ]
        return _embedding_files_for_pubchem(embed_method)

    if embed_method == 'ReSimNet':
        return [library_path(f'ReSimNet_{input_db}', f'ReSimNet_{input_db}_7.pkl')]
    return [library_path(f'{embed_method}_{input_db}', f'{embed_method}_{input_db}.pkl')]


def _collect_vectors_from_pickle_files(pickle_files, selected_keys, dtype=np.uint8):
    selected_keys = set(str(key) for key in selected_keys)
    if not selected_keys:
        return pd.DataFrame()
    found = {}
    for pickle_file in tqdm(
        pickle_files,
        total=len(pickle_files),
        desc='Selecting result vectors from embedding files',
        unit='file',
        dynamic_ncols=True,
    ):
        if not selected_keys:
            break
        with open(pickle_file, "rb") as f:
            embedding_vectors_dict = pickle.load(f)
        try:
            for key in list(selected_keys):
                if key in embedding_vectors_dict:
                    found[key] = _flatten_vector(embedding_vectors_dict[key]).astype(dtype, copy=False)
                    selected_keys.remove(key)
        finally:
            del embedding_vectors_dict
            cleanup_runtime()

    if not found:
        return pd.DataFrame()
    with tqdm(
        total=1,
        desc='Assembling result-vector table',
        unit='step',
        dynamic_ncols=True,
    ) as progress:
        result = pd.DataFrame.from_dict(found, orient='index')
        progress.update(1)
    return result


def select_embedding_vectors(input_db, embed_method, selected_keys):
    """Load only selected vectors into the returned dict.

    Pickle files still have to be opened as whole Python objects, but this avoids
    converting entire libraries to pandas DataFrames or keeping full dictionaries
    alive after the requested vectors have been copied.
    """
    vector_dtype = np.uint8 if embed_method in ['ECFP', 'MACCSKeys'] else np.float32
    if embed_method in ['ECFP', 'MACCSKeys'] and input_db != 'custom':
        vectors_df = _select_fingerprint_vectors_from_tables(input_db, embed_method, selected_keys)
    elif input_db != 'custom' and _load_continuous_embedding_cache(input_db, embed_method) is not None:
        vectors_df = _select_continuous_embedding_cache_vectors(
            input_db,
            embed_method,
            selected_keys,
        )
    else:
        vectors_df = _collect_vectors_from_pickle_files(
            _embedding_files_for_input(input_db, embed_method),
            selected_keys,
            dtype=vector_dtype,
        )
    if vectors_df.empty:
        return {}
    return {
        str(index): row.to_numpy(dtype=vector_dtype, copy=True)
        for index, row in vectors_df.iterrows()
    }


def _search_mapping_topk(library_mapping, output_embed_filename, sim_method, topk_candidate, name):
    query_names, query_vectors = _load_query_matrix(output_embed_filename)
    larger_is_better = sim_method == 'Cosine'
    accumulators = [_TopK(topk_candidate, larger_is_better=larger_is_better) for _ in query_names]
    keys, library_vectors = _items_to_matrix(library_mapping.items(), dtype=np.float32)
    with tqdm(
        total=len(keys),
        desc=f'{name}: custom-library similarity search',
        unit='compound',
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        if library_vectors.size:
            scores, indices = _faiss_search_batch(
                query_vectors,
                library_vectors,
                sim_method,
                topk_candidate,
            )
            _push_faiss_results(accumulators, keys, scores, indices)
        progress.update(len(keys))

    print(f"{name}: Searched from {len(keys)} candidates...")
    ordered_by_query = [accumulator.ordered() for accumulator in accumulators]
    max_len = max((len(rows) for rows in ordered_by_query), default=0)
    similarity = np.full((len(ordered_by_query), max_len), np.nan, dtype=np.float32)
    index = np.full((len(ordered_by_query), max_len), -1, dtype=np.int64)

    result_df_list = []
    for row_idx, rows in enumerate(ordered_by_query):
        keys = [key for key, _ in rows]
        scores = [score for _, score in rows]
        similarity[row_idx, :len(scores)] = scores
        index[row_idx, :len(keys)] = np.arange(len(keys), dtype=np.int64)
        result_df_list.append(pd.DataFrame(index=[str(key) for key in keys]))

    return similarity, index, result_df_list, ordered_by_query


def _search_pickle_files_topk(
    pickle_files,
    output_embed_filename,
    sim_method,
    topk_candidate,
    name,
    per_file=False,
):
    query_names, query_vectors = _load_query_matrix(output_embed_filename)
    larger_is_better = sim_method == 'Cosine'
    accumulators = [_TopK(topk_candidate, larger_is_better=larger_is_better) for _ in query_names]
    ordered_by_query = [[] for _ in query_names] if per_file else None
    total = 0

    for pickle_file in tqdm(
        pickle_files,
        total=len(pickle_files),
        desc=f"{name}: embedding-file search",
        unit='file',
        dynamic_ncols=True,
    ):
        with open(pickle_file, "rb") as f:
            embedding_vectors_dict = pickle.load(f)
        try:
            keys, library_vectors = _items_to_matrix(
                embedding_vectors_dict.items(),
                dtype=np.float32,
            )
            if library_vectors.size:
                scores, indices = _faiss_search_batch(
                    query_vectors,
                    library_vectors,
                    sim_method,
                    topk_candidate,
                )
                total += len(keys)
                _push_faiss_results(accumulators, keys, scores, indices)
            if per_file:
                for query_idx, accumulator in enumerate(accumulators):
                    ordered_by_query[query_idx].extend(accumulator.ordered())
                accumulators = [
                    _TopK(topk_candidate, larger_is_better=larger_is_better)
                    for _ in query_names
                ]
        finally:
            try:
                del library_vectors
            except UnboundLocalError:
                pass
            del embedding_vectors_dict
            cleanup_runtime()

    print(f"{name}: Searched from {total} candidates...")
    if not per_file:
        ordered_by_query = [accumulator.ordered() for accumulator in accumulators]
    max_len = max((len(rows) for rows in ordered_by_query), default=0)
    similarity = np.full((len(ordered_by_query), max_len), np.nan, dtype=np.float32)
    index = np.full((len(ordered_by_query), max_len), -1, dtype=np.int64)

    result_df_list = []
    for row_idx, rows in enumerate(ordered_by_query):
        keys = [key for key, _ in rows]
        scores = [score for _, score in rows]
        similarity[row_idx, :len(scores)] = scores
        index[row_idx, :len(keys)] = np.arange(len(keys), dtype=np.int64)
        result_df_list.append(pd.DataFrame(index=[str(key) for key in keys]))

    return similarity, index, result_df_list, ordered_by_query

######## Check the conversion of smiles to RDKit mol ########
def check_smiles(row):
    try:
        if Chem.MolFromSmiles(row['compound_smiles']) is None:
            raise ValueError(f"Compound {row['compound_name']} cannot be converted to an RDKit Mol object.")
    except Exception as e:
        print(e)
        raise

######## Extract smiles ########
def extract_smiles(html_string):
    match = re.search(r'#query=([^&]+)&', html_string)
    if match:
        return match.group(1)
    return html_string


def prioritize_scatter_traces(
    fig,
    background_trace_names,
    highlight_trace_names,
    background_opacity=0.15,
    highlight_opacity=0.9,
):
    """Draw library traces first and query/top-k traces last in Plotly 3D."""
    background_names = {str(name) for name in background_trace_names}
    highlight_names = {str(name) for name in highlight_trace_names}
    background_traces = []
    regular_traces = []
    highlight_traces = []

    for trace in fig.data:
        marker_size = trace.marker.size
        if marker_size is not None and not np.isscalar(marker_size):
            marker_size_array = np.asarray(marker_size)
            if marker_size_array.size and np.all(marker_size_array == marker_size_array.flat[0]):
                # Plotly Express emits one identical size value per point.  A
                # scalar renders and hovers identically without serializing a
                # ~100k-element array into the Appyter result notebook.
                trace.marker.size = marker_size_array.flat[0].item()

        trace_name = str(trace.name)
        if trace_name in background_names:
            trace.marker.opacity = background_opacity
            background_traces.append(trace)
        elif trace_name in highlight_names:
            trace.marker.opacity = highlight_opacity
            highlight_traces.append(trace)
        else:
            regular_traces.append(trace)

    # Plotly renders later traces after earlier ones.  The 3D depth buffer still
    # applies, but this makes coincident/overlapping highlighted points visible.
    fig.data = tuple(background_traces + regular_traces + highlight_traces)
    return fig

######## Toggle ########
def create_toggle(toggle_text, content):
    return HTML(f"""
    <style>
        .toggle-button {{
            display: inline-flex;
            align-items: center;
            cursor: pointer;
            font-size: 16px;
            user-select: none;
            margin-top: 10px;

        }}
        .toggle-button::before {{
            content: "▶";
            display: inline-block;
            margin-right: 5px;
            transform: rotate(0deg);
            transition: transform 0.3s ease;
        }}
        .toggle-button[aria-expanded="true"]::before {{
            transform: rotate(90deg);
        }}
    </style>
    <div class="toggle-button" onclick="let content=document.getElementById('{toggle_text}'); content.style.display = content.style.display == 'none' ? 'block' : 'none'; this.setAttribute('aria-expanded', content.style.display == 'block');">
    {toggle_text}
    </div>
    <div id="{toggle_text}" style="display:none; margin-top: 10px;">
        {content}
    </div>
    """)
    
def section_create_toggle(toggle_id, toggle_text, content):
    return f"""
    <style>
        #{toggle_id}-button {{
            display: inline-flex;
            align-items: center;
            cursor: pointer;
            font-size: 1.5em;
            font-weight: bold;
            user-select: none;
        }}
        #{toggle_id}-button::before {{
            content: "▶";
            display: inline-block;
            margin-right: 5px;
            transition: transform 0.3s ease;
        }}
        #{toggle_id}-button[aria-expanded="true"]::before {{
            transform: rotate(90deg);
        }}
        #{toggle_id}-content {{
            display: none;
            margin-top: 10px;
        }}
    </style>

    <div id="{toggle_id}-button" onclick="let content=document.getElementById('{toggle_id}-content'); content.style.display = content.style.display == 'none' ? 'block' : 'none'; this.setAttribute('aria-expanded', content.style.display == 'block');">
        {toggle_text}
    </div>
    <div id="{toggle_id}-content">
        {content}
    </div>
    """

######## Extract library dataframe and index ########
def pubchem_library_npl_chunk(file_path):
    library_df = pd.read_csv(file_path, sep='\t')
    library_df.rename(columns={'Name': 'drug2_name', 'SMILES': 'drug2_smiles'}, inplace=True)
    library_npl = np.arange(len(library_df))
    return library_df, library_npl


def library_npl(input_db):
    if input_db == 'kcb':
        df = pd.read_csv(library_path('kcb.csv'), index_col=0)
    elif input_db == 'kcb_cn':
        df = pd.read_csv(library_path('kcb_common_name.tsv'), sep='\t', index_col=0)
    elif input_db == 'zinc':
        df = pd.read_csv(library_path('ZINC_named+waited.tsv'), sep='\t', index_col=0)
    elif input_db == 'mce':
        df = pd.read_csv(library_path('MCE_library.tsv'), sep='\t', index_col=0)
    elif input_db == 'selleck':
        df = pd.read_csv(library_path('Selleckchem_library.tsv'), sep='\t', index_col=0)
    elif input_db == 'pubchem':
        return None, None 
    elif input_db == 'pubchem_cn':
        df = pd.read_csv(library_path('pubchem_common_name.tsv'), sep='\t')
    elif input_db == 'chembl':
        df = pd.read_csv(library_path('chembl.tsv'), sep='\t')
    elif input_db == 'chembl_cn':
        df = pd.read_csv(library_path('chembl_common_name.tsv'), sep='\t')
    else:
        raise ValueError("Invalid input_db value")

    if df is not None:
        df.rename(columns={'Name': 'drug2_name', 'SMILES': 'drug2_smiles'}, inplace=True)
        df1 = df['drug2_name']
        library_L = list(range(len(df1)))
        npl = np.array(library_L)
        return df, npl
    else:
        return None, None

def custom_npl(custom_df):
    custom_df.rename(columns={'compound_name': 'drug_name', 'compound_smiles': 'drug_smiles'}, inplace=True)
    custom_df1 = custom_df['drug_name']
    library_L = []
    
    for i in range(len(custom_df1)):
        library_L.append(int(i))
        
    library_npl = np.array(library_L)
    return custom_df, library_npl

def pubchem_library_df():
    chunks = []
    for chunk_number in tqdm(
        range(1, 5),
        total=4,
        desc='PubChem: loading metadata',
        unit='file',
        dynamic_ncols=True,
    ):
        chunks.append(
            pd.read_csv(library_path(f'PubChem_chunk_{chunk_number}.tsv'), sep='\t')
        )

    with tqdm(
        total=1,
        desc='PubChem: combining metadata',
        unit='step',
        dynamic_ncols=True,
    ) as progress:
        library_df = pd.concat(chunks)
        progress.update(1)
    library_df = library_df.rename(columns={'Name':'drug2_name', 'SMILES':'drug2_smiles'})

    return library_df

######## Load the default library embedding vectors ########
def embed_vector_lib(input_db, embed_method, file_format='pickle'):
    base_path = library_path(f'{embed_method}_{input_db}')

    if input_db == 'pubchem':
        embedding_vectors = {}
        if embed_method == 'ReSimNet':
            for i in range(1, 5):
                with open(f'{base_path}_{i}/{embed_method}_{input_db}_{i}_7.pkl', 'rb') as f:
                    embedding_vectors.update(pickle.load(f))
        elif embed_method != 'MACAW':
            for i in range(1, 5):
                with open(f'{base_path}_{i}/{embed_method}_{input_db}_{i}.pkl', 'rb') as f:
                    embedding_vectors.update(pickle.load(f))
        else:
            with open(f'{base_path}/{embed_method}_{input_db}.pkl', 'rb') as f:
                embedding_vectors = pickle.load(f)
    else:
        if embed_method == 'ReSimNet' and input_db != 'pubchem':
            with open(f'{base_path}/{embed_method}_{input_db}_7.pkl', 'rb') as f:
                embedding_vectors = pickle.load(f)
        else:
            with open(f'{base_path}/{embed_method}_{input_db}.pkl', 'rb') as f:
                embedding_vectors = pickle.load(f)

    return embedding_vectors


######## Embedding methods ########
def smiles2fp(smilesstr):
    mol = Chem.MolFromSmiles(smilesstr)
    if mol is None:
        raise ValueError(f"SMILES cannot be converted to an RDKit Mol object: {smilesstr}")
    fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp_obj, arr)
    return arr

def ecfp(smiles_list):
    results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append(np.zeros((2048,), dtype=np.uint8))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        results.append(arr)
    return results

def maccskeys(smiles_list):
    results = []
    maccs_featurizer = dc.feat.MACCSKeysFingerprint()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = maccs_featurizer.featurize([mol])[0]
            results.append(fp)
        else:
            results.append(np.zeros((167,)))
    return results

def mol2vec(smiles_list):
    results = []
    model_path = methods_path('mol2vec', 'mol2vec_model_300dim.pkl')
    mol2vec_featurizer = dc.feat.Mol2VecFingerprint(model_path)
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            vec = mol2vec_featurizer.featurize([mol])[0]
            results.append(vec)
        else:
            results.append(np.zeros((300,)))
    return results

def custom_embedding(drug_dict, embed_method, output_path, output_file, batch_size=10):
    if embed_method == 'ECFP':
        featurizer = ecfp
    elif embed_method == 'MACCSKeys':
        featurizer = maccskeys
    elif embed_method == 'Mol2vec':
        featurizer = mol2vec
    else:
        raise ValueError(f"Unsupported embedding method for custom_embedding: {embed_method}")
        
    names_list, smiles_list = zip(*drug_dict.items())
    
    batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]

    max_workers = min(len(batches), int(os.environ.get('DREMB_MAX_WORKERS', os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batch_results = list(executor.map(featurizer, batches))
        
    embed_dict = {name: fp for batch, names in zip(batch_results, [names_list[i:i + batch_size] for i in range(0, len(names_list), batch_size)]) for name, fp in zip(names, batch)}
    
    with open(output_path + output_file,'wb') as f:
        pickle.dump(embed_dict, f)
    
    return embed_dict

def drug_embeddings(drug_dict):
    result_dict = dict()
    global model
    model = methods.moable.model.DrugEncoder()
    model.load_state_dict(torch.load(methods_path('moable', 'models', 'moable.pth')))
    model.to(device)
    model.eval()
    
    for key in drug_dict:
        smiles = drug_dict[key]
        ecfp = torch.from_numpy(smiles2fp(smiles)).to(device)
        ecfp = ecfp.reshape(1,-1)
        embedding = model(ecfp.float()).cpu().detach().numpy().flatten()
        magnitude = np.linalg.norm(embedding)
        embedding = embedding / magnitude
        result_dict[key] = embedding

    return result_dict

def pretrained_MACAW(input_db):
    with open(methods_path('MACAW', f'MACAW_{input_db}_pre.pkl'), 'rb') as model_file:
        mcw = pickle.load(model_file)
        
    return mcw

######## FAISS-based search (Jaccard similarity) & Create results ########
def jaccard_finder(input_db, embed_dict, embed_method, queries, topk_candidates):
    topk_similarities = {}
    if input_db == 'custom' and embed_method in ['ECFP', 'MACCSKeys']:
        library_ecfp = embed_dict
    elif input_db != 'custom' and embed_method in ['ECFP', 'MACCSKeys']:
        library_ecfp = None
    else:
        print('Jaccard similarity is only supported for ECFP and MACCSKeys')
        return topk_similarities

    try:
        accumulators = {
            query_name: _TopK(topk_candidates, larger_is_better=True)
            for query_name in queries
        }
        query_vectors = {query_name: _flatten_vector(query_ecfp) for query_name, query_ecfp in queries.items()}

        if input_db == 'custom':
            batch_iter = (
                _items_to_matrix(batch, dtype=np.uint8)
                for batch in _iter_item_batches(library_ecfp)
            )
            expected_total = len(library_ecfp)
        else:
            batch_iter = _iter_library_fingerprint_batches(
                input_db,
                embed_method,
                batch_size=DEFAULT_BATCH_SIZE,
            )
            cached = _load_fingerprint_cache(input_db, embed_method)
            expected_total = len(cached[1]) if cached is not None else None

        with tqdm(
            total=expected_total,
            desc=f'{embed_method}: Jaccard similarity search',
            unit='compound',
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            for keys, library_matrix in batch_iter:
                if library_matrix.size:
                    for query_name, query_vector in query_vectors.items():
                        scores = _jaccard_scores(query_vector, library_matrix)
                        accumulators[query_name].push_many(keys, scores)
                progress.update(len(keys))

        for query_name, accumulator in accumulators.items():
            topk_similarities[query_name] = dict(accumulator.ordered())
    finally:
        del library_ecfp
        cleanup_runtime()

    return topk_similarities

def create_result_dataframe(results):
    data = []
    for query_name, topk in results.items():
        for key, similarity in topk.items():
            data.append([query_name, key, similarity])
    
    columns = ["Query", "Library Compound", "Jaccard Similarity"]
    result_df = pd.DataFrame(data, columns=columns)
    
    return result_df

def jaccard_dataframes(input_db, custom_embed_dict, embed_method, embed_dict, topk_candidate):
    results = jaccard_finder(input_db, custom_embed_dict, embed_method, embed_dict, topk_candidate)
    dataframes = {}
    for query_name, topk in results.items():
        data = []
        for key, similarity in topk.items():
            data.append([key, similarity])
        columns = ["drug_name", "Jaccard Similarity"]
        dataframes[query_name] = pd.DataFrame(data, columns=columns)

    del results
    gc.collect()

    return dataframes


def pubchem_jaccard_finder(embed_method, queries, topk_candidates):
    started_at = time.perf_counter()
    topk_similarities = {}

    if embed_method not in ['ECFP', 'MACCSKeys']:
        print('Jaccard similarity is only supported for ECFP and MACCSKeys')
        return topk_similarities, pd.DataFrame()

    query_vectors = {query_name: _flatten_vector(query_ecfp) for query_name, query_ecfp in queries.items()}
    accumulators = {
        query_name: _TopK(topk_candidates, larger_is_better=True)
        for query_name in queries
    }
    current_source = None

    def flush_source():
        for query_name, accumulator in accumulators.items():
            local_topk = dict(accumulator.ordered())
            if query_name not in topk_similarities:
                topk_similarities[query_name] = local_topk
            else:
                topk_similarities[query_name].update(local_topk)

    cached = _load_fingerprint_cache('pubchem', embed_method)
    expected_total = len(cached[1]) if cached is not None else None
    batch_iter = _iter_library_fingerprint_batches(
        'pubchem',
        embed_method,
        batch_size=DEFAULT_BATCH_SIZE,
        with_source_index=True,
    )
    with tqdm(
        total=expected_total,
        desc=f'PubChem {embed_method}: Jaccard similarity search',
        unit='compound',
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for source_index, keys, library_matrix in batch_iter:
            if current_source is not None and source_index != current_source:
                flush_source()
                accumulators = {
                    query_name: _TopK(topk_candidates, larger_is_better=True)
                    for query_name in queries
                }
            current_source = source_index
            if library_matrix.size:
                for query_name, query_vector in query_vectors.items():
                    if library_matrix.shape[1] != len(query_vector):
                        continue
                    scores = _jaccard_scores(query_vector, library_matrix)
                    accumulators[query_name].push_many(keys, scores)
            progress.update(len(keys))

    if current_source is not None:
        flush_source()

    selected_keys = set()
    for topk in topk_similarities.values():
        selected_keys.update(topk)

    combined_embedding_vectors = _select_fingerprint_vectors_from_tables(
        'pubchem',
        embed_method,
        selected_keys,
    )

    print(
        f'PubChem {embed_method}: candidate preparation completed in '
        f'{_format_elapsed(time.perf_counter() - started_at)}'
    )

    return topk_similarities, combined_embedding_vectors

def pubchem_jaccard_dataframes(embed_method, queries, topk_candidate):
    results, combined_embedding_vectors = pubchem_jaccard_finder(embed_method, queries, topk_candidate)
    dataframes = {}
    for query_name, topk in results.items():
        data = []
        for key, similarity in topk.items():
            data.append([key, similarity])
        columns = ["drug_name", "Jaccard Similarity"]
        dataframes[query_name] = pd.DataFrame(data, columns=columns)

    del results
    gc.collect()

    return dataframes, combined_embedding_vectors


######## FAISS-based search (ReSimNet) ########
def resimnet_finder(input_db, npl, output_embed_filename, topk_candidate, name, resimnet_model):
    if resimnet_model == 'ReSimNet7':
        cached_result = _search_continuous_embedding_cache(
            input_db,
            'ReSimNet',
            output_embed_filename,
            'Cosine',
            topk_candidate,
            name,
        )
        if cached_result is not None:
            similarity, index, result_df_list, _ = cached_result
            return similarity, index, result_df_list

    embedding_vectors_directory = library_path(f"ReSimNet_{input_db}") + os.sep
    embedding_vectors_filenames = os.listdir(embedding_vectors_directory)
    result_df_list = list()

    try:
        faiss_index
        del faiss_index
    except:
        pass

    for i in range(10):        
        if resimnet_model != "All" and str(i) not in resimnet_model:
            continue
            
        model_filenames = [
            filename for filename in embedding_vectors_filenames if f'_{i}' in filename
        ]
        for embedding_vectors_filename in tqdm(
            model_filenames,
            total=len(model_filenames),
            desc=f'{name}: ReSimNet model {i} files',
            unit='file',
            dynamic_ncols=True,
        ):
            if f"_{i}" in embedding_vectors_filename: 
                with open(embedding_vectors_directory+embedding_vectors_filename, "rb") as f:
                    embedding_vectors_dict = pickle.load(f)
                    
                embedding_vectors_df = pd.DataFrame.from_dict(embedding_vectors_dict).T
                embedding_vectors = np.ascontiguousarray(np.float32(embedding_vectors_df.values))
                    
                faiss.normalize_L2(embedding_vectors)
            
                try:
                    faiss_index
                except:
                    faiss_index = faiss.IndexFlatIP(embedding_vectors.shape[1])
                    faiss_index = faiss.IndexIDMap2(faiss_index)
                    
                faiss_index.add_with_ids(embedding_vectors, npl)

        print(f"{name}: Searching from {faiss_index.ntotal} candidates...")

        with open(output_embed_filename, "rb") as f:
            query_embedding_vectors = pickle.load(f)

        query_embedding_vectors = np.ascontiguousarray(
            np.float32(pd.DataFrame.from_dict(query_embedding_vectors).values).T
        )
        faiss.normalize_L2(query_embedding_vectors)

        Similarity, Index = faiss_index.search(query_embedding_vectors, topk_candidate)
        
        embedding_df = embedding_vectors_df.reset_index()
        embedding_df = embedding_df['index']
        emb_list = list(embedding_df[Index[0]])
        result_tmp_df = pd.DataFrame(index=[str(x) for x in emb_list])
        result_df_list.append(result_tmp_df)
                
    return Similarity, Index, result_df_list


def pubchem_resimnet_finder(input_db, output_embed_filename, topk_candidate, name, resimnet_model):
    started_at = time.perf_counter()
    if resimnet_model == 'ReSimNet7':
        cached_result = _search_continuous_embedding_cache(
            input_db,
            'ReSimNet',
            output_embed_filename,
            'Cosine',
            topk_candidate,
            name,
            per_source=True,
        )
        if cached_result is not None:
            _, _, result_df_list, ordered_by_query = cached_result
            selected_keys = [key for rows in ordered_by_query for key, _ in rows]
            combined_embedding_vectors = _select_continuous_embedding_cache_vectors(
                input_db,
                'ReSimNet',
                selected_keys,
            )
            combined_library_df = pubchem_library_df()
            print(
                f'{name}: PubChem ReSimNet candidate preparation completed in '
                f'{_format_elapsed(time.perf_counter() - started_at)}'
            )
            return result_df_list, combined_library_df, combined_embedding_vectors

    result_df_list = []
    library_df_list = []
    embedding_vector_list = []

    for n in range(1, 5):
        library_df, npl = pubchem_library_npl_chunk(library_path(f'PubChem_chunk_{n}.tsv'))
        library_df_list.append(library_df)
        
        embedding_vectors_directory = library_path(f'ReSimNet_{input_db}_{n}') + os.sep
        embedding_vectors_filenames = os.listdir(embedding_vectors_directory)
        
        try:
            faiss_index
            del faiss_index
        except:
            pass

        for i in range(10):
            if resimnet_model != "All" and str(i) not in resimnet_model:
                continue

            model_filenames = [
                filename for filename in embedding_vectors_filenames if f'_{i}' in filename
            ]
            for embedding_vectors_filename in tqdm(
                model_filenames,
                total=len(model_filenames),
                desc=f'{name}: PubChem chunk {n} ReSimNet model {i}',
                unit='file',
                dynamic_ncols=True,
            ):
                if f"_{i}" in embedding_vectors_filename:
                    with open(embedding_vectors_directory + embedding_vectors_filename, "rb") as f:
                        embedding_vectors_dict = pickle.load(f)

                    embedding_vectors_df = pd.DataFrame.from_dict(embedding_vectors_dict).T
                    embedding_vectors = np.ascontiguousarray(np.float32(embedding_vectors_df.values))

                    faiss.normalize_L2(embedding_vectors)

                    try:
                        faiss_index
                    except:
                        faiss_index = faiss.IndexFlatIP(embedding_vectors.shape[1])
                        faiss_index = faiss.IndexIDMap2(faiss_index)

                    faiss_index.add_with_ids(embedding_vectors, npl)

            print(f"chunk {n}; {name}: Searching from {faiss_index.ntotal} candidates...")

            with open(output_embed_filename, "rb") as f:
                query_embedding_vectors = pickle.load(f)

            query_embedding_vectors = np.ascontiguousarray(
                np.float32(pd.DataFrame.from_dict(query_embedding_vectors).values).T
            )
            faiss.normalize_L2(query_embedding_vectors)

            Similarity, Index = faiss_index.search(query_embedding_vectors, topk_candidate)

            embedding_df = embedding_vectors_df.reset_index()
            embedding_df = embedding_df['index']
            emb_list = list(embedding_df[Index[0]])
            result_tmp_df = pd.DataFrame(index=[str(x) for x in emb_list])
            result_df_list.append(result_tmp_df)

            # Extract embedding vectors for the result indices
            embedding_vectors_for_result = embedding_vectors_df.loc[emb_list]
            embedding_vector_list.append(embedding_vectors_for_result)

    # Merge all library_df into a single DataFrame
    combined_library_df = pd.concat(library_df_list, axis=0)
    combined_embedding_vectors = pd.concat(embedding_vector_list, axis=0)

    print(
        f'{name}: PubChem ReSimNet candidate preparation completed in '
        f'{_format_elapsed(time.perf_counter() - started_at)}'
    )

    return result_df_list, combined_library_df, combined_embedding_vectors



######## FAISS-based search (Custom) ########
def custom_finder(custom_dict, embed_method, npl, sim_method, output_embed_filename, topk_candidate, name):
    similarity, index, result_df_list, _ = _search_mapping_topk(
        custom_dict,
        output_embed_filename,
        sim_method,
        topk_candidate,
        name,
    )
    return similarity, index, result_df_list


######## FAISS-based search (Methods except ReSimNet) ########
def pubchem_chunks_search(input_db, embed_method, sim_method, output_embed_filename, topk_candidate, name):
    started_at = time.perf_counter()
    if embed_method in ['ECFP', 'MACCSKeys']:
        _, _, _, ordered_by_query = _search_fingerprints_from_tables(
            input_db,
            embed_method,
            output_embed_filename,
            sim_method,
            topk_candidate,
            name,
            per_source=True,
        )
        ordered = ordered_by_query[0] if ordered_by_query else []
        combined_results = pd.DataFrame({
            'drug2_name': [key for key, _ in ordered],
            'similarity': [score for _, score in ordered],
            'index': list(range(len(ordered))),
        })
        combined_embedding_vectors = _select_fingerprint_vectors_from_tables(
            input_db,
            embed_method,
            [key for key, _ in ordered],
        )
        print(
            f'{name}: PubChem {embed_method} candidate preparation completed in '
            f'{_format_elapsed(time.perf_counter() - started_at)}'
        )
        return combined_results, combined_embedding_vectors

    cached_result = _search_continuous_embedding_cache(
        input_db,
        embed_method,
        output_embed_filename,
        sim_method,
        topk_candidate,
        name,
        per_source=True,
    )
    if cached_result is not None:
        _, _, _, ordered_by_query = cached_result
        ordered = ordered_by_query[0] if ordered_by_query else []
        combined_results = pd.DataFrame({
            'drug2_name': [key for key, _ in ordered],
            'similarity': [score for _, score in ordered],
            'index': list(range(len(ordered))),
        })
        combined_embedding_vectors = _select_continuous_embedding_cache_vectors(
            input_db,
            embed_method,
            [key for key, _ in ordered],
        )
        print(
            f'{name}: PubChem {embed_method} candidate preparation completed in '
            f'{_format_elapsed(time.perf_counter() - started_at)}'
        )
        return combined_results, combined_embedding_vectors

    embedding_vectors_directory = library_path(f'{embed_method}_{input_db}')
    chunk_embed_files = [
        os.path.join(embedding_vectors_directory, f'{embed_method}_{input_db}_{n}.pkl')
        for n in range(1, 5)
    ]
    _, _, _, ordered_by_query = _search_pickle_files_topk(
        chunk_embed_files,
        output_embed_filename,
        sim_method,
        topk_candidate,
        name,
        per_file=True,
    )
    ordered = ordered_by_query[0] if ordered_by_query else []
    combined_results = pd.DataFrame({
        'drug2_name': [key for key, _ in ordered],
        'similarity': [score for _, score in ordered],
        'index': list(range(len(ordered))),
    })
    selected_keys = [key for key, _ in ordered]
    vector_dtype = np.uint8 if embed_method in ['ECFP', 'MACCSKeys'] else np.float32
    combined_embedding_vectors = _collect_vectors_from_pickle_files(
        chunk_embed_files,
        selected_keys,
        dtype=vector_dtype,
    )

    print(
        f'{name}: PubChem {embed_method} candidate preparation completed in '
        f'{_format_elapsed(time.perf_counter() - started_at)}'
    )

    return combined_results, combined_embedding_vectors


def finder(input_db, embed_method, npl, sim_method, output_embed_filename, topk_candidate, name):
    if embed_method in ['ECFP', 'MACCSKeys']:
        similarity, index, result_df_list, _ = _search_fingerprints_from_tables(
            input_db,
            embed_method,
            output_embed_filename,
            sim_method,
            topk_candidate,
            name,
        )
        return similarity, index, result_df_list

    cached_result = _search_continuous_embedding_cache(
        input_db,
        embed_method,
        output_embed_filename,
        sim_method,
        topk_candidate,
        name,
    )
    if cached_result is not None:
        similarity, index, result_df_list, _ = cached_result
        return similarity, index, result_df_list

    embedding_vectors_directory = library_path(f"{embed_method}_{input_db}")
    embedding_vectors_filenames = sorted(
        filename for filename in os.listdir(embedding_vectors_directory)
        if filename.endswith('.pkl')
    )
    pickle_files = [os.path.join(embedding_vectors_directory, filename) for filename in embedding_vectors_filenames]
    similarity, index, result_df_list, _ = _search_pickle_files_topk(
        pickle_files,
        output_embed_filename,
        sim_method,
        topk_candidate,
        name,
    )
    return similarity, index, result_df_list

def MA_finder(input_db, embed_method, npl, sim_method, output_embed_filename, topk_candidate, name):
    if embed_method in ['ECFP', 'MACCSKeys']:
        similarity, index, result_df_list, _ = _search_fingerprints_from_tables(
            input_db,
            embed_method,
            output_embed_filename,
            sim_method,
            topk_candidate,
            name,
        )
        return similarity, index, result_df_list

    cached_result = _search_continuous_embedding_cache(
        input_db,
        embed_method,
        output_embed_filename,
        sim_method,
        topk_candidate,
        name,
    )
    if cached_result is not None:
        similarity, index, result_df_list, _ = cached_result
        return similarity, index, result_df_list

    embedding_vectors_directory = library_path(f"{embed_method}_{input_db}")
    embedding_vectors_filenames = sorted(
        filename for filename in os.listdir(embedding_vectors_directory)
        if filename.endswith('.pkl')
    )
    pickle_files = [os.path.join(embedding_vectors_directory, filename) for filename in embedding_vectors_filenames]
    similarity, index, result_df_list, _ = _search_pickle_files_topk(
        pickle_files,
        output_embed_filename,
        sim_method,
        topk_candidate,
        name,
    )
    return similarity, index, result_df_list


######## Make results URL ########
def make_clickable(smiles, site="pubchem"):
    if site == "pubchem":
        url = f"https://pubchem.ncbi.nlm.nih.gov/#query={smiles}&input_type=smiles"
    else:
        url = f"https://zinc15.docking.org/substances/{smiles}/"
    return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(url,smiles)

def create_download_link(output_embed_filename):  
    html = "<a href=\"./{}\" target='_blank'>{}</a>".format(output_embed_filename, f"Download ReSimNet {name} embedding vectors")
    return HTML(html)


######## Calculate similarity or distance ########
def calculate_cosine_similarity(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    normalized_vectors = _normalize_rows(vectors.copy())
    return np.dot(normalized_vectors, normalized_vectors.T)

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def jaccard_similarity(v1, v2):
    intersection = np.logical_and(v1, v2)
    union = np.logical_or(v1, v2)
    union_sum = union.sum()
    if union_sum == 0:
        return 1.0
    return intersection.sum() / union_sum


######## UpSet Plot ########
def find_duplicate_names(dictionary):
    name_to_keys = {}

    for key, value in dictionary.items():
        for name in value:
            if name in name_to_keys:
                name_to_keys[name].append(key)
            else:
                name_to_keys[name] = [key]

    result_dict = {name: keys for name, keys in name_to_keys.items() if len(keys) > 1}

    return result_dict
