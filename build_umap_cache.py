"""Precompute query-independent library projections for pretrained UMAP models."""

from __future__ import annotations

import argparse
import gc
import os
import time

from joblib import load
import numpy as np

import utils


LIBRARIES = ['selleck', 'mce', 'zinc', 'kcb_cn', 'chembl_cn']
EMBEDDINGS = ['ReSimNet', 'MoAble', 'ECFP', 'MACCSKeys', 'Mol2vec', 'MACAW']


def parse_csv(value, allowed):
    if value == 'all':
        return list(allowed)
    requested = [part.strip() for part in value.split(',') if part.strip()]
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise SystemExit(f'Unknown values: {unknown}')
    return requested


def model_identity(model_path):
    stat = os.stat(model_path)
    return (
        f'pretrained:{os.path.basename(model_path)}:'
        f'{stat.st_size}:{stat.st_mtime_ns}'
    )


def build_one(input_db, embed_method, force=False):
    cached_vectors = (
        utils._load_fingerprint_cache(input_db, embed_method)
        if embed_method in {'ECFP', 'MACCSKeys'}
        else utils._load_continuous_embedding_cache(input_db, embed_method)
    )
    if cached_vectors is None:
        raise FileNotFoundError(
            f'A valid search cache is required for {input_db}/{embed_method}'
        )
    _, vectors, _ = cached_vectors

    model_path = utils.methods_path(
        'UMAP', f'UMAP_{input_db}_{embed_method}.joblib'
    )
    identity = model_identity(model_path)
    cache_key = utils.umap_library_projection_cache_key(
        vectors, input_db, embed_method, identity
    )
    existing = utils.load_umap_projection_cache(
        input_db, embed_method, 'library', cache_key, len(vectors), 0
    )
    if existing is not None and not force:
        print(f'SKIP {input_db}/{embed_method}: valid projection cache exists')
        return

    started = time.perf_counter()
    reducer = load(model_path)
    reducer.verbose = False
    projection = reducer.transform(vectors)
    saved = utils.save_umap_projection_cache(
        input_db,
        embed_method,
        'library',
        cache_key,
        projection,
        np.empty((0, 3), dtype=np.float32),
    )
    if not saved:
        raise OSError(f'Could not save the UMAP cache for {input_db}/{embed_method}')
    verified = utils.load_umap_projection_cache(
        input_db, embed_method, 'library', cache_key, len(vectors), 0
    )
    if verified is None:
        raise ValueError(f'Invalid UMAP cache for {input_db}/{embed_method}')
    print(
        f'DONE {input_db}/{embed_method}: {len(vectors):,} rows in '
        f'{time.perf_counter() - started:.2f}s'
    )
    del reducer, projection, verified
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--libraries', default='all')
    parser.add_argument('--embeddings', default='all')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    libraries = parse_csv(args.libraries, LIBRARIES)
    embeddings = parse_csv(args.embeddings, EMBEDDINGS)
    for input_db in libraries:
        for embed_method in embeddings:
            build_one(input_db, embed_method, force=args.force)


if __name__ == '__main__':
    main()
