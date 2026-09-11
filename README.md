# Dr.Emb Appyter

Dr.Emb Appyter is a web platform for finding compounds that are close to query compounds in multiple chemical embedding spaces. It supports ReSimNet, MoAble, Mol2vec, MACAW, ECFP, and MACCS Keys, with exact FAISS search for cosine similarity and Euclidean distance and exact Jaccard search for fingerprint embeddings.

Paper: [Dr.Emb Appyter: A Web Platform for Drug Discovery using Embedding Vectors](https://doi.org/10.1002/jcc.27469)

Service: [https://dremb.korea.ac.kr](https://dremb.korea.ac.kr)

User guide: [Google Drive](https://drive.google.com/file/d/1_ljSSSSlcQ5HTlpctqH2Ks_WWTus-93v/view?usp=sharing)

## Version 0.2

Version 0.2 keeps the deployed compound order and scores while reducing peak memory use and startup/search time:

- memory-mapped caches for all 9 built-in libraries and all 6 embeddings;
- exact FAISS search for cosine similarity and Euclidean distance;
- batched exact Jaccard search for ECFP and MACCS Keys;
- reusable UMAP projection cache keyed by the exact library/query input;
- Top K default of 30;
- quieter Appyter output, accurate progress reporting, and foreground query/result points in 3D plots.

Valid similarity combinations are cosine and Euclidean for every embedding, plus Jaccard for ECFP and MACCS Keys. Single- and multiple-query inputs are supported. Large-library visualization preserves the established query-specific 100,000-candidate behavior.

## Required data

Large library vectors, model files, and generated caches are intentionally not stored in Git. Place the existing library files under `Library/` and model files under `methods/`.

The separately distributed cache archives are named:

```text
dremb-search-cache-0.2.tar.zst
dremb-search-cache-0.2.tar.zst.sha256
dremb-umap-cache-0.2.tar.zst
dremb-umap-cache-0.2.tar.zst.sha256
```

The commands below assume that the current directory is the application repository
root: the directory containing this `README.md`, `dr_emb.ipynb`, and `Library/`.
Copy all four downloaded files into that directory first. For example, when they
were downloaded to `~/Downloads`:

```bash
cd /path/to/The-Dr.Emb-Appyter
cp ~/Downloads/dremb-search-cache-0.2.tar.zst* .
cp ~/Downloads/dremb-umap-cache-0.2.tar.zst* .
```

Verify the downloads before extracting them:

```bash
sha256sum -c dremb-search-cache-0.2.tar.zst.sha256
sha256sum -c dremb-umap-cache-0.2.tar.zst.sha256
```

Both commands must report `OK`. Then extract the archives from the same
application repository root:

```bash
tar --zstd -xf dremb-search-cache-0.2.tar.zst
tar --zstd -xf dremb-umap-cache-0.2.tar.zst
```

Do not extract either archive inside `Library/`; each archive already contains its
own leading `Library/` directory. The resulting layout must be:

```text
The-Dr.Emb-Appyter/
├── dr_emb.ipynb
├── Library/
│   ├── .dremb_fingerprint_cache/
│   │   ├── ECFP_chembl.npy
│   │   ├── ECFP_chembl.json
│   │   └── ...
│   ├── .dremb_umap_cache/
│   │   └── ...
│   └── ... existing library files ...
└── methods/
    └── ... existing model files ...
```

Check that the cache directories are populated:

```bash
find Library/.dremb_fingerprint_cache -maxdepth 1 -type f | wc -l
find Library/.dremb_umap_cache -maxdepth 1 -type f | wc -l
```

The search archive restores 162 files: 54 metadata files, 54 vector arrays, and
54 name lists. The precomputed UMAP archive contains 30 library-projection
caches: five pretrained libraries times all six embedding methods. Additional
query-dependent UMAP entries are generated automatically after a cache miss, and
an identical rerun loads the saved projection.

On the production server, its installation-specific Compose configuration
bind-mounts the application repository's `Library/` directory as `/app/Library`.
Consequently, the two cache directories are discovered automatically in the
container and must not be copied into the Docker image. Production Compose,
nginx, certificate, and orchestrator configuration is managed separately from
this application repository.

If `tar` reports that the `zstd` program is missing, install the `zstd` package
with the operating system package manager and run the extraction commands again.
Search caches are portable because their source identities are stored relative to
`Library/`.

If the archive is unavailable, rebuild all search caches from the deployed pickle files without changing vector values or insertion order:

```bash
python build_fingerprint_cache.py \
  --libraries all \
  --embeddings all \
  --source pickle
```

Then precompute the query-independent UMAP library projections used by all six embeddings for Selleckchem, MCE, ZINC20, KCB common-name, and ChEMBL common-name libraries:

```bash
python build_umap_cache.py --libraries all --embeddings all
```

## Run locally

Install the dependencies and start the canonical notebook:

```bash
pip install --no-cache-dir -r requirements.txt
appyter dr_emb.ipynb --extras=toggle-code --extras=toc --extras=hide-code
```

Paths are resolved relative to the repository by default. Optional `DREMB_APP_ROOT`, `DREMB_LIBRARY_ROOT`, `DREMB_METHODS_ROOT`, `DREMB_FINGERPRINT_CACHE_ROOT`, and `DREMB_UMAP_CACHE_ROOT` environment variables can override them.

## Build Docker image 0.2

The code-only 0.2 image intentionally layers on the preserved 0.1 dependency image:

```bash
docker image inspect dremb-deploy:0.1
docker build -f Dockerfile.0.2 -t dremb-deploy:0.2 .
```

Library vectors and models remain bind-mounted at runtime and are not copied into the image. The image serves `dr_emb.ipynb`; no `_renew` filename is required.

## Production deployment and rollback

The existing Dr.Emb server keeps its installation-specific Compose and rollback
files outside this application repository. Its service/container name remains
`dremb-deploy`, so nginx, certbot, and the Appyter orchestrator continue to use
the same endpoint. From that server's deployment directory, recreate only the
application container and reload nginx:

```bash
sudo docker compose build dremb-deploy
sudo docker compose up -d --no-deps dremb-deploy
sudo docker exec nginx nginx -s reload
```

The `dremb-deploy:0.1` image is not overwritten or removed. On the existing
server, roll back with its locally maintained Compose overlay:

```bash
sudo docker compose \
  -f docker-compose.yml \
  -f docker-compose.rollback-0.1.yml \
  up -d --no-deps --no-build dremb-deploy
sudo docker exec nginx nginx -s reload
```

Confirm the running image after either operation:

```bash
sudo docker inspect dremb-deploy \
  --format 'image={{.Config.Image}} image_id={{.Image}} status={{.State.Status}}'
```

## Inputs and outputs

Inputs are a built-in or custom library, one or more query compounds, Top K, an embedding method, and a similarity method. Results include the ranked candidate table, a 3D UMAP scatter plot, an UpSet plot, a query-similarity heatmap, and drug-set enrichment analysis.

Example input files are available at `static/antiviral_drugs.txt` and `static/custom_library.tsv`.

## License

See [LICENSE](LICENSE).
