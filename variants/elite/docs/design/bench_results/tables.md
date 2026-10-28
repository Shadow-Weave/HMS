### T1 sqlite, max_candidates=2000 (shipped default)

| N | query | cands | selects | total ms | axis SQL | hydrate | index.search | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | entity | 30 | 1 | 1.65 | 0.24 | 0.45 | 0.13 | 0.38 | 0.03 | 0.01 | 0.04 | 0.00 | 0.02 | 0.03 | 0.03 | 0.05 | 0.24 |
| 300 | identifier | 292 | 1 | 12.93 | 0.87 | 4.49 | 1.17 | 3.19 | 0.49 | 0.07 | 0.34 | 0.03 | 0.18 | 0.24 | 0.03 | 0.07 | 1.76 |
| 300 | month | 16 | 1 | 0.86 | 0.10 | 0.24 | 0.07 | 0.18 | 0.02 | 0.00 | 0.02 | 0.00 | 0.01 | 0.01 | 0.02 | 0.05 | 0.14 |
| 300 | preference | 292 | 4 | 12.39 | 0.99 | 4.27 | 1.12 | 3.13 | 0.52 | 0.00 | 0.34 | 0.03 | 0.16 | 0.23 | 0.03 | 0.06 | 1.51 |
| 300 | cjk | 292 | 1 | 11.99 | 0.81 | 4.27 | 1.13 | 3.12 | 0.46 | 0.00 | 0.33 | 0.03 | 0.16 | 0.23 | 0.03 | 0.04 | 1.39 |
| 3000 | entity | 300 | 1 | 14.95 | 2.29 | 4.63 | 1.33 | 3.71 | 0.27 | 0.14 | 0.34 | 0.03 | 0.18 | 0.36 | 0.03 | 0.07 | 1.56 |
| 3000 | identifier | 2000 | 1 | 89.47 | 7.27 | 30.82 | 8.78 | 21.47 | 3.35 | 0.44 | 2.26 | 0.20 | 1.18 | 1.69 | 0.04 | 0.10 | 11.88 |
| 3000 | month | 166 | 1 | 7.70 | 0.97 | 2.55 | 0.74 | 1.80 | 0.23 | 0.00 | 0.19 | 0.02 | 0.09 | 0.14 | 0.03 | 0.07 | 0.87 |
| 3000 | preference | 2000 | 4 | 89.52 | 7.93 | 30.86 | 8.86 | 21.70 | 3.50 | 0.00 | 2.29 | 0.21 | 1.08 | 1.78 | 0.04 | 0.08 | 11.19 |
| 3000 | cjk | 2000 | 1 | 88.32 | 7.34 | 30.97 | 8.77 | 21.63 | 3.12 | 0.00 | 2.28 | 0.20 | 1.07 | 1.72 | 0.04 | 0.06 | 11.11 |
| 30000 | entity | 2000 | 1 | 119.85 | 26.86 | 35.94 | 11.01 | 25.23 | 1.98 | 0.89 | 2.28 | 0.20 | 1.19 | 1.81 | 0.04 | 0.16 | 12.27 |
| 30000 | identifier | 2000 | 1 | 110.02 | 20.96 | 35.03 | 10.66 | 21.85 | 3.27 | 0.45 | 2.26 | 0.20 | 1.19 | 1.73 | 0.04 | 0.14 | 12.23 |
| 30000 | month | 1666 | 1 | 83.19 | 10.54 | 29.27 | 8.93 | 18.08 | 2.22 | 0.00 | 1.90 | 0.17 | 0.89 | 1.73 | 0.04 | 0.09 | 9.32 |
| 30000 | preference | 2000 | 4 | 122.15 | 34.33 | 34.98 | 10.63 | 21.71 | 3.62 | 0.00 | 2.29 | 0.20 | 1.07 | 1.78 | 0.04 | 0.08 | 11.42 |
| 30000 | cjk | 2000 | 1 | 108.49 | 21.03 | 34.74 | 10.81 | 21.72 | 3.22 | 0.00 | 2.29 | 0.21 | 1.07 | 1.75 | 0.04 | 0.06 | 11.54 |

### T2 per-candidate cost (µs / hydrated candidate), sqlite capped run

| N | query | cands | axis SQL | hydrate | index.search | kw tokens | kw overlap | recency | fuse | explain | sum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | entity | 30 | 8.0 | 15.1 | 4.3 | 12.7 | 1.0 | 1.2 | 0.6 | 0.9 | 43.8 |
| 300 | identifier | 292 | 3.0 | 15.4 | 4.0 | 10.9 | 1.7 | 1.2 | 0.6 | 0.8 | 37.5 |
| 300 | month | 16 | 6.4 | 14.8 | 4.3 | 11.2 | 1.4 | 1.2 | 0.6 | 0.9 | 40.9 |
| 300 | preference | 292 | 3.4 | 14.6 | 3.8 | 10.7 | 1.8 | 1.1 | 0.5 | 0.8 | 36.9 |
| 300 | cjk | 292 | 2.8 | 14.6 | 3.9 | 10.7 | 1.6 | 1.1 | 0.5 | 0.8 | 36.0 |
| 3000 | entity | 300 | 7.6 | 15.4 | 4.4 | 12.4 | 0.9 | 1.1 | 0.6 | 1.2 | 43.7 |
| 3000 | identifier | 2000 | 3.6 | 15.4 | 4.4 | 10.7 | 1.7 | 1.1 | 0.6 | 0.8 | 38.4 |
| 3000 | month | 166 | 5.9 | 15.4 | 4.5 | 10.8 | 1.4 | 1.2 | 0.5 | 0.8 | 40.4 |
| 3000 | preference | 2000 | 4.0 | 15.4 | 4.4 | 10.8 | 1.8 | 1.1 | 0.5 | 0.9 | 39.0 |
| 3000 | cjk | 2000 | 3.7 | 15.5 | 4.4 | 10.8 | 1.6 | 1.1 | 0.5 | 0.9 | 38.5 |
| 30000 | entity | 2000 | 13.4 | 18.0 | 5.5 | 12.6 | 1.0 | 1.1 | 0.6 | 0.9 | 53.1 |
| 30000 | identifier | 2000 | 10.5 | 17.5 | 5.3 | 10.9 | 1.6 | 1.1 | 0.6 | 0.9 | 48.5 |
| 30000 | month | 1666 | 6.3 | 17.6 | 5.4 | 10.9 | 1.3 | 1.1 | 0.5 | 1.0 | 44.2 |
| 30000 | preference | 2000 | 17.2 | 17.5 | 5.3 | 10.9 | 1.8 | 1.1 | 0.5 | 0.9 | 55.2 |
| 30000 | cjk | 2000 | 10.5 | 17.4 | 5.4 | 10.9 | 1.6 | 1.1 | 0.5 | 0.9 | 48.3 |

### T3 sqlite, max_candidates=1000000 (true O(N) cost, no truncation)

| N | query | cands | selects | total ms | axis SQL | hydrate | index.search | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | entity | 30 | 1 | 1.72 | 0.25 | 0.46 | 0.13 | 0.39 | 0.03 | 0.01 | 0.04 | 0.00 | 0.02 | 0.03 | 0.03 | 0.05 | 0.28 |
| 300 | identifier | 292 | 1 | 12.45 | 0.83 | 4.39 | 1.17 | 3.19 | 0.48 | 0.07 | 0.34 | 0.03 | 0.17 | 0.24 | 0.03 | 0.07 | 1.43 |
| 300 | month | 16 | 1 | 0.89 | 0.10 | 0.24 | 0.07 | 0.18 | 0.02 | 0.00 | 0.02 | 0.00 | 0.01 | 0.01 | 0.02 | 0.05 | 0.15 |
| 300 | preference | 292 | 4 | 12.46 | 0.99 | 4.29 | 1.14 | 3.17 | 0.52 | 0.00 | 0.34 | 0.03 | 0.16 | 0.24 | 0.03 | 0.06 | 1.50 |
| 300 | cjk | 292 | 1 | 12.05 | 0.81 | 4.28 | 1.14 | 3.16 | 0.46 | 0.00 | 0.34 | 0.03 | 0.16 | 0.23 | 0.03 | 0.04 | 1.37 |
| 3000 | entity | 300 | 1 | 15.06 | 2.29 | 4.64 | 1.36 | 3.78 | 0.27 | 0.13 | 0.34 | 0.03 | 0.18 | 0.36 | 0.03 | 0.07 | 1.56 |
| 3000 | identifier | 2917 | 1 | 132.92 | 10.14 | 46.03 | 12.95 | 32.29 | 4.93 | 0.65 | 3.35 | 0.31 | 1.73 | 2.58 | 0.04 | 0.12 | 17.80 |
| 3000 | month | 166 | 1 | 7.82 | 1.01 | 2.55 | 0.76 | 1.83 | 0.23 | 0.00 | 0.19 | 0.02 | 0.09 | 0.14 | 0.03 | 0.07 | 0.91 |
| 3000 | preference | 2917 | 4 | 132.23 | 10.67 | 46.01 | 12.85 | 32.12 | 5.25 | 0.00 | 3.31 | 0.30 | 1.58 | 3.59 | 0.04 | 0.08 | 16.42 |
| 3000 | cjk | 2917 | 1 | 130.89 | 10.25 | 46.30 | 12.64 | 31.91 | 4.68 | 0.00 | 3.32 | 0.30 | 1.58 | 2.60 | 0.04 | 0.06 | 17.22 |
| 30000 | entity | 3000 | 1 | 170.65 | 29.81 | 54.53 | 16.48 | 38.44 | 2.92 | 1.30 | 3.39 | 0.31 | 1.78 | 2.80 | 0.04 | 0.16 | 18.67 |
| 30000 | identifier | 29167 | 1 | 1612.90 | 145.20 | 580.45 | 163.80 | 320.04 | 50.16 | 6.33 | 32.48 | 3.04 | 17.19 | 30.78 | 0.05 | 0.17 | 263.19 |
| 30000 | month | 1666 | 1 | 83.67 | 10.70 | 29.20 | 8.87 | 18.46 | 2.25 | 0.00 | 1.88 | 0.17 | 0.90 | 1.69 | 0.04 | 0.09 | 9.41 |
| 30000 | preference | 29167 | 4 | 1640.51 | 158.66 | 602.36 | 181.59 | 318.97 | 54.37 | 0.00 | 32.58 | 3.07 | 15.71 | 67.33 | 0.05 | 0.10 | 205.73 |
| 30000 | cjk | 29167 | 1 | 1698.74 | 151.97 | 618.59 | 193.50 | 325.06 | 59.22 | 0.00 | 33.26 | 3.11 | 16.08 | 34.97 | 0.05 | 0.08 | 262.83 |

cProfile, N=30000, query `identifier` (one search; cumulative ms / own ms / calls):

| cum ms | own ms | calls | function |
|---|---|---|---|
| 3185.9 | 0.1 | 1 | `hms_elite/memory/main.py:442(search)` |
| 3185.7 | 0.0 | 1 | `hms_elite/mvmi/query.py:496(search)` |
| 3185.7 | 149.0 | 233363 | `bench/profile_search.py:122(wrapper)` |
| 3154.4 | 70.1 | 1 | `hms_elite/mvmi/query.py:224(execute_axis_query)` |
| 1477.5 | 77.8 | 29167 | `hms_elite/utils/scoring.py:255(compute_signals)` |
| 1125.5 | 0.0 | 1 | `hms_elite/mvmi/query.py:309(_select)` |
| 1125.5 | 2.9 | 1 | `hms_elite/memory/storage.py:2331(axis_select)` |
| 1122.6 | 1.9 | 1 | `bench/profile_search.py:142(select_atoms)` |
| 975.5 | 149.3 | 1 | `hms_elite/memory/storage.py:1703(_hydrate_atoms)` |
| 908.2 | 47.8 | 29167 | `hms_elite/utils/scoring.py:84(atom_keyword_tokens)` |
| 738.8 | 35.8 | 29168 | `hms_elite/utils/text.py:453(content_tokens)` |
| 426.4 | 7.8 | 58359 | `site-packages/pydantic/main.py:253(__init__)` |
| 418.6 | 172.2 | 58359 | `~:0(<method 'validate_python' of 'pydantic_core._pydantic_core.SchemaValidator' objects>)` |
| 406.9 | 128.2 | 64168 | `hms_elite/utils/text.py:397(tokenize)` |

### T4 milvus-lite (HNSW), N=3000, max_candidates=2000

| N | query | cands | selects | total ms | axis SQL | hydrate | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | expr build | ANN rpc+parse | rescore | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | entity | 300 | 1 | 60.27 | 2.75 | 5.32 | 4.29 | 0.32 | 0.16 | 0.39 | 0.04 | 0.20 | 0.27 | 0.04 | 0.04 | 43.51 | 0.79 | 0.12 | 2.05 |
| 3000 | identifier | 2000 | 1 | 218.74 | 8.30 | 35.46 | 24.45 | 3.83 | 0.51 | 2.57 | 0.23 | 1.29 | 1.89 | 0.04 | 0.29 | 119.99 | 5.51 | 0.14 | 14.24 |
| 3000 | month | 166 | 1 | 72.31 | 1.81 | 4.50 | 3.20 | 0.40 | 0.00 | 0.34 | 0.03 | 0.15 | 0.23 | 0.06 | 0.04 | 58.93 | 0.68 | 0.12 | 1.82 |
| 3000 | preference | 2000 | 4 | 223.49 | 9.20 | 35.87 | 24.94 | 4.10 | 0.00 | 2.57 | 0.23 | 1.21 | 2.05 | 0.04 | 0.29 | 117.98 | 5.53 | 0.09 | 19.39 |
| 3000 | cjk | 2000 | 1 | 225.87 | 8.68 | 36.55 | 25.29 | 3.83 | 0.00 | 2.65 | 0.24 | 1.23 | 1.90 | 0.05 | 0.30 | 122.95 | 5.71 | 0.07 | 16.43 |

### T5 milvus-lite (HNSW), N=3000, max_candidates=1000000

| N | query | cands | selects | total ms | axis SQL | hydrate | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | expr build | ANN rpc+parse | rescore | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | entity | 300 | 1 | 68.39 | 3.06 | 6.11 | 4.86 | 0.36 | 0.18 | 0.45 | 0.04 | 0.23 | 0.31 | 0.05 | 0.05 | 49.14 | 0.88 | 0.12 | 2.55 |
| 3000 | identifier | 2917 | 1 | 316.94 | 11.19 | 50.21 | 33.91 | 5.31 | 0.72 | 3.63 | 0.33 | 1.85 | 2.64 | 0.05 | 0.42 | 173.00 | 7.58 | 0.15 | 25.95 |
| 3000 | month | 166 | 1 | 47.67 | 1.26 | 2.88 | 2.00 | 0.24 | 0.00 | 0.22 | 0.02 | 0.10 | 0.14 | 0.04 | 0.03 | 39.12 | 0.42 | 0.08 | 1.11 |
| 3000 | preference | 2917 | 4 | 317.53 | 11.88 | 50.42 | 34.51 | 5.82 | 0.00 | 3.68 | 0.33 | 1.75 | 3.39 | 0.05 | 0.42 | 175.20 | 7.51 | 0.09 | 22.49 |
| 3000 | cjk | 2917 | 1 | 317.81 | 11.26 | 51.74 | 34.14 | 5.20 | 0.00 | 3.64 | 0.33 | 1.74 | 2.72 | 0.04 | 0.41 | 176.76 | 7.70 | 0.07 | 22.06 |

cProfile, N=3000, query `identifier` (one search; cumulative ms / own ms / calls):

| cum ms | own ms | calls | function |
|---|---|---|---|
| 578.4 | 0.1 | 2 | `hms_elite/memory/main.py:442(search)` |
| 578.4 | 17.2 | 23370 | `bench/profile_search.py:122(wrapper)` |
| 578.3 | 0.0 | 1 | `hms_elite/mvmi/query.py:496(search)` |
| 472.1 | 6.5 | 1 | `hms_elite/mvmi/query.py:224(execute_axis_query)` |
| 282.0 | 0.4 | 1 | `hms_elite/utils/scoring.py:112(semantic_scores)` |
| 279.8 | 0.0 | 1 | `hms_elite/index/milvus.py:500(search)` |
| 267.4 | 1.0 | 3 | `hms_elite/index/milvus.py:342(_ann)` |
| 226.7 | 0.0 | 3 | `site-packages/pymilvus/milvus_client/milvus_client.py:420(search)` |
| 226.7 | 0.0 | 3 | `site-packages/pymilvus/decorators.py:514(handler)` |
| 226.7 | 0.0 | 3 | `site-packages/pymilvus/decorators.py:560(handler)` |

### T6 milvus-lite scalar-filter ANN probe

milvus-lite, N=3000, limit=80, 20 runs, median ms:

| index | filter variant | expr bytes | rows matching | hits / recall gap | ids only | + vector payload | top_k=len(chunk) | top_k=len + vector |
|---|---|---|---|---|---|---|---|---|
| FLAT | no_filter | 0 | 3000 | 80 / 0 | 2.57 | 2.76 | - | - |
| FLAT | scope_validity | 139 | 2050 | 80 / 0 | 2.66 | 2.84 | - | - |
| FLAT | scope_validity_scale | 200 | 1174 | 80 / 0 | 2.69 | 2.87 | - | - |
| FLAT | scope_validity_scale_obs | 214 | 167 | 80 / 0 | 2.65 | 2.96 | - | - |
| FLAT | scope_validity_time_gate | 223 | 133 | 80 / 0 | 2.71 | 2.85 | - | - |
| FLAT | id_list_300 | 4511 | 300 | 80 / 0 | 3.15 | 3.23 | 7.469 | 8.17 |
| FLAT | id_list_1024 | 15371 | 1024 | 80 / 0 | 4.19 | 4.36 | 22.986 | 25.876 |
| FLAT | id_list_2000 | 30011 | 2000 | 80 / 0 | 5.68 | 6.01 | 24.36 | 28.158 |
| AUTOINDEX | no_filter | 0 | 3000 | 80 / 0 | 2.98 | 2.74 | - | - |
| AUTOINDEX | scope_validity | 139 | 2050 | 80 / 0 | 2.67 | 3.09 | - | - |
| AUTOINDEX | scope_validity_scale | 200 | 1174 | 80 / 0 | 2.87 | 2.94 | - | - |
| AUTOINDEX | scope_validity_scale_obs | 214 | 167 | 67 / 13 | 2.54 | 2.73 | - | - |
| AUTOINDEX | scope_validity_time_gate | 223 | 133 | 51 / 29 | 2.12 | 2.23 | - | - |
| AUTOINDEX | id_list_300 | 4511 | 300 | 80 / 0 | 3.12 | 3.33 | 3.967 | 4.374 |
| AUTOINDEX | id_list_1024 | 15371 | 1024 | 80 / 0 | 4.26 | 4.42 | 11.753 | 12.759 |
| AUTOINDEX | id_list_2000 | 30011 | 2000 | 80 / 0 | 5.92 | 6.11 | 21.845 | 24.202 |
| HNSW | no_filter | 0 | 3000 | 80 / 0 | 2.75 | 2.66 | - | - |
| HNSW | scope_validity | 139 | 2050 | 80 / 0 | 2.91 | 3.25 | - | - |
| HNSW | scope_validity_scale | 200 | 1174 | 80 / 0 | 3.06 | 3.44 | - | - |
| HNSW | scope_validity_scale_obs | 214 | 167 | 80 / 0 | 2.97 | 3.04 | - | - |
| HNSW | scope_validity_time_gate | 223 | 133 | 60 / 20 | 2.37 | 2.49 | - | - |
| HNSW | id_list_300 | 4511 | 300 | 80 / 0 | 3.14 | 3.40 | 7.063 | 7.897 |
| HNSW | id_list_1024 | 15371 | 1024 | 80 / 0 | 4.21 | 4.43 | 24.13 | 27.439 |
| HNSW | id_list_2000 | 30011 | 2000 | 80 / 0 | 5.84 | 6.12 | 25.547 | 28.648 |

expression draft: `user_id == "alice" and archived == false and valid_from <= "2026-06-15T09:00:00Z" and (valid_to == "" or valid_to > "2026-06-15T09:00:00Z")`; VARCHAR instant comparison count check: {'milvus_count': 2050, 'python_count': 2050, 'equal': True}

### T7 FTS5 keyword channel probe

FTS5 available: True (sqlite 3.53.2); K_kw = 30; 20 runs, median ms:

| N | base insert s | FTS build s | query | matching docs | FTS5 MATCH+bm25 LIMIT 30 ms | LIKE+time index LIMIT 30 ms |
|---|---|---|---|---|---|---|
| 300 | 0.002 | 0.0 | entity | 30 | 0.036 | 0.076 |
| 300 | 0.002 | 0.0 | identifier | 120 | 0.086 | 0.025 |
| 300 | 0.002 | 0.0 | month | 60 | 0.054 | 0.024 |
| 300 | 0.002 | 0.0 | preference | 60 | 0.059 | 0.059 |
| 300 | 0.002 | 0.0 | cjk | 36 | 0.040 | 0.067 |
| 3000 | 0.018 | 0.002 | entity | 300 | 0.202 | 0.132 |
| 3000 | 0.018 | 0.002 | identifier | 1200 | 0.741 | 0.028 |
| 3000 | 0.018 | 0.002 | month | 600 | 0.401 | 0.033 |
| 3000 | 0.018 | 0.002 | preference | 600 | 0.415 | 0.066 |
| 3000 | 0.018 | 0.002 | cjk | 360 | 0.254 | 0.073 |
| 30000 | 0.184 | 0.027 | entity | 3000 | 2.206 | 0.625 |
| 30000 | 0.184 | 0.027 | identifier | 12000 | 7.986 | 0.124 |
| 30000 | 0.184 | 0.027 | month | 6000 | 4.223 | 0.145 |
| 30000 | 0.184 | 0.027 | preference | 6000 | 4.297 | 0.161 |
| 30000 | 0.184 | 0.027 | cjk | 3600 | 2.697 | 0.127 |

tokenize sample: `Sarah patched src/billing/retry.ts 今天部署了支付服务` → `['sarah', 'patched', 'src/billing/retry.ts', '今天', '天部', '部署', '署了', '了支', '支付', '付服', '服务']`; identifier_tokens → `['src/billing/retry.ts']`

### T8 sqlite semantic channel prototype

sqlite semantic channel prototype, dims=64, numpy 2.4.6, median ms (whole scope gated in):

| N | current (load 500-chunks + per-row np.dot, top_k=N) | B: VALUES-CTE join + matmul + argpartition K=60/80/240 | D: B + exact np.dot rescore of the K | C: scope JOIN in SQL, K=60/80/240 | E: cached float64 matrix, K=60/80/240 | rows where matmul ≠ dot (max abs diff) | order differs | D bit-equal |
|---|---|---|---|---|---|---|---|---|
| 300 | 0.76 | 0.12/0.18/0.15 | 0.33/0.92/0.81 | 0.12/0.12/0.15 | 0.01/0.02/0.05 | 81 (5.6e-17) | False | True |
| 3000 | 8.56 | 1.21/1.21/1.22 | 1.37/1.42/1.98 | 1.11/1.14/1.15 | 0.04/0.04/0.07 | 828 (6.9e-17) | False | True |
| 30000 | 97.06 | 27.43/30.24/29.02 | 28.00/26.83/29.97 | 28.16/30.16/28.64 | 1.62/1.14/0.94 | 8349 (1.1e-16) | False | True |
