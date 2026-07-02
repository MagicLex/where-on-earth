# Where on Earth

![Where on Earth](assets/banner.svg)

[![awesome-ml-systems](https://img.shields.io/badge/awesome--ml--systems-%23004-34d399?labelColor=0b0e11&style=flat)](https://github.com/MagicLex/awesome-ml-systems)
[![Hopsworks](https://img.shields.io/badge/built_on-Hopsworks-1CB182?labelColor=0b0e11&style=flat)](https://www.hopsworks.ai/)

Which country was a photo taken in? A frozen CLIP ViT-B/32 turns the photo into
512 numbers, a head trained on OSV5M street view turns the numbers into one of 200+
countries, and the head has to beat CLIP zero-shot ("a photo taken in France", no
training at all) to justify existing.

## The result

**Top-1 52.3%, top-5 79.8% across 173 countries, on held-out places the model
never saw.** The champion (a one-hidden-layer MLP on frozen CLIP ViT-B/32
embeddings) is 2.5x the CLIP zero-shot baseline it had to beat.

407,340 images embedded from 33 OSV5M zip shards (per-country cap: DE/US/FR/RU/JP
stop at ~5,100, the median country keeps ~900), split by quadtree cell into
330,955 train / 54,816 test rows over 1,563 held-out cells.

| config | top-1 | top-5 |
|---|---:|---:|
| majority class (always US) | 1.4% | 7.8% |
| CLIP zero-shot ("a photo taken in X") | 21.2% | 50.8% |
| centroid (nearest class-mean) | 27.6% | 56.0% |
| logistic regression | 49.3% | 77.6% |
| **MLP (champion)** | **52.3%** | **79.8%** |

Where it knows the world: strongest on countries with distinctive roadscapes and
solid Mapillary coverage (Pakistan 99%, Zambia 89%, Kuwait 88%, Nigeria 87%,
Iceland 86%), weakest where coverage is thin or the scenery is generic
(Montenegro 12%, Cambodia 3%, Ukraine ~0% on few test photos). The full
per-country table ships in `assets/per_country_acc.json`.

![Accuracy map](assets/accuracy_map.png)
![Best and worst countries](assets/best_worst.png)

## Caveats

Read these before quoting the number anywhere.

- **Country-level only, by design.** This does not and will not locate a street,
  a building, or a person. See [docs/HONESTY.md](docs/HONESTY.md).
- **Selection.** OSV5M is Mapillary street view: dashcams and phones on roads.
  Strong on roadscapes; weaker on interiors, food, portraits, and anywhere
  Mapillary coverage is thin. A per-country cap deliberately trades US/EU
  accuracy for global coverage.
- **Split by place, not by row.** Train/test are separated by OSV5M quadtree
  cell, so two frames of the same street never sit on both sides. Row-level
  splits on street imagery are how fake accuracy gets made.
- **The backbone is frozen.** Nothing fine-tunes. Heads train on stored vectors
  in minutes on CPU; the trade is a ceiling on what pixels can say.

## Architecture

An FTI (feature, training, inference) system on Hopsworks. Images become vectors
at the door: the embed jobs stream OSV5M zip shards (2.5 GB each), embed what
survives the cap, and delete the pixels. The feature group stores 512 floats and a
label per photo -- nobody ever re-embeds.

```mermaid
flowchart LR
    subgraph sources
      O[OSV5M 5.1M photos] --> E
      P[OSV5M held-out playset] --> APP
    end
    subgraph Feature
      E[shard-parallel embed jobs<br/>frozen CLIP ViT-B/32] --> FG[(geo_image_embeddings<br/>512-d vectors, no pixels)]
    end
    FG --> FV[geo_country_fv]
    subgraph Training
      FV --> T[centroid / logreg / mlp<br/>vs majority + CLIP zero-shot] --> M[(geo_country)]
      T --> LB[(geo_leaderboard)]
    end
    subgraph Inference
      M --> D[whereonearth KServe] --> APP[geoapp Streamlit]
    end
```

The file-by-file map:

```
collect/slim_metadata.py     2.9 GB CSV -> 5M-row parquet             (terminal, I/O)
pipelines/embed_pipeline.py  zip shards -> CLIP vectors parquet       (3 parallel jobs)
pipelines/insert_fg.py       vectors -> feature group + feature view  (terminal)
pipelines/train.py           heads vs baselines -> model registry     (Hopsworks job)
serving/predictor.py         photo -> same embed module -> country    (KServe)
app/app.py                   upload / play-vs-model / honesty tab     (Hopsworks app)
embed_features.py            shared CLIP embedding (no train/serve skew)
```

## Reproduce

Clone into a Hopsworks project on the `/hopsfs/...` FUSE mount.

**Fast path (skip ~10h of CPU):** the precomputed embeddings are a
[release asset](https://github.com/MagicLex/where-on-earth/releases/tag/embeddings-v1).

```bash
mkdir -p data/emb
curl -L https://github.com/MagicLex/where-on-earth/releases/download/embeddings-v1/embeddings_train.tar | tar x -C data/emb
curl -L https://github.com/MagicLex/where-on-earth/releases/download/embeddings-v1/embeddings_test.tar  | tar x -C data/emb
curl -Lo data/country_text_emb.parquet https://github.com/MagicLex/where-on-earth/releases/download/embeddings-v1/country_text_emb.parquet
make insert && make train-job && make serve && make app
```

**Full path (rebuild the vectors yourself):**

```bash
curl -o data/train.csv https://huggingface.co/datasets/osv5m/osv5m/resolve/main/train.csv
curl -o data/test.csv  https://huggingface.co/datasets/osv5m/osv5m/resolve/main/test.csv
make meta            # slim the CSVs
make embed-fleet     # 3 parallel embed jobs over disjoint zip shards
make prompts-job     # CLIP text embeddings of all 222 countries
make insert          # vectors -> FG + FV
make train-job       # heads vs baselines -> registry
make serve           # whereonearth KServe endpoint
make app             # geoapp Streamlit front-end
```

No GPU required anywhere: embedding is shard-parallel CPU jobs, heads are
sklearn on vectors, serving embeds one photo per request.

## The demo

Upload a photo and get a country distribution, or play a round against the model
on a real held-out OSV5M street-view photo -- you guess, it guesses, the map
reveals the truth. The honesty tab shows the leaderboard including the baselines.
The play-set ships in-repo with per-author attribution (CC-BY-SA).
