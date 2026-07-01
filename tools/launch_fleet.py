"""Launch the shard-parallel embed fleet: 3 jobs, disjoint shard slices, 4 cores each.

Each job keeps its own per-country cap, so the global cap is roughly cap*n_jobs for
saturated countries. Progress lands as data/emb/train_shard_XX.parquet, resumable:
a re-run skips shards whose parquet already exists.
"""
import sys

import hopsworks

SLICES = {"embed-a": "1-10", "embed-b": "11-20", "embed-c": "21-30"}
CAP = 1200
SCRIPT = "Users/lex00000/where-on-earth/pipelines/embed_pipeline.py"


def main():
    proj = hopsworks.login()
    ja = proj.get_job_api()
    base = ja.get_job("embed-shards")           # deployed shell, reuse its appPath
    app_path = base.config["appPath"]
    for name, shards in SLICES.items():
        cfg = dict(base.config)
        cfg["appName"] = name
        cfg["defaultArgs"] = f"--split train --shards {shards} --cap {CAP}"
        cfg["resourceConfig"] = {"cores": 4.0, "memory": 6144, "gpus": 0, "shmSize": 128}
        job = ja.create_job(name, cfg) if ja.get_job(name) is None else ja.get_job(name)
        job.config.update(cfg)
        job.save()
        ex = job.run(args=f"--split train --shards {shards} --cap {CAP}",
                     await_termination=False)
        print(f"{name}: shards {shards}, execution {ex.id}", flush=True)


if __name__ == "__main__":
    main()
