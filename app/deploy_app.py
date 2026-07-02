"""Deploy the geoguessing front-end as a Hopsworks Streamlit app.

Run from a Hopsworks terminal where this repo lives under the project FUSE mount.
The app is a thin client of the whereonearth endpoint: no pickle, no pinned ML
stack, stock app base env.

Redeploys use the full recovery sequence, because `app.stop()` returns before the
execution actually dies and the naive stop-then-run desyncs the platform's state
machine (KILLED shown while a healthy pod serves, then 130062 "already running"
forever -- see the playbook BLOCKERS "Apps" entry). Order: stop, purge the k8s
deployment if it lingers, wait until the pods are gone, stop zombie executions,
settle, run.
"""
import subprocess
import time
from pathlib import Path

import hopsworks

APP_NAME = "geoapp"
ENV_NAME = "python-app-pipeline"

# /hopsfs/<...project-relative...>/app/deploy_app.py -> project-relative path
rel = str(Path(__file__).resolve()).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "app.py")


def _pods():
    out = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines() if APP_NAME in l]


def _purge_k8s():
    out = subprocess.run(["kubectl", "get", "deployment"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if APP_NAME in line:
            name = line.split()[0]
            subprocess.run(["kubectl", "delete", "deployment", name], capture_output=True)
            print(f"purged k8s deployment {name}", flush=True)
    for _ in range(60):                     # bounded wait for pods to drain
        if not _pods():
            return
        time.sleep(5)
    raise RuntimeError("app pods refused to drain")


def _stop_zombies(project):
    job = project.get_job_api().get_job(APP_NAME)
    if job is None:
        return
    for ex in job.get_executions() or []:
        if ex.final_status in ("UNDEFINED", None):
            try:
                ex.stop()
                print(f"stopped zombie execution {ex.id}", flush=True)
            except Exception:
                pass


def main():
    project = hopsworks.login()
    apps = project.get_app_api()
    print(f"app_path={APP_PATH} env={ENV_NAME}", flush=True)
    app = apps.get_app(APP_NAME)
    if app is None:
        app = apps.create_app(name=APP_NAME, app_path=APP_PATH,
                              environment=ENV_NAME, memory=2048, cores=1.0)
    else:
        try:
            app.stop()
        except Exception:
            pass
        _purge_k8s()
        _stop_zombies(project)
        time.sleep(10)                      # let the platform settle before run
    app.run(await_serving=True)
    print("serving:", app.serving)
    print("URL:", app.get_url())


if __name__ == "__main__":
    main()
