# where-on-earth -- FTI on Hopsworks: photo -> country
# Feature (CLIP embeddings FG) -> Training (heads vs zero-shot) -> Inference (KServe + app)

meta:                ## slim OSV5M metadata (curl CSVs to data/ first)
	python3 collect/slim_metadata.py

embed-fleet:         ## 3 parallel embed jobs over disjoint shard slices
	python3 tools/launch_fleet.py

prompts-job:         ## one-shot CLIP text embeddings of all countries
	hops job deploy geo-prompts tools/embed_country_prompts.py \
		--env torch-training-pipeline --run --wait --overwrite

insert:              ## embedded parquets -> FG + FV
	python3 pipelines/insert_fg.py

train-job:           ## heads vs baselines, register champion
	hops job deploy geo-train pipelines/train.py \
		--env pandas-training-pipeline --run --wait --overwrite

serve:               ## deploy the whereonearth KServe endpoint
	python3 serving/deploy.py

app:                 ## deploy the Streamlit front-end
	python3 app/deploy_app.py

rescore:             ## re-score every played photo against the deployed model
	python3 pipelines/rescore_pipeline.py

feedback-job:        ## ingest played rounds into the geo_feedback FG (schedule daily)
	hops job deploy geo-feedback pipelines/feedback_pipeline.py \
		--env torch-training-pipeline --run --wait --overwrite

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/  --/'
.PHONY: meta embed-fleet prompts-job insert train-job serve app rescore feedback-job help
