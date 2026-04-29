.PHONY: all download extract-embeddings train-xgboost train-mlp compare visualize clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

download:  ## Download ALS CHM transects from Zenodo
	python -m src.data_processing.download_chm

download-global-lidar:  ## Download global spaceborne lidar samples (GEDI + ICESat-2)
	python -m src.data_processing.download_global_lidar

extract-embeddings:  ## Extract Google AlphaEarth embeddings paired with CHM labels
	python -m src.data_processing.extract_embeddings

build-grid:  ## Process all ALS footprints into 1km validation grid
	python -m src.data_processing.run_all_footprints

train-xgboost:  ## Train XGBoost baseline on embeddings
	python -m src.models.xgboost_baseline

train-mlp:  ## Train fine-tuned MLP on embeddings
	python -m src.models.mlp_finetune

compare:  ## Run three-way model comparison (XGBoost vs MLP vs Meta)
	python -m src.models.compare_models

visualize:  ## Build 3-panel interactive map
	python -m src.data_processing.build_3panel_map

test:  ## Run tests
	python -m pytest tests/ -v

all: download extract-embeddings build-grid train-xgboost train-mlp compare visualize  ## Run full pipeline
