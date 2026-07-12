# Set the default goal so running `make` with no arguments prints the help menu
.DEFAULT_GOAL := help
environment := ft_lr

.PHONY: help
help: ## Show this help menu
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: predict
predict: ## Predict car price form car's kilometers
	python3 predict.py

.PHONY: train
train: ## Train model with data from data.csv
	python3 train.py

.PHONY: showdata
showdata: ## sort
	@head -n 1 data.csv; tail -n +2 data.csv | sort -t ',' -k1,1n
.PHONY: set
set: ## Set a python environmen for this proyect
	## bash ;	python3 -m venv $(environment); . ./$(environment)/bin/activate; pip install -r requirements.txt
	bash -c "python3 -m venv $(environment) && source $(environment)/bin/activate && pip install -r requirements.txt"

.PHONY: unset
unset: ## removes the python 
	rm -rf $(environment)

.PHONY: upgrade
upgrade: ## Upgrades pip
	pip install --upgrade pip