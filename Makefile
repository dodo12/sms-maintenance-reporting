.PHONY: install data smoke all figures controls clean
install:
	pip install -r requirements.txt
data:
	bash scripts/download_data.sh
smoke:
	python experiments/smoke_test.py
all:
	python experiments/run_models.py
	python experiments/text_study.py
	python experiments/controls_images.py
	python experiments/controls_vibration.py
	python experiments/stats_text_models.py
	python experiments/make_figures.py
controls:
	python experiments/controls_images.py
	python experiments/controls_vibration.py
figures:
	python experiments/make_figures.py
clean:
	rm -rf cache results/curves results/figures
