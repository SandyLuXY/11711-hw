# Aggregation and PLDA Approach
## PLDA installation
## This repo is based on https://github.com/google-research/google-research/tree/master/goemotions

```bash

pip install https://github.com/RaviSoji/plda/tarball/master
```
# Generate Embeddings for PLDA:
Please refer to coref_resolution's readme.
## Running CMD:
```bash
# First run the scripts in attribute.ipynb
# Then run the following cmds to get the metrics for the experiment results:
python calculate_metrics.py --test_data test_data --predictions predictions --output output --emotion_file emotion_file --threshold threshold

# example:
python calculate_metrics.py --test_data data/test_ekman.tsv --predictions test_out/plda.tsv --output test_out/plda_out.json --emotion_file data/ekman.txt --threshold 0.2

```
