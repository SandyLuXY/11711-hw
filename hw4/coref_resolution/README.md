## Data Ambiguation with Coreference Resolution
- To run on the GoEmotions dataset, put the following files under the GoEmotions directory. "https://github.com/google-research/google-research/tree/master/goemotions" 
- The coreference resolution module is written in ```coreference_resolution.py```
- To run the coreference resolution pipeline on the GoEmotions dataset, execute the following command.
    - ```python coreference_resolution.py```
- To train the BERT with the processed data, run the following command.
    - ```python bert_classifier_w_coreference_resolution.py```
- To evaluate the test results and calculate the metrics, run the following command.
    - ```python calculate_metric.py```
- To generate the embeddings from the BERT model, in bert_classifier.py, set the do_train flag to be false, set the init_checkpoint to be the checkpoint file and set the output_dir to be the current directory. Then run the command 
```python bert_classifier.py```. The training features and test features will be saved to train.tsv.featrues.txt and test.tsv.features.txt.