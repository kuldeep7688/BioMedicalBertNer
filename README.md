# Bio Medical Bert Named Entity Recognition
    - Named Entity Recognition for biomedical data using [Scibert](https://arxiv.org/abs/1903.10676) and [Biobert](https://arxiv.org/abs/1901.08746) .
    - Different classification architectures like Bert-Crf, Bert-LSTM-Crf on top of Bert Layer.
    - Finetuning script is compatible with the datasets provided by [Scibert git repo](https://github.com/allenai/scibert/tree/master/data/ner) data folder.

## Finetuning Steps
    - add lables file to the data_dir; a labels file should have all the labels from the data, one label per line.
    - Edit the config file of your chosen architecture and save.
    - Execute the below command
```
python finettune_with_logger.py
    --model_config_path=/path/to/the/edited/config/file
    --data_dir=/path/to/the/data_dir/
    --logger_file_dir=/path/to/the/dir/where/logs/should/be/saved/
    --labels_file=/path/to/the/labels_file/of/the/dataset/file
```
## Results

| Data        | Architecture           | Best Test F1  |
| ------------- |:-------------:|      -----:          |
| JNLPBA        | Bert-CRF      |      -         |
|            | Bert-TokenClassification |   -        |
|               | Bert-LSTM-CRF      |    -          |
| NCBI-Disease      | Bert-CRF |  |
|       | Bert-TokenClassification     |   - |
|  | Bert-LSTM-CRF      |    - |
| BC5CDR        | Bert-CRF      |      -         |
|            | Bert-TokenClassification |   -        |
|               | Bert-LSTM-CRF      |    -          |
| SCIIE        | Bert-CRF      |      -         |
|            | Bert-TokenClassification |   -        |
|               | Bert-LSTM-CRF      |    -          |
