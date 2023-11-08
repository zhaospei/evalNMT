# evalNMT
Automatic evaluation metrics used in text generation tasks, including **BLEU**, **ROUGE-L** and **METEOR**, ...

### Preinstall
Install package `psutil` `prettytable` through pip:

```
pip install psutil prettytable
```

### Result Evaluation

Compute the metrics score by running `eval.py` with the reference file path and the inference result path specified. 

```bash
python eval.py  --gold_dir  [The path of the reference file] --prd_dir [The path of the predicted file]
```

If your predict file or gold file already have row ids, you can choose option --prd_index and --gold_index instead.

For example,
- With files without contain row ids

    ```bash
    python3 eval.py --prd_dir results/test.output --gold_dir results/test.gold
    ```
    Ouput
    ```
    Refs:  6406
    Preds:  6406
    BLEU: 16.72
    Meteor:  10.67
    Rouge-L:  14.82
    Cider:  0.98
    EM:  8.57
    Precision:  15.2
    Recall:  15.15
    ```

- With files contain row ids

    ```bash
    python3 eval.py --prd_dir results/index.test.output --gold_dir results/index.test.gold --prd_index --gold_index
    ```
    Ouput
    ```
    Refs:  25726
    Preds:  25726
    BLEU: 18.11
    Meteor:  10.12
    Rouge-L:  22.49
    Cider:  0.43
    EM:  0.72
    Precision:  27.34
    Recall:  22.28
    ```

