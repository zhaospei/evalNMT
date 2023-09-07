# evalNMT
Automatic evaluation metrics used in natural language generation tasks, including **BLEU**, **ROUGE-L** and **METEOR**.

!python3 /kaggle/working/evalNMT/eval.py --prd_dir /kaggle/working/evalNMT/data/nngen.codisum.test.msg --gold_dir /kaggle/working/evalNMT/data/codisum.test.msg 

### Result Evaluation

Compute the **BLUE-4**, **METEOR** and **ROUGE-L** score by running the script `./score.sh` with the reference file path and the inference result path specified. 

```bash
./score.sh REFERENCE PREDICTION
```

psutil
