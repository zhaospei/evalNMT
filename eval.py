import os
import argparse
import numpy as np
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu 

def calc_prec(pred, tgt):
    prds = pred.split()
    tgs = tgt.split()
    count = 0 
    for prd in prds:
        if prd in tgs:
            count += 1
    return count/ len(prds)

def cal_metrics(prd_dir, gold_dir, has_index):
    predictions = list()
    golds = list()
    
    if has_index:
        with open(prd_dir) as f:
            predictions =  [line.strip() for line in f.readlines()]
        with open(gold_dir) as f:
            golds =  [line.strip() for line in f.readlines()]
    else:
        with open(prd_dir) as f:
            predictions =  [str(i)+ '\t' + line.strip() for i, line in enumerate(f.readlines())]
        with open(gold_dir) as f:
            golds =  [str(i)+ '\t' + line.strip() for i, line in enumerate(f.readlines())]
    tmp_file = 'tmpgold.txt'
    with open(tmp_file,'w+') as f:
        f.write('\n'.join(golds))
    
    EM = list()
    precs = list()
    for i, (ref, gold) in enumerate(zip(predictions, golds)):
        EM.append(ref.split() == gold.split())
        precs.append(calc_prec(ref,gold))
    EM = round(np.mean(EM)*100, 3)
    precs = round(np.mean(precs)*100, 3)

    print("EM = %s" % (str(EM)))
    print("precs = %s" % (str(precs)))

    res = {k: [' '.join(v.split('\t')[1:]).strip().lower()] for k, v in enumerate(predictions)}
    tgt = {k: [' '.join(v.split('\t')[1:]).strip().lower()] for k, v in enumerate(golds)}
    
    # precision 1-gram
    print(len(res))
    print(len(tgt))
    score_Meteor, scores_Meteor = Meteor().compute_score(tgt, res)
    print("Meteor: %s" % (float(score_Meteor)*100))
    score_Rouge, scores_Rouge = Rouge().compute_score(tgt, res)
    print("ROUGE-L: %s" % (float(score_Rouge)*100))

    score_Bleu, scores_Bleu = Bleu().compute_score(tgt, res)
    print("Bleu: %s" % (float(score_Bleu[3])*100))

    os.remove(tmp_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prd_dir", default=None, type=str,
                        help="File dir to read predict msg")
    parser.add_argument("--gold_dir", default=None, type=str,
                        help="File dir to read gold msg")
    parser.add_argument("--has_index", action='store_true',
                        help="Contain index line in file")
    
    args = parser.parse_args()

    cal_metrics(args.prd_dir, args.gold_dir, args.has_index)

if __name__ == "__main__":
    main()