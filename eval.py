import numpy as np

from ptbtokenize import PTBTokenizer
from cider import Cider
from bleu import Bleu
from meteor import Meteor
from rouge import Rouge


class COCOEvalCap:
    def __init__(self):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

    def evaluate(self, gts, res):

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # print(gts)
        # =================================================
        # Set up scorers
        # =================================================
        # print('----- gts', gts)
        # print('----- res', res)
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            if method == 'METEOR':
                continue
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                print("%s: %0.3f" % (method, score))

    def setEval(self, score, method):
        self.eval[method] = score
