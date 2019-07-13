# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import time
import spacy
import config
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from data_utils import read_pubmed_json_file, read_bioasq_json_file
from gensim.summarization.bm25 import BM25

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)

wtok = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"]).tokenizer


class BM25ParallelWrap:
    
    def __init__(self, k=1000):
        files = {
            filename for filename in os.listdir(config.pubmed_dir) 
            if filename.startswith("tokenized_")
        }
        self.index2docid = {}
        self.corpus = []
        idx = 0
        for f in files:
            tokens = read_pubmed_json_file(os.path.join(config.pubmed_dir, f)).lower().split() 
            if tokens:
                self.corpus.append(tokens)
                self.index2docid[idx] = f.split("_")[-1].split(".")[0]
                idx += 1
        
        self.bm25 = BM25(self.corpus)
        self.average_idf = sum(float(val) for val in self.bm25.idf.values()) / len(self.bm25.idf)
        self.k = k
        self.qid2topk = {}
    
    def get_scores(self, id_and_question):
        q_id, question = id_and_question
        scores = self.bm25.get_scores(question)
        topk = np.flip(np.argsort(scores)[-self.k:])
        topk = [(self.index2docid[idx], scores[idx]) for idx in topk]
        return topk, q_id
    
    def compute_bm25_ranks(self, ids_and_questions, n_jobs=12):
        t = time.time()
        with ProcessPoolExecutor(max_workers=n_jobs) as exe:
            emap = exe.map(self.get_scores, ids_and_questions, chunksize=100)
            for i, (topk, q_id) in enumerate(emap):
                logger.info("[PROGRESS : BM25-ranking] - {} / {}".format(
                    i, len(ids_and_questions))
                )
                self.qid2topk[q_id] = topk
        t = time.time() - t
        logger.info("BM25 Ranking took %0.3f seconds" % t)


def evaluate(questions, qid2topk, plot=True):
    q2true = {}
    for question in questions:
        q2true[question["id"]] = [url.split("/")[-1] for url in question["documents"]]
    ps = []
    rs = []
    ks = np.linspace(10, 1000, 50).astype('int')
    for k in ks:
        tps = 0
        fps = 0
        all_tps = 0
        for qid in qid2topk:
            if not qid in q2true:
                continue
            y_pred = set([i for i, j in qid2topk[qid][:k]])
            y_true = set(q2true[qid])
            common = y_pred.intersection(y_true)
            diff = y_pred - y_true
            tps += len(common)
            fps += len(diff)
            all_tps += len(y_true)
        p = tps / (tps + fps)
        r = tps / all_tps
        logger.info("P: %0.3f%% | R: %0.3f%% @ k = %d" % (p*100, r*100, k))
        ps.append(p)
        rs.append(r)
    
    if plot:
        # first type plot
        x = ks
        plt.rc('xtick',labelsize=18)
        plt.rc('ytick',labelsize=18)
        plt.plot(x, rs, label="Recall", marker="+")
        plt.plot(x, ps, label="Precision", marker='x')
        plt.title(
            r"Precision and Recall for different values of BM25 $k$",
            size=25
        )
        plt.xlabel(r"BM25 top-k value", size=20)
        plt.ylabel("Scores", size=20)
        plt.legend(loc="best", fontsize=18)
        plt.show()
        # second type plot
        plt.rc('xtick',labelsize=18)
        plt.rc('ytick',labelsize=18)
        plt.plot(rs, ps, marker="+", markersize=12, color="salmon")
        plt.title(
            r"Precision and Recall for $k$ sweep (10-1000)",
            size=25
        )
        plt.xlabel("Recall", size=20)
        plt.ylabel("Precision", size=20)
        plt.show()
    return ps, rs, ks


if __name__=="__main__":
    # read all train questions
    train_questions = read_bioasq_json_file(config.trainfile_6b)
    test_questions = []
    for testfile in config.testfiles.values():
        test_questions.extend(read_bioasq_json_file(testfile))
    questions = train_questions + test_questions
    
    ids_and_questions = []
    for question in questions:
        question_text = question["body"].strip()
        if not question_text:
            continue
        question_text = " ".join([
            t.text.strip() for t in wtok(question_text)
            if t.text.strip()
        ]).strip()
        if not question_text:
            continue
        question_text = question_text.split()
        ids_and_questions.append((question["id"], question_text))
    
    ranker = BM25ParallelWrap()
    logger.info("Running BM25 for %d questions" % len(ids_and_questions))
    ranker.compute_bm25_ranks(ids_and_questions, n_jobs=12)
    
    output_file = os.path.join(config.output_dir, "bm25ranks.pkl")
    
    with open(output_file, "wb") as wf:
        pickle.dump(ranker.qid2topk, wf) 
    
    _ = evaluate(train_questions, ranker.qid2topk)

#
# for docs:
#
# 2019-07-11 13:19:17,154 : INFO : BM25 Ranking took 5236.401 seconds
# 1.45 hrs
#
# for questions:
# 2019-07-11 19:41:16,748 : INFO : BM25 Ranking took 93.662 seconds
#