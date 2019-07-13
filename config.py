# -*- coding: utf-8 -*-

import os

biobert_dir = "/home/mlt/saad/tmp/biobert_squad_tuned"
# "/raid/data/saam01/pretrained/bert/biobert_v1.0_pubmed_pmc"
pubmed_dir = "/home/mlt/saad/tmp/BioASQ_Datasets/Task7b/PubMed"
# "/raid/data/saam01/data/BioASQ/Task6b/PhaseA/PubMed"
data_dir = os.path.join(os.path.abspath("."), "data")
os.makedirs(data_dir, exist_ok=True)

trainfile_6b = "/home/mlt/saad/tmp/BioASQ_Datasets/Task6b/BioASQ-training6b/BioASQ-trainingDataset6b.json"
# "/raid/data/saam01/data/BioASQ/Task6b/train/BioASQ-trainingDataset6b.json"

test_dir = "/home/mlt/saad/tmp/BioASQ_Datasets/Task6b/Task6BGoldenEnriched"
# "/raid/data/saam01/data/BioASQ/Task6b/test"
testfiles = {
    "6b1": os.path.join(test_dir, "6B1_golden.json"),
    "6b2": os.path.join(test_dir, "6B2_golden.json"),
    "6b3": os.path.join(test_dir, "6B3_golden.json"),
    "6b4": os.path.join(test_dir, "6B4_golden.json"),
    "6b5": os.path.join(test_dir, "6B5_golden.json")
}

output_dir = os.path.join(os.path.abspath("."), "output")
os.makedirs(output_dir, exist_ok=True)

seed = 2019

eval_every_step = 500
