# -*- coding: utf-8 -*-

import os
import json
import random
import config
import re
import numpy as np
import pickle
import csv
import logging
import sys


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_bioasq_json_file(filename, factoid=True, flist=True, 
                          summary=True, yesno=True):
    """Read BioASQ train / test file."""
    with open(filename, encoding="utf-8", errors="ignore") as rf:
        json_data = json.load(rf)
    qas_list = []
    for qtype in ("factoid", "list", "yesno", "summary"):
        qas_list.extend([d for d in json_data['questions'] if d['type'] == qtype])
    return qas_list


def read_pubmed_json_file(filename, with_meshterms=False):
    with open(filename, encoding="utf-8", errors="ignore") as rf:
        json_data = json.load(rf)
    doc = json_data["title"]
    doc += " " + json_data["abstract"]
    if with_meshterms:
        doc += " " + ", ".join(json_data["meshterms"])
    doc = doc.strip()
    return doc


def read_question_relevant_doc_pairs(qas_list):
    """Returns questions and true relevant documents pairs."""
    
    # all articles available locally
    files = {
        filename for filename in os.listdir(config.pubmed_dir) 
        if not filename.startswith("tokenized_") # ignore tokenized files
    }
    
    for question in qas_list:
        # skip if documents field missing (can happen?)
        if "documents" not in question:
            continue
        
        qas_data = {
            "id": question["id"],
            "body": question["body"],
            "documents": {}
        }
        
        for pubmed_url in question["documents"]:
            doc_id = pubmed_url.split("/")[-1]
            pubmed_filename = doc_id + ".json"
            
            # skip document if we do not have local copy
            if pubmed_filename not in files:
                continue
            
            doc_text = read_pubmed_json_file(os.path.join(config.pubmed_dir, pubmed_filename))
            doc_text = doc_text.strip()
            # skip over empty documents (if so?)
            if not doc_text:
                continue
            
            qas_data["documents"][doc_id] = doc_text
        
        if qas_data["documents"]:
            yield qas_data


def create_train_val_split(indices, val_size):
    random.Random(config.seed).shuffle(indices)
    val_size = int(len(indices) * val_size)
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]
    return train_indices, val_indices


def ws_normalize(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DataHandler:
    
    def __init__(self, bm25_file,  validation_size=0.2, ng_sampling="rand"):
        # not memory friendly but can be easily improved
        self._train6b_qas = read_bioasq_json_file(config.trainfile_6b)
        self._train_qp_pairs = list(read_question_relevant_doc_pairs(self._train6b_qas))
        
        # create training and validation sets
        indices = list(range(len(self._train_qp_pairs)))
        train_indices, val_indices = create_train_val_split(indices, validation_size)
        
        # true relevant (positive) question-document (qd) pairs
        self.train_pos_qd = [self._train_qp_pairs[i] for i in train_indices]
        self.val_pos_qd = [self._train_qp_pairs[i] for i in val_indices]
        
        with open(bm25_file, "rb") as rf:
            self.qid2bm25ranks = pickle.load(rf)
        self.ng_sampling = ng_sampling
        
        # keep all doc ids
        self.all_docids = {
            filename.split(".")[0] for filename in os.listdir(config.pubmed_dir) 
            if not filename.startswith("tokenized_")
        }
        
        # load test data files
        self.test_qas = {}
        for testbatch, testfile in config.testfiles.items():
            self.test_qas[testbatch] = read_bioasq_json_file(testfile)
    
    def random_negative_samples(self, qas, k):
        # collect all positive samples
        pos_docids = set(qas["documents"].keys())
        neg_pool = list(self.all_docids - pos_docids)
        # randomize and take k
        random.Random(config.seed).shuffle(neg_pool)
        neg_samples = neg_pool[:k]
        return neg_samples
    
    def bm25_negative_samples(self, qas, k):
        # collect all positive samples
        pos_docids = set(qas["documents"].keys())
        neg_pool = [i for i, _ in self.qid2bm25ranks[qas["id"]] if i not in pos_docids]
        random.Random(config.seed).shuffle(neg_pool)
        neg_samples = neg_pool[:k]
        return neg_samples
    
    def create_train_file(self, k=5):
        wf = open(
            os.path.join(config.data_dir, "train.tsv"),
            "w",
            encoding="utf-8",
            errors="ignore"
        )
        # header
        wf.write("q_id\tquestion\tpos_doc\tpos_docid\tneg_doc\tneg_docid\n")
        
        for qas in self.train_pos_qd:
            
            qid = qas["id"]
            # qid: 5172f8118ed59a060a000019 corrupted in train file, contain newline
            qtext = ws_normalize(qas["body"])
            if not qtext or not qid:
                continue
            
            if self.ng_sampling != "bm25":
                neg_samples = self.random_negative_samples(qas, k)
            else:
                neg_samples = self.bm25_negative_samples(qas, k)
            
            for i, pos_docid in enumerate(qas["documents"]):
                pos_doctext = ws_normalize(qas["documents"][pos_docid])
                if not pos_doctext:
                    continue
                for j, neg_docid in enumerate(neg_samples):
                    neg_doctext = read_pubmed_json_file(
                        os.path.join(config.pubmed_dir, neg_docid + ".json")
                    )
                    neg_doctext = ws_normalize(neg_doctext)
                    if not neg_doctext:
                        continue
                    example_id = qid + "_p_" + str(i) + "_n_" + str(j)
                    line = "\t".join([
                        example_id,
                        qtext,
                        pos_doctext,
                        pos_docid,
                        neg_doctext,
                        neg_docid
                    ])
                    wf.write(line + "\n")
        wf.close()
    
    def create_dev_file(self, k=300):
        wf = open(
            os.path.join(config.data_dir, "dev.tsv"),
            "w",
            encoding="utf-8",
            errors="ignore"
        )
        # header
        wf.write("q_id\tquestion\tdoc\tdocid\n")
        id2rels = {}
        for qas in self.val_pos_qd:
            
            qid = qas["id"]
            qtext = ws_normalize(qas["body"])
            if not qtext or not qid:
                continue
            
            id2rels[qid] = [url.split("/")[-1] for url in qas["documents"]]
            qdocs = [i for i, _ in self.qid2bm25ranks[qid]][:k]
            for i, qdoc in enumerate(qdocs):
                example_id = qid + "_" + qdoc
                qdoctext = read_pubmed_json_file(
                    os.path.join(config.pubmed_dir, qdoc + ".json")
                )
                qdoctext = ws_normalize(qdoctext)
                if not qdoctext:
                    continue
                line = "\t".join([
                    example_id,
                    qtext,
                    qdoctext,
                    qdoc
                ])
                wf.write(line + "\n")
        wf.close()
        with open(os.path.join(config.data_dir, "dev_id2rels.pkl"), "wb") as wf:
            pickle.dump(id2rels, wf)
    
    def create_test_file(self, k=100):
        for batch_name, batch_qas in self.test_qas.items():
            wf = open(
                os.path.join(config.data_dir, "test_{}.tsv".format(batch_name)),
                "w",
                encoding="utf-8",
                errors="ignore"
            )
            # header
            wf.write("q_id\tquestion\tdoc\tdocid\n")
            for qas in batch_qas:
                
                qid = qas["id"]
                qtext = ws_normalize(qas["body"])
                if not qtext or not qid:
                    continue
                
                qdocs = [i for i, _ in self.qid2bm25ranks[qid]][:k]
                for i, qdoc in enumerate(qdocs):
                    example_id = qid + "_" + qdoc
                    qdoctext = read_pubmed_json_file(
                        os.path.join(config.pubmed_dir, qdoc + ".json")
                    )
                    qdoctext = ws_normalize(qdoctext)
                    if not qdoctext:
                        continue
                    line = "\t".join([
                        example_id,
                        qtext,
                        qdoctext,
                        qdoc
                    ])
                    wf.write(line + "\n")
            wf.close()
            # also generate template file for predictions
            for idx, qas in enumerate(batch_qas):
                if "documents" in qas:
                    batch_qas[idx]["documents"] = []
            wf = open(
                os.path.join(config.data_dir, "test_template_{}.json".format(batch_name)),
                "w",
                encoding="utf-8",
                errors="ignore"
            )
            batch_qas = {
                "questions": batch_qas
            }
            json.dump(batch_qas, wf, indent=2)
            wf.close()



## BERT utilities


class BertTextTransform:
    
    def __init__(self, tokenizer, labels_list=None):
        self.tokenizer = tokenizer
        if labels_list:
            self.label2id = {label:i for i, label in enumerate(labels_list)}
            self.id2label = {i:label for label, i in self.label2id.items()}
    
    def __call__(self, text_a, text_b=None, max_seq_length=128, 
                 label=None, return_tokens=False):
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = None
        
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if label:
            if isinstance(label, list):
                label_id = [self.label2id for label in label]
            else:
                label_id = self.label2id[label]
        else:
            label_id = None
        
        if return_tokens:
            return input_ids, input_mask, segment_ids, label_id, tokens
        else:
            return input_ids, input_mask, segment_ids, label_id
    
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to m"""
        
        # Here we truncate the pair such that all query terms (tokens_a)
        # are kept while removing document terms only
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            else:
                tokens_b.pop()


class InputExample(object):
    
    def __init__(self, guid, text_a, text_b=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class BioasqProcessor:
    """Processor for the BioASQ Task6b PhaseA."""
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    
    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            try:
                examples.append((
                    InputExample(guid=guid, text_a=line[1], text_b=line[2]),
                    InputExample(guid=guid, text_a=line[1], text_b=line[4])
                ))
            except IndexError:
                continue
        return examples
    
    def _eval_examples(self, lines):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            try:
                examples.append(
                    InputExample(guid=guid, text_a=line[1], text_b=line[2])
                )
            except IndexError:
                continue
        return examples
    
    def get_dev_examples(self, data_dir):
        return self._eval_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")))
    
    def get_test_examples(self, data_dir, batch_num):
        batch_num = "6b" + str(batch_num)
        testfile = "test_{}.tsv".format(batch_num)
        return self._eval_examples(self._read_tsv(os.path.join(data_dir, testfile)))


def example_log(guid, tokens, input_ids, segment_ids, input_mask):
    logger.info("*** Example ***")
    logger.info("guid: %s" % guid)
    logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))


def convert_train_examples_to_features(examples, max_seq_length, tokenizer):
    btt = BertTextTransform(tokenizer)
    features = []
    # example <question, +ve doc>, <question, -ve doc>
    for ex_index, (ex_qp, ex_qn) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        local_features = []
        
        input_ids_qp, input_mask_qp, segment_ids_qp, _, tokens_qp = btt(
            ex_qp.text_a, ex_qp.text_b, 
            max_seq_length=max_seq_length, return_tokens=True
        )
        guid_qp = ex_qp.guid
        local_features.append(InputFeatures(input_ids_qp, input_mask_qp, segment_ids_qp))
        
        input_ids_qn, input_mask_qn, segment_ids_qn, _, tokens_qn = btt(
            ex_qn.text_a, ex_qn.text_b, 
            max_seq_length=max_seq_length, return_tokens=True
        )
        guid_qn = ex_qn.guid
        local_features.append(InputFeatures(input_ids_qn, input_mask_qn, segment_ids_qn))
        
        features.append(local_features)
    
        if ex_index < 5:
            example_log(guid_qp, tokens_qp, input_ids_qp, segment_ids_qp, input_mask_qp)
            example_log(guid_qn, tokens_qn, input_ids_qn, segment_ids_qn, input_mask_qn)
    
    return features


def convert_eval_examples_to_features(examples, max_seq_length, tokenizer):
    btt = BertTextTransform(tokenizer)
    features = []
    guids = []
    # example <question, document>
    for ex_index, ex_qd in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        input_ids_qd, input_mask_qd, segment_ids_qd, _, tokens_qd = btt(
            ex_qd.text_a, ex_qd.text_b,
            max_seq_length=max_seq_length, return_tokens=True
        )
        guid_qd = ex_qd.guid
        
        features.append(InputFeatures(input_ids_qd, input_mask_qd, segment_ids_qd))
        guids.append(guid_qd)
        
        if ex_index < 5:
            example_log(guid_qd, tokens_qd, input_ids_qd, segment_ids_qd, input_mask_qd)
    
    return features, guids


processors = {
    "retrieval": BioasqProcessor
}


if __name__=="__main__":
    dh = DataHandler(
        os.path.join(config.output_dir, "bm25ranks.pkl"),
        validation_size=0.2, ng_sampling="bm25"
    )
    dh.create_train_file(k=5)
    dh.create_dev_file(k=100)
    dh.create_test_file(k=100)
