import os
import argparse
import logging
from functools import lru_cache
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import sys  # for parent directory imports

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from vectorspace import SensesVSM
from sentence_encoder import TransformerSentenceEncoder, min_maxlen_encoder

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


@lru_cache()
def wn_sensekey2synset(sensekey):
    """Convert sensekey to synset."""
    lemma = sensekey.split('%')[0]
    for synset in wn.synsets(lemma):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                return synset
    return None


@lru_cache()
def wn_lemmatize(w, postag=None):
    w = w.lower()
    if postag is not None:
        return wn_lemmatizer.lemmatize(w, pos=postag[0].lower())
    else:
        return wn_lemmatizer.lemmatize(w)


def load_wic(wic_path='external/wic-tsv', folder='Training', setname='train'):
    data_entries = []
    examples_path = '%s/%s/%s_examples.txt' % (wic_path, folder, setname)
    definitions_path = '%s/%s/%s_definitions.txt' % (wic_path, folder, setname)
    for example, definition in zip(open(examples_path, encoding='utf-8'), open(definitions_path, encoding='utf-8')):
        word, idx, ex = example.strip().split('\t')
        defi = definition.strip().split('\n')
        data_entries.append([word, idx, ex, defi[0]])

    if setname == 'test':  # no gold
        return [e + [None] for e in data_entries]

    gold_entries = []
    gold_path = '%s/%s/%s_labels.txt' % (wic_path, folder, setname)
    for line in open(gold_path):
        gold = line.strip()
        if gold == 'T':
            gold_entries.append(True)
        elif gold == 'F':
            gold_entries.append(False)

    assert len(data_entries) == len(gold_entries)
    return [e + [gold_entries[i]] for i, e in enumerate(data_entries)]


def process_entry_batch(batch, sentence_encoder, seg_spec=None):
    """Passes a batch of sentence entries through bert and
    """
    batch_sents = batch
    if seg_spec is None:
        batch_bert = sentence_encoder.token_embeddings(batch_sents)
    else:
        (minlen, maxlen, batch_size) = seg_spec
        with min_maxlen_encoder(sentence_encoder, minlen, maxlen) as sent_encoder:
            batch_bert = sent_encoder.token_embeddings(batch_sents)

    return batch_bert


def infer_arch_from_name(model_name):
    if model_name.startswith('bert-'):
        return "BERT"
    elif model_name.startswith('roberta-'):
        return "RoBERTa"
    elif model_name.startswith('xlnet-'):
        return "XLNet"
    elif model_name.startswith('gpt2-'):
        return "GPT2"
    elif model_name.startswith('ctrl'):
        return "CTRL"
    else:
        raise ValueError(model_name)


def encoder_config(args):
    return {
        'model_name_or_path': args.pytorch_model,
        'model_arch': infer_arch_from_name(args.pytorch_model),
        'min_seq_len': args.min_seq_len,
        'max_seq_len': args.max_seq_len,
        'pooling_strategy': 'NONE',
        'pooling_layer': args.pooling_layer,  # [-4, -3, -2, -1],
        'tok_merge_strategy': args.merge_strategy}


def build_encoder(args):
    backend = args.backend
    enc_cfg = encoder_config(args)
    if backend == 'bert-as-service':
        return BertServiceSentenceEncoder(enc_cfg)
    elif backend == 'pytorch-transformer':
        return TransformerSentenceEncoder(enc_cfg)
    else:
        raise NotImplementedError("backend " + backend)


def run_train(args, postag=None):
    sentence_encoder = build_encoder(args)
    instances, labels = [], []
    count = 0
    out_of_vocab = 0
    for wic_idx, wic_entry in enumerate(load_wic(args.wic_path)):
        word, idx, ex, defi, gold = wic_entry
        bert_ex, bert_defi = process_entry_batch([ex, defi], sentence_encoder)

        # example
        ex_curr_word, ex_curr_vector = bert_ex[int(idx)]
        ex_curr_lemma = wn_lemmatize(word, postag)
        ex_curr_vector = ex_curr_vector / np.linalg.norm(ex_curr_vector)

        if senses_vsm.ndims == 1024:
            ex_curr_vector = ex_curr_vector
        elif senses_vsm.ndims == 1024 + 1024:
            ex_curr_vector = np.hstack((ex_curr_vector, ex_curr_vector))

        ex_curr_vector = ex_curr_vector / np.linalg.norm(ex_curr_vector)
        ex_matches = senses_vsm.match_senses(ex_curr_vector, lemma=ex_curr_lemma, postag=postag, topn=None)
        #ex_synsets = [(wn_sensekey2synset(sk), score) for sk, score in ex_matches]
        try:
            ex_wsd_vector = senses_vsm.get_vec(ex_matches[0][0])
        except IndexError:
            out_of_vocab += 1
            ex_wsd_vector = ex_curr_vector

        # definition
        vecs = []
        for vec in bert_defi:
            defi_curr_word, defi_curr_vec = vec
            vecs.append(defi_curr_vec)
        curr_sent_vec = np.array(vecs).mean(axis=0)
        curr_sent_vec = curr_sent_vec / np.linalg.norm(curr_sent_vec)

        if senses_vsm.ndims == 1024:
            curr_sent_vec = curr_sent_vec
        elif senses_vsm.ndims == 1024 + 1024:
            curr_sent_vec = np.hstack((curr_sent_vec, curr_sent_vec))

        vecs = []
        for vec in bert_ex:
            ex_word, ex_vec = vec
            vecs.append(ex_vec)
        ex_sent_emb = np.array(vecs).mean(axis=0)
        ex_sent_emb = ex_sent_emb / np.linalg.norm(ex_sent_emb)

        if senses_vsm.ndims == 1024:
            ex_sent_emb = ex_sent_emb
        elif senses_vsm.ndims == 1024 + 1024:
            ex_sent_emb = np.hstack((ex_sent_emb, ex_sent_emb))

        s1_sim = np.dot(ex_curr_vector, ex_wsd_vector)
        s2_sim = np.dot(ex_curr_vector, curr_sent_vec)
        s3_sim = np.dot(ex_wsd_vector, curr_sent_vec)
        #s4_sim = np.dot(ex_sent_emb, curr_sent_vec)

        instances.append([s1_sim, s2_sim, s3_sim])
        labels.append(gold)
        count += 1
        logging.info('target word: ' + ex_curr_word + ' | ' + 'index: ' + str(count))
    print('out of vocab: ', out_of_vocab)

    logging.info('Training Logistic Regression ...')
    clf = LogisticRegression(random_state=42)
    clf.fit(instances, labels)

    logging.info('Saving model to %s' % args.out_path)
    joblib.dump(clf, args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training of WiC-TSV solution using LMMS and LogReg for binary classification.')
    parser.add_argument('-vecs_path', help='Path to LMMS sense vectors', required=True)
    parser.add_argument('-out_path', help='Path to .pkl classifier generated',
                        default='data/models/wic_lr.pkl', required=False)
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('-min_seq_len', type=int, default=3, help='Minimum sequence length (BERT)', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy',
                        required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-pooling_layer', help='Which layers in the model to take for subtoken embeddings',
                        default=[-4, -3, -2, -1], type=int, nargs='+')
    parser.add_argument('-backend', type=str, default='pytorch-transformer',
                        help='Underlying BERT model provider',
                        required=False,
                        choices=['bert-as-service', 'pytorch-transformer'])
    parser.add_argument('-pytorch_model', type=str, default='bert-large-cased',
                        help='Pre-trained pytorch transformer name or path',
                        required=False)
    parser.add_argument('-wic_path', help='Path to wic data', default='external/wic-tsv', required=False)

    args = parser.parse_args()

    if args.backend == 'bert-as-service':
        from bert_as_service import BertServiceSentenceEncoder

    logging.info('Loading SensesVSM ...')
    senses_vsm = SensesVSM(args.vecs_path, normalize=True)

    wn_lemmatizer = WordNetLemmatizer()

    logging.info('Processing sentences ...')
    run_train(args)

