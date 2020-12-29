#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Derived from Radim Rehurek's Wiki processor


"""
USAGE: %(program)s OL_file OUTPUT_PREFIX [NTOPICS] [VOCABULARY_SIZE]

Convert a one-line corpus to (sparse) vectors using tf-idf and runs LDA.

The output Matrix Market files can then be compressed (e.g., by bzip2) to save
disk space; gensim's corpus iterators can work with compressed input, too.

`VOCABULARY_SIZE` controls how many of the most frequent words to keep (after
removing tokens that appear in more than 10%% of all documents). Defaults to
300,000.

Example:
  python3 make_lda_corpus.py corpus.ol corpus_model 100 50000
"""


import logging
import os.path
import sys

from gensim.corpora import Dictionary, HashDictionary, MmCorpus, TextCorpus
from gensim.models import TfidfModel
import gensim
from gensim.models.ldamulticore import LdaMulticore

DEFAULT_DICT_SIZE = 300000
ntopics = 100

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))

# check and process input arguments
if len(sys.argv) < 3:
    print(globals()['__doc__'] % locals())
    sys.exit(1)
inp, model_name = sys.argv[1:3]

if len(sys.argv) > 3:
    ntopics = int(sys.argv[3])

if len(sys.argv) > 4:
    keep_words = int(sys.argv[4])
else:
    keep_words = DEFAULT_DICT_SIZE

if os.path.exists(outp + '_wordids.txt.bz2') and os.path.exists(outp + '_corpus.pkl.bz2'):
    dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')
    wiki=TextCorpus.load(outp + '_corpus.pkl.bz2')
else:
    wiki = TextCorpus(inp)
    # only keep the most frequent words
    wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=keep_words)
    wiki.dictionary.save_as_text(outp + '_wordids.txt.bz2')
    wiki.save(outp + '_corpus.pkl.bz2')
    # load back the id->word mapping directly from file
    # this seems to save more memory, compared to keeping the wiki.dictionary object from above
    dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')

# build tfidf
if os.path.exists(outp+'_tfidf.mm'):
    mm = gensim.corpora.MmCorpus(outp+'_tfidf.mm')
else:
    tfidf = TfidfModel(wiki, id2word=dictionary, normalize=True)
    #tfidf.save(outp + '.tfidf_model')

    # save tfidf vectors in matrix market format
    mm=tfidf[wiki]
    MmCorpus.serialize(outp + '_tfidf.mm', mm, progress_cnt=10000)

logger.info("finished pre-processing, starting LDA %s", program)

lda = LdaMulticore(mm, id2word=dictionary, workers=10, num_topics=ntopics)
lda.save(model_name)
topics=lda.show_topics(num_topics=ntopics, num_words=30)
print(topics)
logger.info("finished LDA %s", program)

toptopics=lda.top_topics(corpus=wiki, dictionary=lda.id2word, coherence='u_mass')
logger.info("top topicsL %s", 'u_mass')
print(toptopics)
