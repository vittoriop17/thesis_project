import re
import spacy
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


# nlp = spacy.load("en_core_web_md")
#
# tag_re = re.compile(r"<(.).*\/(\1)>")
# text = "I have had Dreamweaver MX2004 since it came out back then. Spent years with it. " \
#        "Feel like I know it real well, but I am still familiar with tables as opposed to CSS. " \
#        "So I thought this would be a great introduction, and it is. The problem is that while I am looking at the video," \
#        " and that simplifies things a lot, I am not getting the intuitive explanation as to why things work the way they do." \
#        " I understand just knowing how to work them is sufficient. " \
#        "I'm tempted to delve into the rich full attributes of Dreamweaver CS5 as explained by this video, " \
#        "albeit difficult to understand the more advanced features, but I won't because this is about the video." \
#        "\n\nThe opening salvo is chock full of... for example, this is the URL; this is where you type in the address bar etc. " \
#        "Only when you start to get into areas such as CSS do you get an introduction that for me is beneficial, " \
#        "and that is only because I am somewhat of a newbie to it. So there you have it, " \
#        "do you get an advanced lesson that flies through things and most find it hard to understand, " \
#        "or do you get this one and appreciate the fact that she walks you through everything step by step, " \
#        "even at the risk of boring the more in the know people, but finally arriving at the section you need to learn," \
#        " and then appreciate that she is not moving at lightning speed.\n\nI'm holding off buying Dreamweaver CS5, " \
#        "although I got the upgrade to Photoshop CS5 and love it, because my MX2004 is not an eligible upgrade version." \
#        " If I could qualify for the upgrade I would grab it, but the learning curve of CSS, " \
#        "and the steep price for the full version when I have MX2004 already is giving me some hesitation. " \
#        "If I was to upgrade, or should I say take the leap to the full version, this tutorial is the one I would use, " \
#        "and I also recommend the&nbsp;<a data-hook=\"product-link-linked\" class=\"a-link-normal\" href=\"/Adobe-Dreamweaver-CS5-on-Demand/dp/0789744449/ref=cm_cr_arp_d_rvw_txt?ie=UTF8\">Adobe Dreamweaver CS5 on Demand</a>&nbsp;which makes a very good reference guide.\nAt some point I will get the new version of CS5 Dreamweaver, and at that point I will do the entire video series, and have a much more in depth analysis of it. But from what I can see after viewing a bunch of the video tutorials, it is the easiest way to indoctrinate you to the newest version. So I recommend it."
# text = re.sub(tag_re, "<TAG>", text)
# text = re.sub("&nbsp;", " ", text)
#
# nlp.tokenizer.add_special_case("<TAG>", [{ORTH: "<TAG>"}])
# doc = nlp(text)
# for token in doc:
#     print(token)
#
# for sent in doc.sents:
#     print(sent)
#
# for sent in doc.sents:
#     tokens_wo_stopwords = [token for token in sent if not token.is_stop]
# print(f"Doc mean vector: {doc.vector}")
#
# # Merge multi-word tokens (e.g.: New York, United Kingdom, ...)
# # This only matches multi-words that are entities
# with doc.retokenize() as retokenizer:
#     for ent in doc.ents:
#        retokenizer.merge(doc[ent.start:ent.end], attrs={"LEMMA": ent.text})

from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# sentences_wo_punct = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']
# sentence_embeddings = model.encode(sentences_wo_punct)
# for sentence, embedding in zip(sentences_wo_punct, sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")


def sentence_len_distribution(df):
    assert hasattr(df, "len_sentence"), "Attribute 'len_sentence' missing. Call function 'extract_words' before"
    df.len_sentence.value_counts()
    df.len_sentence.value_counts().plot.bar()
    plt.xticks(fontsize=6, rotation=90)
    plt.show()


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2)) if norm(v1)!=0 and norm(v2)!=0 else 0

def euclidean_similarity(v1, v2):
    return np.linalg.norm(v1-v2)
