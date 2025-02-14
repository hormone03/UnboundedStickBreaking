import pandas as pd
#from gensim.models import CoherenceModel
#from gensim.corpora import Dictionary
import torch
import argparse
import random
import numpy as np
import math
import nltk
from tqdm import tqdm
from itertools import chain
from collections import Counter
import pickle


lookahead = 5
prior_alpha = torch.tensor(0.01)
prior_beta = torch.tensor(0.02)
glogit_prior_mu = torch.Tensor([-1.6])
prior_sigma = torch.Tensor([1.])
uniform_low = torch.Tensor([.01])
uniform_high = torch.Tensor([.99])

parametrizations = dict(Kumar='Kumaraswamy', GLogit='Gauss_SBRK', gaussian='Gaussian', GEM='GEM', Dir='Dirichlet_dist', GD='GDVAE', gdwo='GDWO')

def beta_func(a, b):
    return (torch.lgamma(a).exp() + torch.lgamma(b).exp()) / torch.lgamma(a + b).exp()


def logistic_func(x):
    return 1 / (1 + torch.exp(-x))
#parametrizations = dict(Kumar='Kumaraswamy', GLogit='Gauss_SBRK', gaussian='Gaussian', GEM='GEM', Dir='Dirichlet_dist', GD='GDVAE', gdwo='GDWO')

parser = argparse.ArgumentParser()
parser.add_argument('--h1_out',   type=int,   default=100,
                    help='for hidden layer')
parser.add_argument('--encoder_output', type=int,   default=2,
                    help='no of params outputed by encoder')
parser.add_argument('--topic_size',        type=int,   default=200)
parser.add_argument('--batch_size',       type=int,   default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')

#parser.add_argument('--dataset_tr', type=str, default='data/20NEWS_data/train20NEWS.txt.npy')
#parser.add_argument('--dataset_te', type=str, default='data/20NEWS_data/test20NEWS.txt.npy')
#parser.add_argument('--data_vocab', type=str, default='data/20NEWS_data/vocab20NEWS.pkl')

#parser.add_argument('--dataset_tr', type=str, default='data/KOS_data/trainKOS.txt.npy')
#parser.add_argument('--dataset_te', type=str, default='data/KOS_data/testKOS.txt.npy')
#parser.add_argument('--data_vocab', type=str, default='data/KOS_data/vocabKOS.pkl')

parser.add_argument('--dataset_tr', type=str, default='data/ERONS_data/trainERONS.txt.npy')
parser.add_argument('--dataset_te', type=str, default='data/ERONS_data/testERONS.txt.npy')
parser.add_argument('--data_vocab', type=str, default='data/ERONS_data/vocabERONS.pkl')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

global data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size, dataset_te


def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)

def make_data():
    data_tr = np.load(args.dataset_tr, allow_pickle=True,  encoding='latin1')
    data_te = np.load(args.dataset_te, allow_pickle=True,  encoding='latin1')
    vocab = pickle.load(open(args.data_vocab,'rb'))
    ## if eerror of different array sizes, use maximum array length, E.g, 28103 for Erons
    vocab_size = 28103 #len(vocab) ## ERONS: 28103, KOS: 6907, NIPS: 12420, 20News: len(vocab)
    print(vocab_size)
    #--------------convert to one-hot representation------------------
    print('Converting data to one-hot representation')
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print('Data Loaded')
    print('Dim Training Data',data_tr.shape)
    print('Dim Test Data',data_te.shape)
    #--------------make tensor datasets-------------------------------
    data_tr=np.vstack(data_tr).astype(np.float)
    data_te=np.vstack(data_te).astype(np.float)
    tensor_tr = torch.from_numpy(data_tr).float()
    tensor_te = torch.from_numpy(data_te).float()
    return  tensor_tr, tensor_te, vocab_size

def make_data_():
    data_tr = np.load(args.dataset_tr, allow_pickle=True,  encoding='latin1')
    data_te = np.load(args.dataset_te, allow_pickle=True,  encoding='latin1')
    vocab = pickle.load(open(args.data_vocab,'rb'))
    vocab_size=len(vocab)
    print(vocab_size)

    tensor_tr = []
    tensor_te = []

    for data in data_tr:
        d = Counter(data)
        tensor_tr.append(d)
    for data_ in data_te:
        d_ = Counter(data_)
        tensor_te.append(d_)
    return  tensor_tr, tensor_te, vocab_size


tensor_tr, tensor_te, vocab_size = make_data()

def set_Data_coherence(url):
    data_te = np.load(url, allow_pickle=True, encoding='latin1')
    dataTEST = []
    count = []
    for data in data_te:
        # d = {x:data.count(x) for x in data}
        count.append(len(data))
        d = Counter(data)
        dataTEST.append(d)
        #print(dataTEST[0:2])
    return dataTEST

#specifically for 20news
#def data_for_coherence(data_url):
#  data = []
#  word_count = []
#  fin = open(data_url)
#  while True:
#    line = fin.readline()
#    if not line:
#      break
#    id_freqs = line.split()
#    doc = {}
#    count = 0
#    for id_freq in id_freqs[1:]:
#      items = id_freq.split(':')
#      if int(items[0])-1<0:
#        print('WARNING INDICES!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#
#      doc[int(items[0])-1] = int(items[1])
#      count += int(items[1])
#    if count > 0:
#      data.append(doc)
#      word_count.append(count)
#  fin.close()
#  return data, word_count

def create_batches(data_size, batch_size, shuffle=True):
  """create index by batches."""
  batches = []
  ids = list(range(data_size))
  if shuffle:
    random.shuffle(ids)
  for i in range(int(data_size / batch_size)):
    start = i * batch_size
    end = (i + 1) * batch_size
    batches.append(ids[start:end])
  rest = data_size % batch_size
  if rest > 0:

    batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))
  return batches

def fetch_data(data, count, idx_batch, vocab_size):

  """fetch input data by batch."""
  batch_size = len(idx_batch)
  data_batch = np.zeros((batch_size, vocab_size))
  count_batch = []
  mask = np.zeros(batch_size)
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      for word_id, freq in data[doc_id].items():
        data_batch[i, word_id-1] = freq
      count_batch.append(count[doc_id])
      mask[i]=1.0
    else:
      count_batch.append(0)
  return data_batch, count_batch, mask

def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
    elif prefix in varname:
      ret_list.append(var)
  return ret_list

def print_top_words(beta, feature_names, n_top_words=10,label_names=None,result_file=None):
    print('---------------Printing the Topics------------------')
    if result_file!=None:
      result_file.write('---------------Printing the Topics------------------\n')
    for i in range(len(beta)):
        topic_string = " ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        print(topic_string)
        if result_file!=None:
          result_file.write(topic_string+'\n')
    if result_file!=None:
      result_file.write('---------------End of Topics------------------\n')
    print('---------------End of Topics------------------')

def count_word_combination(dataset,combination):
  count = 0
  w1,w2 = combination
  for data in dataset:
    w1_found=False
    w2_found=False
    for word_id, freq in data.items():
      if not w1_found and word_id==w1:
        w1_found=True
      elif not w2_found and word_id==w2:
        w2_found=True
      if w1_found and w2_found:
        count+=1
        break
  return count

def count_word(dataset,word):
  count=0
  for data in dataset:
    for word_id, freq in data.items():
      if word_id==word:
        count+=1
        break
  return count

def topic_coherence(dataset,beta, n_top_words=10):
  word_counts={}
  word_combination_counts={}
  length = len(dataset)

  coherence_sum=0.0
  coherence_count=0
  topic_coherence_sum=0.0

  for i in range(len(beta)):
    top_words = [j
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]
    topic_coherence = 0
    topic_coherence_count=0.0
    for i,word in enumerate(top_words):
      if word not in word_counts:
        count = count_word(dataset,word)
        word_counts[word]=count
      for j in range(i):
        word2 = top_words[j]
        combination = (word,word2)
        if combination not in word_combination_counts:
          count = count_word_combination(dataset,combination)
          word_combination_counts[combination]=count
        #now calculate coherence
        wc1 = word_counts[word]/float(length)
        wc2 = word_counts[word2]/float(length)
        cc = (word_combination_counts[combination])/float(length)
        if cc>0:
          coherence = math.log(cc/float(wc1*wc2))/(-math.log(cc))
          topic_coherence+=coherence
          coherence_sum+=coherence
        coherence_count+=1
        topic_coherence_count+=1
    topic_coherence_sum+=topic_coherence/float(topic_coherence_count)
  return coherence_sum/float(coherence_count),topic_coherence_sum/float(len(beta))


def get_topics(beta, n_top_words=10):
    topics = []
    for i in range(len(beta)):
      top_words = [j
              for j in beta[i].argsort()[:-n_top_words - 1:-1]]
      topics.append(top_words)
    return topics

def Redundancy(beta, n=10):
    """
    Compute topic redundancy score from
    https://jmlr.csail.mit.edu/papers/volume20/18-569/18-569.pdf
    """
    topics = get_topics(beta, n_top_words=10)

    tr_results = []
    k = len(topics)
    for i, topics_i in enumerate(topics): #enumarate guves index and key in dict
        w_counts = 0
        for w in topics_i[:n]:
            w_counts += np.sum([w in topics_j[:n] for j, topics_j in enumerate(topics) if j != i]) # count(k, l)
        tr_results.append((1 / (k - 1)) * w_counts)
    return sum(tr_results)/len(tr_results)


def uniqueness(beta, n=10):
    """
    Topic uniqueness measure from
    https://www.aclweb.org/anthory/P19-1640.pdf
    """
    topics = get_topics(beta, n_top_words=10)
    tu_results = []
    for topics_i in topics:
        w_counts = 0
        for w in topics_i[:n]:
            w_counts += 1 / np.sum([w in topics_j[:n] for topics_j in topics]) # count(k, l)
        tu_results.append((1 / n) * w_counts)
    return sum(tu_results)/len(tu_results)

def diversity(beta, n=25):
    """
    Compute topic diversity from
    https://doi.org/10.1162/tacl_a_00325
    """
    topics = get_topics(beta, n_top_words=25)
    words = [w for topic in topics for w in topic[:n]]
    return len(set(words)) / len(words)



def overlap(beta, n=10):
    """
    Calculate topic overlap (number of unique topic pairs sharing words)
    """
    topics = get_topics(beta, n_top_words=10)
    k = len(topics)
    overlaps = np.zeros((k, k), dtype=float)
    common_terms = np.zeros((k, k), dtype=float)
    words = Counter([w for topic in topics for w in topic[:n]])

    for i, t_i in enumerate(topics):
        for j, t_j in enumerate(topics[i+1:], start=i+1):
            if i != j:
                overlap_ij = set(t_i[:n]) & set(t_j[:n])
                overlaps[i, j] = len(overlap_ij)
                common_terms[i, j] = sum(words[w] for w in overlap_ij)

    return overlaps.sum()



def Redundancy___(input_data: pd.Series, n: int = 10):
    print(input_data)
    try:
        assert 5 <= n <= 15

    except AssertionError:

        "Invalid Value for n (int) [5,15]"

        raise

    ngram = [0]*len(input_data)
    #ngram = [len(a) * [0] for a in input_data]
    #ngram = [0]*(np.shape(input_data))
    for i in tqdm(range(len(input_data)), desc = 'Get the Redundancy'):

        ngram[i] = [0]*len(input_data[i])
        for j in range(len(input_data[i])):
            if input_data[i] !='':
                print("===>" * 4)
                print(input_data[i])
                print("===>" * 5)
                print(input_data[i][j])
                print("===>" * 6)
                ngram[i][j] = list(nltk.ngrams(input_data([i][j]).split(),10))

    list_ngrams_per_doc = [list(chain(*ngram[i])) for i in range(len(ngram))]

    redundancy = [pd.DataFrame({"Ten_grams":list_ngrams_per_doc[i]}).\
                  value_counts().\
                  loc[lambda x: x>1].\
                  sum()/\
                  len(list_ngrams_per_doc[i]) for i in range(len(list_ngrams_per_doc))]

    return redundancy
