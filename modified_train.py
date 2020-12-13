import argparse
import random
import logging
import os

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

# from pytorch_pretrained_bert import BertForTokenClassification // very old lib. will not work perfectly

from transformers import BertForTokenClassification

from data_loader import DataLoader
from evaluate import evaluate
import utils


import torch.autograd as autograd
import torch.optim as optim

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./BERT-keyphrase-extraction/data/task1', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='bert-base-uncased', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='./BERT-keyphrase-extraction/experiments/base_model', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--test_file', default = './BERT-keyphrase-extraction/h1_7.txt', help = 'path to test file' )



def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def loading_data(lebel):
    print("Loading started", lebel)
    sentences_file = os.path.join(args.data_dir, lebel, 'sentences.txt')
    tags_file = os.path.join(args.data_dir, lebel, 'tags.txt')

    sentences = []
    tags = []

    with open(sentences_file, 'r', encoding='utf-8') as file:
        for line in file:
            # replace each token by its index
            tokens = line.split()
            sentences.append(tokens)
    
    with open(tags_file, 'r') as file:
        for line in file:
            # replace each tag by its index
            tag_seq = line.strip().split(' ')
            tags.append(tag_seq)

    # checks to ensure there is a tag for each token
    assert len(sentences) == len(tags)
    for i in range(len(sentences)):
        # print(sentences[i], tags[i])
        assert len(tags[i]) == len(sentences[i])

    return sentences, tags

def load_final_sentences(sentences_file):
    test_sentences = []
    with open(sentences_file, 'r',encoding='utf-8') as file:
        for cnt,line in enumerate(file):
            tokens = line.split()
            test_sentences.append(tokens)
    return test_sentences

def convert(sentences):
    word_to_ix = {}
    for i in range(0, len(sentences)):
      for j in range(0, len(sentences[i])):
        if sentences[i][j] not in word_to_ix:
          word_to_ix[sentences[i][j]] = len(word_to_ix)

    return word_to_ix

def eval(true_val, pred_val):
    assert len(true_val) == len(pred_val)
    tot = 0
    acc = 0
    for i in range(0, len(true_val)):
        if true_val[i] == pred_val[i]:
            acc += 1
        tot += 1
    
    return acc, tot

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq



if __name__ == '__main__':
    args = parser.parse_args()

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    sentences, tags = loading_data("train")
    s_val, t_val = loading_data("val")
    s_test, t_test = loading_data("test")


    word_to_ix = convert(sentences)
    word_v = convert(s_val)
    word_t = convert(s_test)        

    tag_to_ix = {"I": 0, "O": 1, START_TAG: 2, STOP_TAG: 3}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(sentences[0], word_to_ix)
    #     precheck_tags = torch.tensor([tag_to_ix[t] for t in tags[0]], dtype=torch.long)
    #     print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(10):
        # please run for some more epochs. I dont have GPU permisssion due to any reason..
        print("Epoch: ", epoch)
        for i in range(0, len(sentences)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentences[i], word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags[i]], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

        # checking acc.
        with torch.no_grad():
          for chk in range(0, len(s_val)):
              acc = 0
              tot = 0
              val_chking = prepare_sequence(s_val[chk], word_v)
              tar_v = torch.tensor([tag_to_ix[t] for t in t_val[chk]], dtype=torch.long)
              acc += eval(model(val_chking)[1], tar_v)[0]
              tot += eval(model(val_chking)[1], tar_v)[1]
          print("Val acc.", acc / tot * 100)

        # Save weights of the network

        model_dir = args.model_dir
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer.optimizer if args.fp16 else optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                              'state_dict': model_to_save.state_dict(),
                              'optim_dict': optimizer_to_save.state_dict()},
                                is_best=True,
                              checkpoint=model_dir)


    # Check predictions after training
    with torch.no_grad():
          for chk in range(0, len(s_test)):
              acc = 0
              tot = 0
              test_chking = prepare_sequence(s_test[chk], word_t)
              tar_t = torch.tensor([tag_to_ix[t] for t in t_test[chk]], dtype=torch.long)
              acc += eval(model(test_chking)[1], tar_t)[0]
              tot += eval(model(test_chking)[1], tar_t)[1]
          print("Test acc.", acc / tot * 100)


    gen = load_final_sentences(sentences_file = args.test_file)
    word_gen = convert(gen)
    got_imp = []

    with torch.no_grad():
      for chk in range(0, len(gen)):
          if gen[chk] > 0:
            gen_chking = prepare_sequence(gen[chk], word_gen)
            my_int = model(gen_chking)[1]
            for op in range (0,len(my_int)):
              if my_int[op] == 0:
                got_imp.append(gen[chk][op])

    
    with open('output_mod.txt', 'w') as f:
        f.write("%s " % got_imp)

        print('output flused to disk')
    print('Done')


    



    


