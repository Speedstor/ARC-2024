import json
import numpy as np

def cut_at_token(output, token_id):
    eos_positions = (output==token_id).nonzero()[0]
    return output[:eos_positions[0]] if len(eos_positions) else output

class ArcDataset(object):
    def __init__(self, queries, replies={}, keys=None, is_orig=False, is_fake=False):
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.keys = sorted(queries.keys()) if keys is None else keys

    @classmethod
    def from_file(cls, queries_file):
        print(f"*** Load challanges from '{queries_file}'...")
        with open(queries_file) as f: 
            queries = f.read()
            return cls(queries=json.loads(queries))

    def get(self, key, formatter):
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'], i=len(self.queries[key]['train']))
        reply = formatter.fmt_reply(self.replies[key], self.faulty.get(key)) if key in self.replies else ''
        return {
            "train": train,
            "query": query,
            "reply": reply,
            "input": train+query,
            "text": train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        }

    def as_list(self, formatter):
        return [self.get(key, formatter) for key in self.keys]


class ArcFormatter(object):
    masking=1
    inp_prefix='I'
    out_prefix='O'
    arr_sep='\n'
    arr_end='\n'
    pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz'
    pre_out=['+/-=']*99
    pretext_corpus_split='\n'

    out2_use=False
    masking=0

    qry_prefix=inp_prefix
    rpl_prefix=out_prefix
    rpl_sep=rpl_prefix
    arr_beg=''
    pre_out_empty = ['']*99
    exa_sep=''
    exa_end=''
    dec_sep = arr_sep
    min_wid=0
    min_pad=''
    collator_kwargs={}
    repeat_input_aug=None
    repeat_input_pre=None

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def fmt_array(self, array):
        return self.arr_beg + self.arr_sep.join(str(row).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')+self.min_pad*max(0, self.min_wid-len(row)) for row in array) + self.arr_end

    def get_pre_out(self, pretext_split):
        if self.pre_out is None: return self.pre_out_empty
        if pretext_split: return [self.pretext_corpus_split.join(list(p) + ['']) for p in self.pre_out]
        return self.pre_out

    def fmt_train(self, train, last_is_challenge=False, pretext_split=False):
        po = self.get_pre_out(pretext_split=pretext_split)
        ex = [(f"{self.fmt_query([x], i, pretext_split=pretext_split)}{self.fmt_reply([x['output']])}" if last_is_challenge and i+1==len(train) else
               f"{self.inp_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{self.out_prefix}{self.fmt_array(x['output'])}") for i, x in enumerate(train)]
        pre = self.pretext_corpus_split.join(list(self.pretext)+['']) if pretext_split else self.pretext
        end = '' if last_is_challenge else (self.exa_end + self.tokenizer.eos_token)
        return pre + (self.exa_end + self.tokenizer.eos_token + self.exa_sep).join(ex) + end

    def fmt_query(self, query, i, pretext_split=False):
        po = self.get_pre_out(pretext_split=pretext_split)
        return ''.join(f"{self.qry_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{self.rpl_prefix}" for x in query[:1])

    def repeat_input(self, x, no_aug=False):
        if self.repeat_input_aug is None: return ''
        return f"{self.repeat_input_pre}{self.fmt_array(((lambda x: x) if no_aug else self.repeat_input_aug)(x['input']))}"

    def fmt_reply(self, reply, fault=None):
        ids = self.fmt_array(reply[0]) + self.exa_end + self.tokenizer.eos_token
        if self.out2_use:
            if fault is None: fault = reply
            ids = self.fmt_array(fault[0]) + self.exa_end + self.out2_token + ids
        return ids

    @staticmethod
    def is_valid_solution(guess):
        return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

    def de_tokenize(self, tokens, scores=None):
        import torch
        tokens_cut = cut_at_token(tokens, self.tokenizer.eos_token_id)
        de_tokenized = self.tokenizer.batch_decode([tokens_cut])[0]
        score_val = None
        if scores is not None:
            tokens_with_eos = tokens[:len(tokens_cut)+1]
            score_val = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1).numpy().copy()[np.arange(len(tokens_with_eos)), tokens_with_eos].sum()
            number_token_ids = [self.tokenizer.vocab[k] for k in map(str, range(10))]
            fault_token_id = self.collator_kwargs.get('fault_token_id')
            if fault_token_id is not None: number_token_ids.append(fault_token_id)
            number_token_ids = np.array(number_token_ids)
            number_positions = (tokens_cut[..., np.newaxis] == number_token_ids).any(-1)
            scores = scores[:len(tokens_cut), number_token_ids][number_positions]
            scores = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1)[:, :10].numpy().copy()
        return max(len(tokens)+1, len(tokens_cut)), score_val, de_tokenized, scores
