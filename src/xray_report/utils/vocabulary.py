"""
Vocabulary class for the report decoder.

Follows the same fixed-special-token-index pattern used in the HW3
seq2seq assignment: <pad>=0, <bos>=1, <eos>=2, <unk>=3.
Fill in build_vocab() once IU X-Ray report text is available.
"""
import pandas as pd
import re
def read_parquet_train_text():
    data= pd.read_parquet('../../../data/chexpert_plus_clean.parquet')
    train_set = data[(data['split']=='train')]
    src_train=train_set['section_impression'].values.tolist()
    return src_train

PUNCT_TO_KEEP = ['.', ',', '(', ')', ':', '/', '-', '"']

def tokenize(text):
    text = text.lower().strip()
    text = text.replace('\n', ' ')
    text = text.replace('&lt;deleted&gt;', ' ')

    # pad each punctuation char with spaces so it splits off as its own token
    for p in PUNCT_TO_KEEP:
        text = text.replace(p, f' {p} ')

    # collapse multiple spaces down to one
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split(' ')


def build_vocab(min_freq,tokenized_text):
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
    token_to_idx= {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    idx_to_token = {token:index for index, token in token_to_idx.items()}
    counter = {}
    for tokens in tokenized_text:
        for token in tokens:
            counter[token] = counter.get(token,0)+1

    for token, freq in sorted(counter.items()):
        if freq >= min_freq and token not in token_to_idx:
            idx= len(token_to_idx)
            token_to_idx[token] = idx
            idx_to_token[idx] = token
    print(f"token size: {len(token_to_idx)}")
    return {'token_to_idx': token_to_idx, 'idx_to_token': idx_to_token}

def encode(tokens, token_to_idx, max_len):
    ids = [token_to_idx.get('<bos>')]
    for tok in tokens:
        ids.append(token_to_idx.get(tok, token_to_idx['<unk>']))
    ids.append(token_to_idx['<eos>'])

    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [token_to_idx['<eos>']]  # keep eos even after truncation
    else:
        ids = ids + [token_to_idx['<pad>']] * (max_len - len(ids))
    return ids

def decode(ids, idx_to_token):
    tokens = []
    for i in ids:
        token = idx_to_token[i]
        if token == '<eos>':
            break
        if token in ('<pad>', '<bos>'):
            continue
        tokens.append(token)
    return ' '.join(tokens)

src_train=read_parquet_train_text()
train_tokens= [tokenize(t) for t in src_train]
print(f"Total sentence :{len(train_tokens)}")
vocab=build_vocab(5,train_tokens)
sample = src_train[0]
tokens = tokenize(sample)
ids = encode(tokens, vocab['token_to_idx'], max_len=50)
decoded = decode(ids, vocab['idx_to_token'])

print("ORIGINAL: ", sample)
print("TOKENS:   ", tokens)
print("IDS:      ", ids)
print("DECODED:  ", decoded)

