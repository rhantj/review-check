from collections import Counter

PAD, UNK = 0, 1

def build_vocab(texts, min_freq=2):
    c = Counter(w for t in texts for w in t.lower().split())
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, f in c.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab, max_len):
    ids = [vocab.get(w, UNK) for w in text.lower().split()][:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids
