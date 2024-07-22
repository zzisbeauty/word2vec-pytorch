import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)


def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer


def get_data_iterator(ds_name, ds_type, data_dir):
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    data_iter = to_map_style_dataset(data_iter)
    return data_iter


def build_vocab(data_iter, tokenizer):
    """Builds vocabulary from iterator"""

    # 统计预料中的 tokens 信息
    tokensiter_tmp = map(tokenizer, data_iter) 
    count_tokens = [] 
    for _ in tokensiter_tmp:
        count_tokens.append(_)
    print(len(count_tokens)) # 可以知道语料中共有多少行数据
    _count_tokens = [item for sublist in count_tokens if sublist for item in sublist]
    print('语料共得到的 tokens 数量：',len(_count_tokens)) # 可以知道语料一共被分为了多少的 tokens， tokens 会比 vocabulary 多

    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter), # 返回一个 token 生成器/迭代器【此 token iterator 在此方法是中是核心，用来返回 vocab】
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    print('vocab lens: ',len(vocab.vocab.itos_))
    return vocab


def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.   # 此备注此时传入的语料已经可以被切割为 tokens 了
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None):

    data_iter = get_data_iterator(ds_name, ds_type, data_dir) # 语料
    tokenizer = get_english_tokenizer() # 英文token获取器【这里还没到 vocabulary 的概念，直接就是 token】

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer) # 两个参数用于构建一个token生成器函数，基于上述获取的 token 获取器，进而得到这一步的 vocabulary
        
    text_pipeline = lambda x: vocab(tokenizer(x)) # 用于后续的 corpus 的进一步处理

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram # 接受的是 one batch data + corpus handle process pipline，即明确每一个 batch 中的数据应该如何处理
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader( # collate_fn 参数决定了每一个 batch data 都应该如何处理，因此 text_pipeline 可以自定义，但是重要是还要定义好接受 text_pipeline 的方法，就是collate_fn
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline), # collate_fn函数接受batch data，它还需要一个参数text_pipeline，因此也在此传入，最终完成dataloader创建
    )
    return dataloader, vocab
    