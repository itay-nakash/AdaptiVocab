import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
import utils.utils_funcs as utils
from collections import defaultdict
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
import json
import os
from tqdm import tqdm


def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(file_path: str, data: Dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


class SortedNgramDict:
    def __init__(self):
        from sortedcontainers import SortedList
        self.dict = {}
        self.sorted_list = SortedList(key=lambda item: -1 * SortedNgramDict.score_ngram(item))

    @staticmethod
    def score_ngram(item) -> int:
        ngram, freq = item
        return freq * len(ngram)

    def __setitem__(self, key: Any, value: Any):
        if key in self.dict:
            old_value = self.dict[key]
            self.sorted_list.discard((key, old_value))
        self.dict[key] = value
        self.sorted_list.add((key, value))

    def __getitem__(self, key: Any) -> Any:
        return self.dict[key]

    def __delitem__(self, key: Any):
        value = self.dict.pop(key)
        self.sorted_list.discard((key, value))

    def __contains__(self, key: Any) -> bool:
        return key in self.dict

    def __len__(self) -> int:
        return len(self.dict)

    def pop(self, key: Any) -> Any:
        if key in self.dict:
            value = self.dict.pop(key)
            self.sorted_list.discard((key, value))
            return value
        else:
            raise KeyError(f"Key '{key}' not found")


def preproccess_texts(texts: List[str],
                      tokenizer: PreTrainedTokenizer,
                      min_words: int = 40,
                      max_words: int = 100) -> List[List[str]]:
    tokenized_texts = []
    for text in texts:
        n_words = len(text.split())
        if n_words < min_words:
            continue
        if n_words > max_words:
            # divide the text into chunks of max_words
            text_chunks = [text[i:i + max_words] for i in range(0, n_words, max_words)]
        else:
            text_chunks = [text]
        for text_chunk in text_chunks:
            tokenized_text = tokenizer.tokenize(text_chunk)
            tokenized_texts.append(tokenized_text)
    return tokenized_texts


def remove_numbers_ngrams(ngram_freqs: Dict[Tuple, int]) -> Dict[Tuple, int]:
    # remove ngrams that contain only numbers
    ngram_freqs_no_numbers = {ngram: freq for ngram, freq in ngram_freqs.items() if not all(token.isdigit() for token in ngram)}
    # log_w_wandb(logger=logger,message=f"Removed ngrams that contain only numbers, original ngrams freq size: {len(ngram_freqs)}, trimmed ngrams freq size: {len(ngram_freqs_no_numbers)}")
    return ngram_freqs_no_numbers


def remove_problematic_ngrams(ngram_freqs: Dict[Tuple, int]) -> Dict[Tuple, int]:
    # don't add <0xE2><0x82><0xA9> ngrams
    # to solve this: self.decode(converted_token_ids)
    #'According to the South Korean National Tax Service, the average annual earnings for a Korean idol in 2013 were KR<0xE2><0x82><0xA9>46.74 million (US$42,000). This was almost double the 2010 figure of KR<0xE2><0x82><0xA9>26.97 million (US$25,275), a rise attributable to the global spread of "Hallyu" in recent years.'
    #texts
    #'According to the South Korean National Tax Service, the average annual earnings for a Korean idol in 2013 were KR₩46.74 million (US$42,000). This was almost double the 2010 figure of KR₩26.97 million (US$25,275), a rise attributable to the global spread of "Hallyu" in recent years.'
    ngram_freqs_wout_problematics = defaultdict(int)
    for ngram, freq in ngram_freqs.items():
        prob_ngram = False
        for token in ngram:
            if '<0x' in token:
                # print(f"problematic ngram: {ngram}")
                prob_ngram = True
                break
        if not prob_ngram:
            ngram_freqs_wout_problematics[ngram] = freq
    # log_w_wandb(logger=logger,message=f"Removed ngrams that contain '<0x' kind of things, original ngrams freq size: {len(ngram_freqs)}, new ngrams freq size: {len(ngram_freqs_wout_problematics)}")
    return ngram_freqs_wout_problematics


def get_affected_ngrams(tokenized_texts: List[List[str]],
                        max_n: int, vocab_size: int,
                        min_freq_affected: int = 5) -> Tuple[Dict[Tuple, int], Dict[Tuple, Dict[Tuple, int]]]:
    # The function returns filtered ngram_freqs and affected_dict
    # we start by counting ngram frequencies
    ngram_freqs = defaultdict(int)
    for tokenized_text in tokenized_texts:
        for n in range(1, max_n + 1):
            for i in range(len(tokenized_text) - n + 1):
                ngram = tuple(tokenized_text[i:i + n])
                ngram_freqs[ngram] += 1
                
    ngram_freqs = remove_numbers_ngrams(ngram_freqs)
    ngram_freqs = remove_problematic_ngrams(ngram_freqs)
    
    unigram_freqs = {ngram: freq for ngram, freq in ngram_freqs.items() if len(ngram) == 1}
    ngram_freqs = {ngram: freq for ngram, freq in ngram_freqs.items() if len(ngram) > 1}
    # remove ngrams that contain only numbers or problematic tokens
    # filter ngram_freqs and keep only 5*vocab size ngrams (and all unigrams)
    ngram_freqs = dict(sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)[:5 * vocab_size])
    ngram_freqs.update(unigram_freqs)
    # now we find affected ngrams
    affected_dict = defaultdict(lambda: defaultdict(int))
    for tokenized_text in tqdm(tokenized_texts):
        # for each text, get ngrams and their locations
        ngrams_with_indices = []
        for n in range(1, max_n + 1):
            for i in range(len(tokenized_text) - n + 1):
                ngram = tuple(tokenized_text[i:i + n])
                if ngram in ngram_freqs:
                    ngrams_with_indices.append((ngram, i, i + n - 1))
        ngrams_with_indices = sorted(ngrams_with_indices, key=lambda x: (x[1], x[2]))
        # for each pair of ngrams, check if they overlap (affected)
        for i, (ngram1, start1, end1) in enumerate(ngrams_with_indices):
            ngram_len1 = len(ngram1)
            #  tokens (unigrams) do not affect ngrams, however, ngrams can affect tokens
            if ngram_len1 == 1:  # skip unigrams
                continue
            # this is for optimizing the search, notice that we only need to search for ngrams
            # that are within max_n * max_n tokens from the current ngram (otherwise, there is no overlap)
            start_range = max(0, i - max_n * max_n)
            end_range = min(len(ngrams_with_indices), i + max_n * max_n + 1)
            for j in range(start_range, end_range):
                if i != j:
                    ngram2, start2, end2 = ngrams_with_indices[j]
                    if max(start1, start2) <= min(end1, end2):
                        affected_dict[ngram1][ngram2] += 1
    # make affected_dict a regular dict
    affected_dict = {ngram: {ngram_a: freq for ngram_a, freq in ngram_affected.items() if freq >= min_freq_affected}
                     for ngram, ngram_affected in affected_dict.items()}
    return ngram_freqs, {k: dict(v) for k, v in affected_dict.items()}


def _add_token_step(queue_dict: SortedNgramDict,
                    current_dict: SortedNgramDict,
                    affected_dict: Dict[Tuple, Dict[Tuple, int]],
                    force_ngram: bool = False,
                    force_unigram: bool = False):
    assert not (force_ngram and force_unigram), "force_ngram and force_unigram cannot be True at the same time"
    ngram_to_add = None
    for ngram, freq in queue_dict.sorted_list:
        if not _is_ngram_addable(ngram):
            utils.log_w_wandb(logger=logger, message=f"Ngram {ngram} is not addable. Skipping.")
            continue
        if force_ngram and len(ngram) > 1:
            ngram_to_add = ngram
            break
        elif force_unigram and len(ngram) == 1:
            ngram_to_add = ngram
            break
        elif not force_ngram and not force_unigram:
            ngram_to_add = ngram
            break
    assert ngram_to_add is not None, "No ngram to add"
    ngram_to_add_freq = queue_dict.pop(ngram_to_add)
    if len(ngram_to_add) > 1:
        for affected_ngram, affected_freq in affected_dict.get(ngram_to_add, {}).items():
            if affected_ngram in queue_dict:
                queue_dict[affected_ngram] -= affected_freq
    current_dict[ngram_to_add] = ngram_to_add_freq
    return ngram_to_add, ngram_to_add_freq

def _is_token_removable(token: str) -> bool:
    if type(token) == tuple and len(token)==1:
        token = token[0]
    not_saved = token not in SAVED_TOKENS 
    longer_than_one = len(token) > 1
    saved_asci = '<0x' in token
    return not_saved and longer_than_one and (not saved_asci)

def _is_ngram_addable(ngram: Tuple[str]) -> bool:
    only_digits = all(token.isdigit() for token in ngram)
    is_asci= any('<0x' in token for token in ngram)
    return not only_digits and not is_asci


def _remove_token_step(queue_dict: SortedNgramDict,
                       current_dict: SortedNgramDict,
                       affected_dict: Dict[Tuple, Dict[Tuple, int]],
                       force_ngram: bool = False,
                       force_unigram: bool = False,
                       remove_saved_tokens:bool = False): #TODO: change here to False when prodcution
    assert not (force_ngram and force_unigram), "force_ngram and force_unigram cannot be True at the same time"
    ngram_to_remove = None
    for ngram, freq in current_dict.sorted_list[::-1]:
        if remove_saved_tokens or (not _is_token_removable(ngram)):
            utils.log_w_wandb(logger=logger, message=f"Token {ngram} is not removable. Skipping.")
            current_dict[ngram]=sys.maxsize
            continue
        if force_ngram and len(ngram) > 1:
            ngram_to_remove = ngram
            break
        elif force_unigram and len(ngram) == 1:
            ngram_to_remove = ngram
            break
        elif not force_ngram and not force_unigram:
            ngram_to_remove = ngram
            break
    assert ngram_to_remove is not None, "No ngram to remove"
    ngram_to_remove_freq = current_dict.pop(ngram_to_remove)
    if len(ngram_to_remove) > 1:
        for affected_ngram, affected_freq in affected_dict.get(ngram_to_remove, {}).items():
            if affected_ngram in queue_dict:
                queue_dict[affected_ngram] += affected_freq
    queue_dict[ngram_to_remove] = ngram_to_remove_freq
    return ngram_to_remove, ngram_to_remove_freq



def find_new_vocab(ngram_freqs: Dict[Tuple, int],
                   affected_dict: Dict[Tuple, Dict[Tuple, int]],
                   current_vocab: List[str],
                   vocab_size: int,
                   n_new_tokens: Optional[int] = None) -> Dict[Union[str, Tuple[str]], int]:
    ratio_diff=max(vocab_size/32_000,32_000/vocab_size)
    if ratio_diff<1.5:
        add_diff=2
    elif ratio_diff<2:
        add_diff=3
    elif ratio_diff<3:
        add_diff=4
    elif ratio_diff<4:
        add_diff=5
    elif ratio_diff<5:
        add_diff=6
    else:
        add_diff=7
        
    current_dict = SortedNgramDict()
    queue_dict = SortedNgramDict()
    current_vocab = set([tuple([token]) for token in current_vocab])
    ngram_freqs.update({ngram: 0 for ngram in current_vocab if ngram not in ngram_freqs})
    for ngram, freq in ngram_freqs.items():
        if ngram in current_vocab:
            current_dict[ngram] = freq
        else:
            if _is_ngram_addable(ngram):
                queue_dict[ngram] = freq

    # check if target vocab size and number of ngrams to add are reachable
    assert len(current_dict) + len(queue_dict) > vocab_size, "The target vocab size cannot be reached."
    if n_new_tokens is not None:
        assert len(queue_dict) >= n_new_tokens, "The number of ngrams to add cannot be reached."
        assert len(current_dict) + n_new_tokens >= vocab_size, "The target vocab size cannot be reached."
    # start iterations to fix the vocab
    max_iters = max(len(current_dict), vocab_size)
    # max_iters = 100
    for i in range(max_iters):
        current_size = len(current_dict)
        if len(queue_dict) == 0:
            print(f"No more tokens left in the queue. Final vocab size: [{current_size}/{vocab_size}]")
            break
        add_steps_per_iter = add_diff if current_size < vocab_size else 1
        remove_steps_per_iter = add_diff if current_size > vocab_size else 1
        # remove tokens from vocab
        for _ in range(remove_steps_per_iter):
            ngram_to_remove, ngram_to_remove_freq = _remove_token_step(queue_dict, current_dict, affected_dict)
            
        # add tokens to vocab
        for _ in range(add_steps_per_iter):
            ngram_to_add, ngram_to_add_freq = _add_token_step(queue_dict, current_dict, affected_dict)
        #print(f"[{current_size}/{vocab_size}]\t\tRemoved: {ngram_to_remove} - Freq: {ngram_to_remove_freq}"
        #      f"\t\tAdded: {ngram_to_add} - Freq: {ngram_to_add_freq}")
        
        # stop if there is no change
        if len(ngram_to_add) * ngram_to_add_freq == len(ngram_to_remove) * ngram_to_remove_freq:
            break
    
    while len(current_dict) > vocab_size:
        ngram_to_remove, ngram_to_remove_freq = _remove_token_step(queue_dict, current_dict, affected_dict)
        print(f"REMOVING EXTRA TOKENS: [{len(current_dict)}/{vocab_size}]\t\tRemoved: {ngram_to_remove} - Freq: {ngram_to_remove_freq}")  
    # vocab is fixed, now check if we need to add or remove ngrams according to number_ngrams_to_add
    if n_new_tokens is not None:
        number_of_ngrams_added = len([ngram for ngram in current_dict.dict if len(ngram) > 1])
        need_to_add = n_new_tokens - number_of_ngrams_added
        # If `need_to_add` > 0, we need to add ngrams. If `need_to_add` < 0, we need to remove ngrams.
        remove_force_ngram = False if need_to_add > 0 else True
        remove_force_unigram = True if need_to_add > 0 else False
        add_force_ngram = True if need_to_add > 0 else False
        add_force_unigram = False if need_to_add > 0 else True
        if need_to_add != 0:
            # remove steps
            for i in range(abs(need_to_add)):
                removed_token, removed_freq = _remove_token_step(
                    queue_dict, current_dict, affected_dict,
                    force_ngram=remove_force_ngram, force_unigram=remove_force_unigram)
        #        print(f"Removing [{abs(need_to_add) - i}/0]\t\tRemoved: {removed_token} - Freq: {removed_freq}")
            # add steps
            for i in range(abs(need_to_add)):
                added_token, added_freq = _add_token_step(
                    queue_dict, current_dict, affected_dict,
                    force_ngram=add_force_ngram, force_unigram=add_force_unigram)
        #        print(f"Adding [{i}/{abs(need_to_add)}]\t\tAdded: {added_token} - Freq: {added_freq}")
    vocab_freqs = {ngram if len(ngram) > 1 else ngram[0]: freq for ngram, freq in current_dict.sorted_list}
    return vocab_freqs


if __name__ == '__main__':
    outputs_folder = "/data/home/nitay/dev24/GenExplain/adaptive_vocab"
    data = load_dataset('vblagoje/cc_news')
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = 15000
    max_n = 4
    min_freq_affected = 2
    current_vocab = list(tokenizer.get_vocab().keys())

    tokenized_texts = preproccess_texts(data['train']['text'][:10000], tokenizer)
    save_json(os.path.join(outputs_folder, 'tokenized_texts.json'), {'tokenized_texts': tokenized_texts})
    tokenized_texts = load_json(os.path.join(outputs_folder, 'tokenized_texts.json'))['tokenized_texts']

    ngram_freqs, affected_dict = get_affected_ngrams(tokenized_texts, max_n, vocab_size, min_freq_affected)

    ngram_freqs_json = {str(k): v for k, v in ngram_freqs.items()}
    affected_dict_json = {str(k): {str(k_a): v_a for k_a, v_a in v.items()} for k, v in affected_dict.items()}
    save_json(os.path.join(outputs_folder, 'ngram_freqs.json'), {'ngram_freqs': ngram_freqs_json})
    save_json(os.path.join(outputs_folder, 'affected_dict.json'), {'affected_dict': affected_dict_json})
    ngram_freqs = load_json(os.path.join(outputs_folder, 'ngram_freqs.json'))['ngram_freqs']
    ngram_freqs = {eval(k): v for k, v in ngram_freqs.items()}
    affected_dict = load_json(os.path.join(outputs_folder, 'affected_dict.json'))['affected_dict']
    affected_dict = {eval(k): {eval(k_a): v_a for k_a, v_a in v.items()} for k, v in affected_dict.items()}

    new_vocab = find_new_vocab(ngram_freqs, affected_dict, current_vocab, vocab_size, n_new_tokens=5000)

    
    new_vocab = {str(k): v for k, v in new_vocab.items()}
    # save_json(os.path.join(outputs_folder, 'new_vocab.json'), {'new_vocab': new_vocab})
    # for token, freq in new_vocab.items():
    #     print(f"{token}: {freq}")
    debug = True