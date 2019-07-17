
def to_token(sentence, max_seq_length=256):
    # Tokenized input
    text = "[CLS]" + sentence
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    padding = [0] * (max_seq_length - len(indexed_tokens))
    tokens = indexed_tokens + padding
    segments = [0] * len(indexed_tokens) + padding
    masks = [1] * len(indexed_tokens) + padding

    print(len(tokens))

    return tokens, segments, masks

