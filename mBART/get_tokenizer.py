from transformers import MBartTokenizer

def get_tokenizer(tokenizer_path):
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained(tokenizer_path)
    return tokenizer