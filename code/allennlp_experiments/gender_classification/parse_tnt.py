import pandas as pd
import re


def parse_attribute(text, attr_name):
    left_pos = text.find(f'{attr_name}="')
    if left_pos == -1:
        return None
    left_pos += len(f'{attr_name}="')
    right_pos = text.find('"', left_pos)
    return text[left_pos:right_pos]
    
def is_good_token(text):
    return re.match(r'\<.*\>', text) is None and 'http' not in text
    
    
def parse_tnt(filename, text_count, min_words_in_text=20):
    result_df = pd.DataFrame()
    cur_gender = ""
    cur_tokens = []
    cur_author = ""
    
    with open(filename, 'r') as fin:
        for line in fin:
            line = line.strip()
            #print(line)
            if line.startswith('<text id'):
                if cur_gender != "" and len(cur_tokens) >= min_words_in_text:
                    result_df = result_df.append(
                        {'text': ' '.join(cur_tokens), 'target': cur_gender, "author": cur_author},
                        ignore_index=True
                    )
                if result_df.shape[0] >= text_count:
                    break
                cur_gender = parse_attribute(line, 'gender')
                cur_author = parse_attribute(line, 'author')
                #print(cur_gender)
                cur_tokens = []
            else:
                token = line.split('\t')[0]
                if is_good_token(token) or token == '<author>':
                    cur_tokens.append(token)
                elif token == '</author>':
                    cur_author_list = []
                    while cur_tokens[-1] != '<author>':
                        cur_author_list.append(cur_tokens[-1])
                        cur_tokens = cur_tokens[:-1]
                    cur_tokens = cur_tokens[:-1]
                    cur_author = ' '.join(cur_author_list[::-1])
    if cur_gender != "" and len(cur_tokens) >= min_words_in_text:
        result_df = result_df.append(
            {'text': ' '.join(cur_tokens), 'target': cur_gender, "author": cur_author},
            ignore_index=True
        )   
    return result_df   