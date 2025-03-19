import json
import random
import nltk
import re
import argparse
from collections import Counter

# 确保下载了必要的数据包
# nltk.download('punkt')

def has_duplicate_sentences_or_phrases(text):
    # 将文本分割为句子
    sentences = nltk.sent_tokenize(text)

    # 创建一个集合用于检测重复句子
    seen_sentences = set()
    for sentence in sentences:
        if sentence in seen_sentences:
            print(sentences)
            return True
        seen_sentences.add(sentence)

    # 使用正则表达式匹配短语
    phrases = nltk.word_tokenize(text)
    phrase_counts = Counter(phrases)

    # 检查是否有重复的短语
    for count in phrase_counts.values():
        if count > 30:
            print(sentences)
            print(phrase_counts)
            return True

    return False

def is_non_text_char(char):
    """检查字符是否为非字母和非数字字符"""
    return not ('A' <= char <= 'Z' or 'a' <= char <= 'z' or '0' <= char <= '9')

def discard_if_majority_non_text(s):
    """检查字符串中非文本字符是否超过一半，如果是则丢弃"""
    non_text_chars = sum(1 for char in s if is_non_text_char(char))
    total_chars = len(s)
    try:
        if non_text_chars / total_chars > 0.5:
            return True
        else:
            return False
    except:
        return True
    
def main(args):
    choosen_file = f"{args.filename}_improve.jsonl"
    choosen_file_ori = f"{args.filename}_ans.jsonl"
    rejected_file = f"{args.filename}_dnAns.jsonl"

    choosen_lines = open(choosen_file, "r", encoding='utf-8').readlines()
    choosen_file_ori = open(choosen_file_ori, "r", encoding='utf-8').readlines()
    rejected_lines = open(rejected_file, "r", encoding='utf-8').readlines()

    message = []

    for cline, rline, cline_ori in zip(choosen_lines, rejected_lines, choosen_file_ori):
        cline = json.loads(cline)
        rline = json.loads(rline)
        cline_ori = json.loads(cline_ori)

        if 'name' in cline['question']:
            print(cline['question'])
            continue

        if has_duplicate_sentences_or_phrases(cline['answer']):
            continue

        if has_duplicate_sentences_or_phrases(rline['answer']):
            continue

        if discard_if_majority_non_text(cline['answer']) or discard_if_majority_non_text(rline['answer']):continue

        assert cline['image'] == rline['image']

        if cline['answer'] == rline['answer']:
            continue

        cans = cline['answer'].replace('</s>', '').replace('\n', '')
        rans = rline['answer'].replace('</s>', '').replace('\n', '')

        if cans == 'Yes' or rans == 'Yes':
            continue

        if cans == 'No' or rans == 'No':
            continue

        if rans.lower() in cans:
            continue
        
        item = {}
        item['chosen'] = cans
        item['reject'] = rans
        item['question'] = cline['question']
        item['image'] = cline['image']
        item['type'] = cline['type']

        message.append(item)

    json.dump(message, open(f"{args.filename}_noScores.json", "w", encoding='utf-8'))
    print(len(choosen_lines))
    print(len(message))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some jsonl files.")
    parser.add_argument('--filename', type=str, required=True, help='Base filename to process')
    
    args = parser.parse_args()
    main(args)
