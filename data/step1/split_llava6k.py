import os
import json
import random
from collections import defaultdict

# 加载 JSON 文件数据
sft_root = "/workspace/tanwentao1/self-reward/SeVa-main/llava-data/llava_v1_5_mix665k.json"
sft_file = json.load(open(sft_root, "r"))

message_dict = defaultdict(list)

# 定义批量处理函数
for item in sft_file:
    if 'image' not in item.keys():
        continue

    prefix = item['image'].split('/')[0]
    image_path = os.path.join('/workspace/tanwentao1/self-reward/SeVa-main/llava-data', item['image'])
    if not os.path.exists(image_path):
        continue

    if prefix in ['ocr_vqa', 'textvqa', 'coco', 'gqa', 'vg']:
        message_dict[prefix].append(item)

# 随机打乱每个数据集
for key in message_dict:
    random.shuffle(message_dict[key])
    
# 存储所有数据集的索引
indices = {key: 0 for key in message_dict}


for i in range(1):
    sampled_data = []
    for key in message_dict:
        sampled_data.extend(message_dict[key][indices[key]:indices[key]+1200])
        indices[key] += 1200
    random.shuffle(sampled_data)
    json_output_path = f'llava6k_{i}.json'
    with open(json_output_path, 'w') as json_file:
        json.dump(sampled_data, json_file, indent=4)

    print(f"File {json_output_path} saved with {len(sampled_data)} samples.")

print("All JSON files saved successfully!")
