import json
data = []
for i in range(8):
    with open(f'/path2/seed-3k-{i}.json','r') as file:
        data += json.load(file)
print(len(data))

count = 0
filter_data = []
with open('dpo_llava6k_0_chatQuestion_corGen_improveAll_ourLoss-scores.json', 'w') as file:
    for item in data:
        if item['chosen_logps_normed'] > item['rejected_logps_normed']:
            count+=1
        filter_data.append(item)
    json.dump(filter_data, file, indent=4, ensure_ascii=False)
print(len(filter_data))
print(count)