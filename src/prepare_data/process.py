import json
import re

path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_A3/distillation_dpo_A3.jsonl'
data = []
with open(path, 'r', encoding='utf-8') as f1:
    for line in f1:
        newline = eval(line)
        data.append(newline)
f1.close()

for item in data:
    query = item['conversations'][0]['value']
    chosen = item['chosen']['value']
    rejected = item['rejected']['value']
    try:

        answer = re.search(r"(<answer>.*?</answer>)", chosen, re.DOTALL).group(1)
        gt = rejected.split('\n')[0]
        if gt in answer:
            with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_A3/distillation_dpo_A3_gt.jsonl', 'a') as f2:
                json_str = json.dumps(item, ensure_ascii=False)
                f2.write(json_str + '\n')
                f2.flush()
        else:
            conversation = []
            conversation.append({'from': 'human', 'value': query})
            chosen = {"from": "gpt", "value": chosen.replace(answer, f"\n <answer>{gt}</answer>")}
            rejected = {"from": "gpt", "value": rejected}
            conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

            with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_A3/distillation_dpo_A3_gt.jsonl', 'a') as f2:
                json_str = json.dumps(conversations, ensure_ascii=False)
                f2.write(json_str + '\n')
                f2.flush()
    except:
        gt = rejected.split('\n')[0]

        conversation = []
        conversation.append({'from': 'human', 'value': query})
        chosen = {"from": "gpt", "value": chosen + f"\n <answer>{gt}</answer>"}
        rejected = {"from": "gpt", "value": rejected}
        conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

        with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_A3/distillation_dpo_A3_gt.jsonl', 'a') as f2:
            json_str = json.dumps(conversations, ensure_ascii=False)
            f2.write(json_str + '\n')
            f2.flush()


