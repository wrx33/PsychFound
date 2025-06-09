import pandas as pd
import numpy as np
import random

# df = pd.read_parquet('/home/sjtu/wrx/code/TinyZero-main/data/medication-instruct-hint-0228-1200-shuffle-400/train.parquet')
df = pd.read_parquet('/home/sjtu/wrx/code/TinyZero-main/data/diagnosis-instruct-hint-icd10-0224-1200-shuffle-400-simple/train.parquet')

prompt_lens = []
for i in range(len(df)):
    prompt_lens.append(len(df['prompt'][i][0]['content']))

print(np.max(prompt_lens))
print(len(prompt_lens))

index = random.randint(0, 400)
print(df.iloc[index]['prompt'])
print(df.iloc[index]['diagnosis'])