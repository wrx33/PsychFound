import ollama
response = ollama.chat(model='qwen2', messages=[
  {
    'role': 'user',
    'content': '为什么天空是蓝色的？',
  },
])
print(response['message']['content'])
