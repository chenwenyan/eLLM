from transformers import pipeline
pipe = pipeline('text-generation', model='meta-llama/Llama-2-7b-hf', device=0, use_cache=False)
# Expected:
# 'Once upon a time, there was a little
output = pipe("Once upon a time,", max_length=50, do_sample=True, temperature=0.9)
print(output)