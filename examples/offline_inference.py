from vllm import LLM, SamplingParams
import argparse

# Create an argument parser.
parser = argparse.ArgumentParser()
# Add an argument for the model path.
parser.add_argument("--model", type=str, required=True, help="Path to the model directory.")
# Parse the arguments.
args = parser.parse_args()


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# 添加参数，使得当GPU memory不够时，可以直接采用recomputation的方式进行预测
llm = LLM(model=args.model, enable_chunked_prefill=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
