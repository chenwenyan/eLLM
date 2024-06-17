import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2,
                        best_of=5,
                        temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=3, best_of=3, use_beam_search=True,
                        temperature=0.0)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    time_to_first_tokens = []
    time_per_output_tokens = []

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()


        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                # 计算每个请求第一个token生成的时间以及后续token生成的平均时间
                time_to_first_token = request_output.metrics.first_token_time - request_output.metrics.arrival_time
                time_per_output_token = (request_output.metrics.finished_time - request_output.metrics.first_token_time)/len(request_output.outputs[0].token_ids)
                print(f"First token time: {time_to_first_token:.3f}s, "
                      f"Avg token time: {time_per_output_token:.3f}s")
                time_to_first_tokens.append(time_to_first_token)
                time_per_output_tokens.append(time_per_output_token)

    print(f"Average time to first token: {sum(time_to_first_tokens)/len(time_to_first_tokens):.3f}s") 
    print(f"Average time per token: {sum(time_per_output_tokens)/len(time_per_output_tokens):.3f}s")       


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
