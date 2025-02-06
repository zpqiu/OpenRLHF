import torch
import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = completions
    rewards = []
    for content, sol in zip(contents, solution):
        # gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        gold_parsed = sol
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = completions
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def extract_qwen_output(prompt):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', prompt, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    return model_output


def reward_func(queries, prompts, **kwargs):
    # queries is prompts + responses
    # print(queries)
    answers = kwargs['answers']
    print(answers)
    responses = [extract_qwen_output(query) for query in queries]
    print(f"responses_count: {len(responses)}, answers_count: {len(answers)}")
    
    # 确保responses非空
    if not responses:
        return torch.tensor([0.0] * len(queries))
    
    accuracy_rewards = accuracy_reward(responses, answers)
    format_rewards = format_reward(responses)
    print(f"accuracy_rewards_count: {len(accuracy_rewards)}, format_rewards_count: {len(format_rewards)}")
    # 确保两个奖励列表长度相同
    if len(accuracy_rewards) != len(format_rewards):
        print(f"Warning: Reward lengths mismatch - accuracy: {len(accuracy_rewards)}, format: {len(format_rewards)}")
        # 使用较短的长度
        min_len = min(len(accuracy_rewards), len(format_rewards))
        accuracy_rewards = accuracy_rewards[:min_len]
        format_rewards = format_rewards[:min_len]
    
    return torch.tensor(accuracy_rewards) + torch.tensor(format_rewards)
