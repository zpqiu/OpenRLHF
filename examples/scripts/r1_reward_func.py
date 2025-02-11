import torch
import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def calculate_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = completions
    rewards = []
    for content, sol in zip(contents, solution):
        if content.strip() == "":
            rewards.append(0.0)
            continue
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
            if verify(answer_parsed, gold_parsed):
                reward = 1.0
            else:
                reward = -0.5
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


# case = '<think>123</think><think>123</think><answer>456</answer>'
def is_format_correct(completion):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
        return False
    # check if all tags only appear once
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        if completion.count(tag) != 1:
            return False
    return True

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
        count -= len(text.split("</answer>")[-1])*0.001
    if text.count("</answer>") == 1:
        count += 0.125
        count -= (len(text.split("</answer>")[-1]) - 1)*0.001
    return count

def calculate_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = completions
    return [1.0 if is_format_correct(content) else -1.0 for content in completion_contents]


def extract_qwen_output(prompt):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', prompt, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    model_output = model_output[len("<|im_start|>assistant"):].strip()
    return model_output


def extract_answer_part(response):
    pattern = r"<answer>(.*?)</answer>$"
    match = re.search(pattern, response)
    if match:
        return match.group(1)
    return ""

def reward_func(queries, prompts, **kwargs):
    # queries is prompts + responses
    # print(queries)
    answers = kwargs['answers']
    # print(answers)
    responses = [extract_qwen_output(query) for query in queries]
    final_answers = [extract_answer_part(response) for response in responses]
    # print(f"responses_count: {len(responses)}, answers_count: {len(answers)}")
    print(f"Response Case: {responses[0]}")
    print(f"Answer Case: {final_answers[0]}")

    # # 确保responses非空
    # if not responses:
    #     return torch.tensor([0.0] * len(queries))
    format_rewards = calculate_format_reward(responses)
    accuracy_rewards = calculate_accuracy_reward(final_answers, answers)

    final_rewards = []
    for final_answer, format_reward, accuracy_reward in zip(final_answers, format_rewards, accuracy_rewards):
        if format_reward == -1.0:
            final_rewards.append(format_reward)
        elif final_answer.strip() != "":
            final_rewards.append(accuracy_reward)
        else:
            final_rewards.append(-0.5)
    
    return torch.tensor(final_rewards)
