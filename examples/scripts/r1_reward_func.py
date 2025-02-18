import torch
import re
import random
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, StringExtractionConfig, ExprExtractionConfig, parse, verify


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def calculate_accuracy_reward(completions, solution, do_print=False):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = completions
    rewards = []
    for content, sol in zip(contents, solution):
        if content.strip() == "":
            rewards.append(-1)
            continue

        last_boxed_str = last_boxed_only_string(content)
        if last_boxed_str is None:
            rewards.append(-1)
            continue

        # remove \boxed
        if last_boxed_str[7:-1].strip() == sol.strip():
            rewards.append(1.0)
            continue

        gold_parsed = parse(f"\\boxed{{{sol}}}", extraction_mode="first_match", extraction_config=[
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
                    ),
                    ExprExtractionConfig(),
                    # StringExtractionConfig()
                ],)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                last_boxed_str,
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
                    ),
                    ExprExtractionConfig(),
                    # StringExtractionConfig()
                ],
                extraction_mode="first_match",
            )
            if do_print:
                print(f"[answer_parsed]: {answer_parsed}  <===> [gold_parsed]: {gold_parsed}")

            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            if verify(answer_parsed, gold_parsed):
                reward = 1.0
            else:
                reward = -1.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 0.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


# case = '<think>123</think><think>123</think><answer>456</answer>'
def is_format_correct(completion):
    # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    pattern = r"^<think>.*?</think>"
    if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
        return False
    # check if all tags only appear once
    # tags = ["<think>", "</think>", "<answer>", "</answer>"]
    tags = ["<think>", "</think>"]
    for tag in tags:
        if completion.count(tag) != 1:
            return False
    
    # check if <think>...</think> is empty
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, completion, re.DOTALL | re.MULTILINE)
    if think_match and think_match.group(1).strip() == "":
        return False
    
    return True

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
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    return ""

def reward_func(queries, prompts, **kwargs):
    # queries is prompts + responses
    answers = kwargs['answers']
    responses = [extract_qwen_output(query) for query in queries]
    # # extract <answer>...</answer>
    # final_answers = [extract_answer_part(response) for response in responses]

    do_print = False
    if random.randint(0, 3) == 1:  
        do_print = True
        
    if do_print:
        print(f"Response Case: {responses[0]}")
        # print(f"Answer Case: {final_answers[0]}")

    format_rewards = calculate_format_reward(responses)
    accuracy_rewards = calculate_accuracy_reward(responses, answers, do_print=do_print)

    final_rewards = []
    for format_reward, accuracy_reward in zip(format_rewards, accuracy_rewards):
        if accuracy_reward == 0.0:
            # will skip this example in reinforce algo
            final_rewards.append(0.0)
        elif accuracy_reward == 1.0 and format_reward == 1.0:
            final_rewards.append(1.0)
        elif accuracy_reward == -1.0 and format_reward == 1.0:
            final_rewards.append(-0.5)
        elif accuracy_reward == 1.0 and format_reward == -1.0:
            final_rewards.append(0.5)
        else:
            final_rewards.append(-1.0)
    
    return torch.tensor(final_rewards)
