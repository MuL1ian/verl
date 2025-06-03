# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import json
import time
import uuid
import requests

from verl.utils.reward_score.deepscaler_math.utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

# def post_reward(data, timeout=600):
#     # url = "http://verification-webapi.dev.dataeng.appspace.baidu.com/api/v1/reward"
#     url = "http://10.11.153.88:8101/api/v1/reward"
#     headers = {"content-type": "application/json"}
#     res = requests.post(
#         url, data=json.dumps(data), headers=headers, timeout=timeout
#     )
#     return res.json()

# def post_reward_result(data, timeout=600):
#     # url = "http://verification-webapi.dev.dataeng.appspace.baidu.com/api/v1/reward/result"
#     url = "http://10.11.153.88:8101/api/v1/reward/result"
#     headers = {"content-type": "application/json"}
#     res = requests.post(
#         url, data=json.dumps(data), headers=headers, timeout=timeout
#     )
#     return res.json()

def process_item(log_id, src, response, verifier_config, max_retries=100):
    data = {
        "data_id": log_id,
        "repeat_id": 0,
        "task_id": log_id,
        "run_id": log_id,
        "repeat_count": 1,
        "max_wait_time": 1000,
        "need_nl_feedback": 1,
        "protocol": "normal",
        "data": {
            "system": "",
            "src": [
                src
            ],
            "tgt": [],
            "response": response,
            "verifier": [
                verifier_config
            ]
        }
    }
    while max_retries>0:
        post_results = post_reward(data)
        if post_results["code"] == 0 and post_results["msg"] == "OK":
            time.sleep(0.1)
            max_retries -= 1
            return
    raise Exception("Stop!")

def process_reward(log_id, max_retries=400):
    data = {
        "task_id": log_id,
        "items": [
            {
                "run_id": log_id,
                "data_id": log_id
            }
        ]
    }
    while max_retries>0:
        if max_retries <= 100:
            print(f"[{log_id}] Reward Waiting!")
        reward_result = post_reward_result(data)
        if reward_result["status"] != "COMPLETE":
            time.sleep(2)
            max_retries -= 1
            continue
        reward_result = reward_result["data"][0]["rewards"][0]
        return reward_result
    raise Exception("Stop!")

def format_check(text):
    text = text.strip()
    # 提取内容的正则表达式
    extract_pattern = r'^<think>(.+)</think>(.+)$'
    # 提取内容
    match = re.search(extract_pattern, text, re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        answer_content = match.group(2).strip()
        return True, think_content, answer_content
    else:
        return False, None, None

def compute_score(query_str, solution_str, protocol, verifier_config, ground_truth=None, **kwargs):
    # import ipdb; ipdb.set_trace()
    """Compute EM score by extracting final answer and comparing to ground truth"""
    
    if protocol == "deep_thinking":
        format_follow_flag, think_content, answer_content = format_check(solution_str)
        if not format_follow_flag:
            return {"score": 0.0, "acc": 0}
    else:
        answer_content = solution_str

    if ground_truth is None:
        print("[Warning] No ground truth provided for EM scoring.")
        return {"score": 0.0, "acc": 0}

    model_answer = extract_answer(answer_content)

    # import ipdb; ipdb.set_trace()

    is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)

    return {"score": float(is_correct), "acc": int(is_correct)}
    
# def compute_score(query_str, solution_str, protocol, verifier_config, ground_truth=None):
#     """The scoring function for GSM8k.

#     Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
#     if protocol == "deep_thinking":

#         format_follow_flag, think_content, answer_content = format_check(solution_str)

#         # import ipdb; ipdb.set_trace()
        
#         if not format_follow_flag:
#             return {"score": 0.0, "acc": 0}
#     else:
#         answer_content = solution_str

#     while True:
#         try:
#             log_id = str(uuid.uuid4())
#             process_item(log_id, query_str, answer_content, verifier_config)
#             reward_result = process_reward(log_id)
#             if reward_result is None or reward_result < 1.0:
#                 reward_result = 0
#             assert reward_result in [0, 1, 0.0, 1.0]
#             return {"score": reward_result, "acc": int(reward_result)}
#         except:
#             print(f"[{log_id}] Reward Failed！")
#             continue