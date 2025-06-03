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

import sys
import re
import concurrent.futures

from verl import DataProto
from verl.utils.reward_score import verify_system
import torch
from collections import defaultdict

class VerifySystemRewardManager:
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 max_resp_len=None,
                 overlong_buffer_cfg=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    # TODO: Is this still necessary in algorithms other than PRIME?
    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        reward_input = []
        valid_responses_length = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_responses_length.append(valid_response_length)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            # if i == 0:
            #     print(f"prompt_str: \n{[prompt_str]}")
            #     print(f"response_str: \n{[response_str]}")
            # pattern_with_prompt = re.compile(r'user\n(.*?)<\|im_end\|>', re.DOTALL)
            pattern_with_prompt = re.compile(r'<｜User｜>(.*?)<｜Assistant｜>', re.DOTALL)
            prompt_match = pattern_with_prompt.search(prompt_str)
            prompt_str = prompt_match.group(0).strip()  #edit

            # response_str = response_str.replace('<|im_end|> <|endoftext|>', '')
            response_str = response_str.replace('<｜end▁of▁sentence｜>', '') 
            # if i == 0:
            #     print(f"format_prompt_str: \n{[prompt_str]}")
            #     print(f"format_response_str: \n{[response_str]}")

            # protocol = data_item.non_tensor_batch['protocol']
            protocol = "deep_thinking"
            verifier_config = data_item.non_tensor_batch['verifier']

            #modify by boye
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            reward_input.append([prompt_str, response_str, protocol, verifier_config,ground_truth])
            

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1


        flag = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            future_to_index = {}
            done_examples = 0
            # 直接对 reward_input 进行并发处理
            for index, sequence in enumerate(reward_input):
                query, response, protocol, verifier_config, ground_truth = sequence
                
                # import ipdb; ipdb.set_trace()

                future = executor.submit(verify_system.compute_score, query, response, protocol, verifier_config, ground_truth)
                future_to_index[future] = index
                
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                score: float
                if isinstance(result, dict):
                    score = result["score"]
                    # Store the information including original reward
                    for key, value in result.items():
                        reward_extra_info[key].append(value)
                else:
                    score = result
                reward = score
                
                query, response, protocol, verifier_config, ground_truth = reward_input[index]
                valid_response_length = valid_responses_length[index]

                if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                    overlong_buffer_len = self.overlong_buffer_cfg.len
                    expected_len = self.max_resp_len - overlong_buffer_len
                    exceed_len = valid_response_length - expected_len
                    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                    reward += overlong_reward
                    if self.overlong_buffer_cfg.log:
                        reward_extra_info["overlong_reward"].append(overlong_reward)
                        reward_extra_info["overlong"].append(overlong_reward < 0)

                
                # if flag == True:
                #     print("*"*30)
                #     print(f"Query:\n{[query]}\n\nResponse:\n{[response]}\n\nVerifier Config:\n{verifier_config}\n\n")
                #     if isinstance(result, dict):
                #         for key, value in result.items():
                #             print(f"[{key}]", value)
                #     else:
                #         print(f"[score]", score)
                #     print(f"Index {index}: {reward}")
                #     print("*"*30)
                #     flag = False

                reward_tensor[index, valid_responses_length[index] - 1] = reward
                done_examples += 1
                # update_progress_bar(done_examples, len(reward_input))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
