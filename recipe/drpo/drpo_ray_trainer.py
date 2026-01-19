# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import collate_fn


class RayDRPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_cfg = getattr(self.config, "data", {})
        self.max_prompt_length = int(getattr(data_cfg, "max_prompt_length", 1024))
        self.apply_chat_template_kwargs = getattr(data_cfg, "apply_chat_template_kwargs", {})
        self.truncation = getattr(data_cfg, "truncation", "error")

        client_cfg = getattr(self.config, "client", {})
        self.api_key = getattr(client_cfg, "api_key", "")
        self.base_url = getattr(client_cfg, "base_url", "")
        self.client_model = getattr(client_cfg, "client_model", "qwen3-max")
        self.temperature = getattr(client_cfg, "temperature", 1.0)
        self.top_p = getattr(client_cfg, "top_p", 0.95)
        self.max_tokens = getattr(client_cfg, "max_tokens", 512)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.prefix = "You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}."

        self.rephrase_prompt = (
            "You are an expert in designing mathematical problems. Your objective is to rephrase a given problem "
            "to create a linguistic variant, ensuring the underlying logic and requirements remain identical.\n\n"
            "[TASK]\n"
            "Given the ORIGINAL_PROBLEM and ORIGINAL_ANSWER below, produce a AUGMENTED_PROBLEM. You should rewrite the text by changing "
            "sentence structure, vocabulary, tone, or formatting. You may convert a wordy problem into a concise one, "
            "or expand a dense problem into a more descriptive one, as long as the meaning is preserved.\n"
            "**Output ONLY the rewritten problem text directly.**\n\n"
            "[CRITICAL CONSTRAINTS]\n"
            "1. SEMANTIC EQUIVALENCE: The specific task, constraints, input/output specifications, examples, and rules "
            "must remain functionally equivalent. Do NOT change any numbers, variable names, formulas, or specific data values.\n"
            "2. NO SOLUTION: Do NOT solve the problem. Do NOT provide hints, reasoning, steps, or the final answer. "
            "Do NOT make assumptions about the solution method.\n"
            "3. NO AMBIGUITY: The rewriting must not introduce ambiguity or lead to a different interpretation of the requirements.\n"
            "4. INTEGRITY: Do not omit any conditions or edge cases present in the original text.\n\n"
            "[EXAMPLE]\n"
            "Original_Problem: \n"
            "Solve for x: 3x + 5 = 20. Then compute y = 2x - 1. What is the value of y?\n"
            "Original_Answer: \n"
            "9\n"
            "JSON_Output:\n"
            "{{\n"
            "  \"rewritten_problem\": \"Find the number x that satisfies the equation 3x + 5 = 20. Using that x, evaluate y = 2x - 1. Report the value of y.\",\n"
            "  \"answer\": \"9\"\n"
            "}}\n\n"
            "[OUTPUT FORMAT]\n"
            "You must output ONLY a single JSON object (no markdown, no code blocks):\n"
            "- \"rewritten_problem\": (string) The rewritten problem.\n"
            "- \"answer\": (string/number) The answer to the rewritten problem (the same as the ORIGINAL_ANSWER).\n\n"
            "[ORIGINAL_PROBLEM]\n"
            "{problem}\n"
            "[ORIGINAL_ANSWER]\n"
            "{answer}"
        )

        self.attack_prompt = (
            "You are an expert in designing mathematical problems. Your objective is to create a variant of a given problem "
            "by injecting irrelevant context, ensuring the core task remains identical.\n\n"
            "[TASK]\n"
            "Given the ORIGINAL_PROBLEM and ORIGINAL_ANSWER below, produce an AUGMENTED_PROBLEM that includes additional context. "
            "The added context should appear topically related (e.g., alternative unused options, or descriptive details), "
            "but must be mathematically and logically irrelevant to the specific question asked. \n"
            "**Output ONLY the rewritten problem text directly.**\n\n"
            "[CRITICAL CONSTRAINTS]\n"
            "1. UNCHANGED LOGIC: The specific task, constraints, input data, rules, and correct answer must remain "
            "EXACTLY the same. Do not alter any numbers, variable names, or requirements from the original text.\n"
            "2. NO SOLUTION: Do NOT solve the problem. Do NOT provide hints, reasoning, steps, or the final answer. "
            "Do NOT make assumptions about the solution method.\n"
            "3. SEEMINGLY RELEVANT NOISE: The added text should blend in naturally. "
            "Use 'distractors'—numbers or details that look like they belong in the problem domain but are not needed for the solution. "
            "It must not introduce conflicts with the original constraints.\n\n"
            "[EXAMPLE]\n"
            "Original_Problem: \n"
            "A factory produces 500 widgets per hour. How many widgets are produced in 8 hours?\n"
            "Original_Answer: \n"
            "4000\n"
            "JSON_Output:\n"
            "{{\n"
            "  \"rewritten_problem\": \"A factory produces 500 widgets per hour. The factory employs 50 workers across 2 shifts, and last year's production average was only 450 widgets per hour due to older machinery. Assuming the current production rate holds steady, how many widgets are produced in 8 hours?\",\n"
            "  \"answer\": \"4000\"\n"
            "}}\n\n"
            "[OUTPUT FORMAT]\n"
            "You must output ONLY a single JSON object (no markdown, no code blocks):\n"
            "- \"rewritten_problem\": (string) The rewritten problem.\n"
            "- \"answer\": (string/number) The answer to the rewritten problem (the same as the ORIGINAL_ANSWER).\n\n"
            "[ORIGINAL_PROBLEM]\n"
            "{problem}"
            "[ORIGINAL_ANSWER]\n"
            "{answer}"
        )

        self.hybrid_prompt = (
            "You are an expert in designing mathematical problems. Your objective is to create a challenging problem variant "
            "by combining two techniques: Irrelevant Context Injection and Linguistic Rephrasing.\n\n"
            "[TASK]\n"
            "Given the ORIGINAL_PROBLEM and ORIGINAL_ANSWER below, produce a AUGMENTED_PROBLEM. You must perform two actions simultaneously:\n"
            "1. INJECT irrelevant context: Add additional context. The added context should appear topically related (e.g., alternative unused options, or descriptive details), but must be mathematically and logically irrelevant to the specific problem asked.\n"
            "2. REPHRASE the core mathematical statement: Change sentence structure, vocabulary, tone, or formatting while keeping the logic and values identical.\n"
            "**Output ONLY the rewritten problem text directly.**\n\n"
            "[CRITICAL CONSTRAINTS]\n"
            "1. PRESERVE VALUES: All numbers, variable names (e.g., x, y), functions, and geometric properties must remain strictly unchanged. "
            "The expected output format must remain UNTOUCHED and the correct answer must be exactly the same as ORIGINAL_ANSWER.\n"
            "2. NO SOLUTION: Do NOT solve the problem. Do NOT provide hints, reasoning, steps, or the final answer. "
            "Do NOT make assumptions about the solution method.\n"
            "3. NO CONFLICT: Ensure your added noise does not contradict the rephrased constraints or create ambiguity regarding the task goal.\n"
            "4. NATURAL INTEGRATION: The rephrased task and the irrelevant noise should blend together into a coherent (albeit noisy) text block.\n\n"
            "[EXAMPLE]\n"
            "Original_Problem: \n"
            "Calculate the area of a circle with a radius of 5 meters.\n"
            "Original_Answer: \n"
            "25\\pi\n"
            "JSON_Output:\n"
            "{{\n"
            "  \"rewritten_problem\": \"Yesterday, while walking through the park near the old library which was built in 1985, I noticed a landscaping project. The architect, who was wearing a blue hat, mentioned they are designing a circular flower bed. Despite the rainy weather, they need to determine the area of this circle, given that its radius is exactly 5 meters.\",\n"
            "  \"answer\": \"25\\pi\"\n"
            "}}\n\n"
            "[OUTPUT FORMAT]\n"
            "You must output ONLY a single JSON object (no markdown, no code blocks):\n"
            "- \"rewritten_problem\": (string) The rewritten problem.\n"
            "- \"answer\": (string/number) The answer to the rewritten problem (the same as the ORIGINAL_ANSWER).\n\n"
            "[ORIGINAL_PROBLEM]\n"
            "{problem}\n"
            "[ORIGINAL_ANSWER]\n"
            "{answer}"
        )

        self.decompose_prompt = (
            "You are an expert in the field of mathematics. Your ONLY goal is to extract and solve the absolute smallest, most atomic first step required to start solving the whole problem.\n\n"
            "[TASK]\n"
            "Given the `ORIGINAL_PROBLEM` and its `ORIGINAL_ANSWER` (for reference), perform the following:\n"
            "1. **Identify the Atomic First Step**: Look for the very first numerical value, expression, "
            "or basic arithmetic operation that is explicitly required to start the reasoning process.\n"
            "2. **Formulate the Sub-Problem**: Create a specific, closed-ended and verifiable Sub-Problem that targets ONLY this atomic first step.\n"
            "3. **Provide the Exact Answer**: Give the precise value or result for this Sub-Problem. The answer must be a constant value (number, simple expression, variable name, or short string).\n\n"
            "[CRITICAL CONSTRAINTS]\n"
            "1. **Absolute Simplicity**: The Sub-Problem must be dead simple. Do NOT ask problems requiring multi-step logic.\n"
            "2. **Strictly One Atomic Problem**: The Sub-Problem must contain EXACTLY ONE question mark and it is indeed a problem in itself. "
            "Never combine problems. It must be a single, standalone query and contains enough context to solve it.\n"
            "3. **Restricted Answer Format**: The answer MUST be one of the following specific types: "
            "a single Number (integer/float), a specific Variable name (e.g., 'a'), a simple Math Expression (e.g., 'a+b'), or a short String. NO sentences and NO explanations. Ensure that the answer is verifiable.\n"
            "4. **Existence**: The answer MUST be explicitly present in the provided context or directly derivable from it. "
            "Do not ask problems that require external knowledge or information not shown in the context.\n"
            "5. **No Ambiguity**: The problem must be self-contained. Do NOT use vague pronouns (e.g. 'it', 'this' or 'that'). Ensure the problem includes all necessary context to be solved in isolation.\n\n"
            "[EXAMPLES]\n"
            "--- Example 1 ---\n"
            "Problem: \"Solve for x: 2(x + 3) = 14.\"\n"
            "Bad Sub-Problem: \"What is x?\" (Reason: Lacks context.)\n"
            "Bad Sub-Problem: \"2(x + 3) = 14. What is x?\" (Reason: Too complex, requires full solution.)\n"
            "Good Sub-Problem: \"What is the value of 14 divided by 2?\"\n"
            "Sub-Problem Answer: \"7\"\n\n"
            "--- Example 2 ---\n"
            "Problem: \"A rectangular garden has a length that is 5 meters longer than its width. The perimeter of the garden is 50 meters. Find the area.\"\n"
            "Bad Sub-Problem: \"A rectangular garden has a length that is 5 meters longer than its width. The perimeter of the garden is 50 meters. What is the width of the garden?\" (Reason: Too complex.)\n"
            "Bad Sub-Problem: \"Let x be the width.\" (Reason: Not a problem.)\n"
            "Good Sub-Problem: \"A rectangular garden has a length that is 5 meters longer than its width. The perimeter of the garden is 50 meters. According to the context, if the width is represented by 'w', what is the algebraic expression for the length?\"\n"
            "Sub-Problem Answer: \"w + 5\"\n\n"
            "--- Example 3 ---\n"
            "Problem: \"A tank contains 12 liters of water. It is filled at a rate of 3 liters per minute. How many liters of water are in the tank after 5 minutes?\"\n"
            "Bad Sub-Problem: \"How many liters of water are in the tank after 5 minutes?\" (Reason: Too complex, requires multi-step reasoning.)\n"
            "Bad Sub-Problem: \"Let t be the time in minutes.\" (Reason: Not a problem.)\n"
            "Good Sub-Problem: \"If a tank is filled at a rate of 3 liters per minute, how many liters are added after 1 minute?\"\n"
            "Sub-Problem Answer: \"3\"\n\n"
            "[OUTPUT FORMAT]\n"
            "Return ONLY a raw JSON object. Do not include Markdown formatting like ```json ... ```.\n"
            "{{\n"
            "  \"first_sub_problem\": \"<The specific simple question>\",\n"
            "  \"sub_problem_answer\": \"<The concise result>\"\n"
            "}}\n\n"
            "[ORIGINAL_PROBLEM]\n"
            "{problem}\n"
            "[ORIGINAL_ANSWER]\n"
            "{answer}"
        )

    def _decode_sample_by_index(self, batch: DataProto, index: int):
        prompt = ""
        ground_truth = ""

        batch_data = getattr(batch, "batch", None)
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)

        if batch_data is None or non_tensor_batch is None:
            return prompt, ground_truth

        if "prompts" in batch_data:
            try:
                prompt = self.tokenizer.decode(batch_data["prompts"][index], skip_special_tokens=True)
            except Exception as e:
                raise RuntimeError(f"Decode prompt failed using 'prompts' at index {index}") from e
        
        elif "input_ids" in batch_data:
            try:
                input_ids = batch_data["input_ids"][index]
                response_len = 0
                if "responses" in batch_data:
                    response_len = int(batch_data["responses"][index].shape[-1])
                
                prompt_ids = input_ids[..., :-response_len] if response_len > 0 else input_ids
                prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            except Exception as e:
                raise RuntimeError(f"Decode prompt failed using 'input_ids' at index {index}") from e
        
        pattern = r"user\n(.*?)\nassistant"
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            prompt = match.group(1).strip()

        try:
            if "reward_model" in non_tensor_batch:
                reward_model_info = non_tensor_batch["reward_model"]
                if isinstance(reward_model_info, (list, np.ndarray)) and len(reward_model_info) > index:
                    ground_truth_info = reward_model_info[index]
                    if isinstance(ground_truth_info, dict) and "ground_truth" in ground_truth_info:
                        ground_truth = ground_truth_info["ground_truth"]
        except Exception as e:
            raise RuntimeError(f"Decode ground truth failed at index {index}") from e
        
        return prompt, ground_truth

    def _adjust_difficulty(self, batch: DataProto, wrong_problems: dict, wrong_gts: dict, wrong_idxs: dict, correct_problems: dict, correct_gts: dict, correct_idxs: dict):

        adjust_prompts = {}
        adjust_gts = {}

        # Adjust wrong problems by decomposition
        for uid, ori_problem in wrong_problems.items():
            ori_answer = wrong_gts[uid]
            input = self.decompose_prompt.format(problem=ori_problem, answer=ori_answer)
            completion = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "user", "content": input},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            output_str = completion.choices[0].message.content

            try:
                output_json = json.loads(output_str)
                if "first_sub_problem" in output_json and "sub_problem_answer" in output_json:
                    adjust_prompts[uid] = output_json["first_sub_problem"]
                    adjust_gts[uid] = output_json["sub_problem_answer"]
                else:
                    adjust_prompts[uid] = ori_problem
                    adjust_gts[uid] = ori_answer
            except json.JSONDecodeError:
                if uid not in adjust_prompts:
                    adjust_prompts[uid] = ori_problem
                    adjust_gts[uid] = ori_answer

        # Adjust correct problems by rephrasing and attacking
        for uid, ori_problem in correct_problems.items():
            ori_answer = correct_gts[uid]
            input = self.hybrid_prompt.format(problem=ori_problem, answer=ori_answer)
            completion = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "user", "content": input},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            output_str = completion.choices[0].message.content

            try:
                output_json = json.loads(output_str)
                if "rewritten_problem" in output_json:
                    adjust_prompts[uid] = output_json["rewritten_problem"]
                    adjust_gts[uid] = ori_answer
                else:
                    adjust_prompts[uid] = ori_problem
                    adjust_gts[uid] = ori_answer
            except json.JSONDecodeError:
                if uid not in adjust_prompts:
                    adjust_prompts[uid] = ori_problem
                    adjust_gts[uid] = ori_answer
        
        # Construct new batch with adjusted prompts
        data = []
        non_tensor_batch = batch.non_tensor_batch
        for idx, uid in enumerate(adjust_prompts):
            data_dict = {}

            messages = [
                {"role": "system", "content": self.prefix},
                {"role": "user", "content": adjust_prompts[uid]},
            ]

            if uid in wrong_idxs:
                src_idx = wrong_idxs[uid][0]
            elif uid in correct_idxs:
                src_idx = correct_idxs[uid][0]
            else:
                raise ValueError("Invalid index")
            
            data_dict["level"] = non_tensor_batch["level"][src_idx]
            data_dict["type"] = non_tensor_batch["type"][src_idx]
            data_dict["data_source"] = non_tensor_batch["data_source"][src_idx]
            data_dict["prompt"] = messages
            data_dict["ability"] = non_tensor_batch["ability"][src_idx]
            data_dict["reward_model"] = {
                "ground_truth": adjust_gts[uid], 
                "style": non_tensor_batch["reward_model"][src_idx]["style"]
            }
            data_dict["extra_info"] = {
                "index": non_tensor_batch["extra_info"][src_idx]["index"], 
                "split": non_tensor_batch["extra_info"][src_idx]["split"]
            }

            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )

            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )

            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

            data_dict["input_ids"] = input_ids[0]
            data_dict["attention_mask"] = attention_mask[0]
            data_dict["position_ids"] = position_ids[0]

            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.max_prompt_length:
                if self.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
                elif self.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                elif self.truncation == "middle":
                    left_half = self.max_prompt_length // 2
                    right_half = self.max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif self.truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
            data_dict["raw_prompt_ids"] = raw_prompt_ids

            if "extra_info" not in data_dict or data_dict["extra_info"] is None:
                data_dict["extra_info"] = dict()
            index = data_dict.get("extra_info", {}).get("index", 0)
            tools_kwargs = data_dict.get("extra_info", {}).get("tools_kwargs", {})
            interaction_kwargs = data_dict.get("extra_info", {}).get("interaction_kwargs", {})
            data_dict["index"] = index
            data_dict["tools_kwargs"] = tools_kwargs
            data_dict["interaction_kwargs"] = interaction_kwargs

            data.append(data_dict)

        # Create new batch
        batch_dict = collate_fn(data)

        data_batch = DataProto.from_single_dict(batch_dict)

        data_batch.non_tensor_batch["uid"] = np.array(
            [str(uid) for uid in adjust_prompts.keys()], dtype=object
        )

        gen_batch = self._get_gen_batch(data_batch)
        gen_batch.meta_info["global_steps"] = self.global_steps

        gen_batch = gen_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
        )

        dp_size = (
            self.actor_rollout_wg.world_size
            if not getattr(self, "async_rollout_mode", False)
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, dp_size)

        # Generate a batch of responses
        if not getattr(self, "async_rollout_mode", False):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_padded)

        gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=pad_size)

        adjust_batch = data_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        adjust_batch = adjust_batch.union(gen_batch_output)

        # Compute response mask
        if "response_mask" not in adjust_batch.batch.keys():
            adjust_batch.batch["response_mask"] = compute_response_mask(adjust_batch)
        
        # Compute reward model score
        reward_tensor, reward_extra_infos = compute_reward(adjust_batch, self.reward_fn)
        adjust_batch.batch["reward_tensor"] = reward_tensor[1] if isinstance(reward_tensor, tuple) else reward_tensor
        if reward_extra_infos:
            adjust_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos.items()})

        adjust_batch.meta_info.pop("timing", None)
        adjust_batch.meta_info["global_token_num"] = torch.sum(adjust_batch.batch["attention_mask"], dim=-1).tolist()

        # Replace the original batch data with adjusted data
        for idx, uid in enumerate(adjust_prompts):
            if uid in wrong_idxs:
                idxs = wrong_idxs[uid]
            elif uid in correct_idxs:
                idxs = correct_idxs[uid]
            else:
                raise ValueError("Invalid index")

            for offset, ori_idx in enumerate(idxs):
                for key in adjust_batch.batch.keys():
                    if key not in batch.batch:
                        continue
                    batch.batch[key][ori_idx] = adjust_batch.batch[key][idx * len(idxs) + offset]
                for key in adjust_batch.non_tensor_batch.keys():
                    if key not in batch.non_tensor_batch:
                        continue
                    batch.non_tensor_batch[key][ori_idx] = adjust_batch.non_tensor_batch[key][idx * len(idxs) + offset]
                if "meta_info" in adjust_batch.meta_info:
                    for key in adjust_batch.meta_info.keys():
                        if key not in batch.meta_info:
                            continue
                        batch.meta_info[key][ori_idx] = adjust_batch.meta_info[key][idx * len(idxs) + offset]

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        # breakpoint()
        self.global_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
            
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # Add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # Training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    # Repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # Compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if not self.use_rm:
                        with marked_timer("adjust", timing_raw):
                            batch.batch["reward_tensor"] = reward_tensor
                            scores = batch.batch["reward_tensor"].sum(dim=-1)
                            uid_list = batch.non_tensor_batch["uid"]

                            uid2score = defaultdict(list)
                            uid2acc = {}

                            bsz = scores.shape[0]
                            if len(uid_list) != bsz:
                                raise ValueError(f"uid_list length {len(uid_list)} != batch size {bsz}")

                            for i in range(bsz):
                                uid2score[uid_list[i]].append(scores[i])
                            for uid in uid2score:
                                if len(uid2score[uid]) < 1:
                                    raise ValueError(f"No score in prompt uid: {uid}")
                                scores_tensor = torch.stack(uid2score[uid])
                                uid2acc[uid] = torch.mean(scores_tensor)
                        
                            all_wrong_uids = []
                            all_correct_uids = []
                            all_valid_uids = []
                            total_count = len(uid2acc)
                            eps = 1e-6
                            for uid, acc in uid2acc.items():
                                if acc <= 0.0:
                                    all_wrong_uids.append(uid)
                                elif acc >= 1.0 - eps:
                                    all_correct_uids.append(uid)
                                else:
                                    all_valid_uids.append(uid)
                            all_wrong_count = len(all_wrong_uids)
                            all_correct_count = len(all_correct_uids)
                            all_valid_count = len(all_valid_uids)

                            metrics["batch_info/total_count"] = total_count
                            metrics["batch_info/all_wrong_count"] = all_wrong_count
                            metrics["batch_info/all_correct_count"] = all_correct_count
                            metrics["batch_info/all_valid_count"] = all_valid_count

                            uid2wrong_idxs = defaultdict(list)
                            uid2correct_idxs = defaultdict(list)
                            uid2wrong_problem, uid2wrong_gt = {}, {}
                            uid2correct_problem, uid2correct_gt = {}, {}
                            for idx, uid in enumerate(uid_list):
                                if uid in all_wrong_uids:
                                    uid2wrong_idxs[uid].append(idx)
                                    if uid not in uid2wrong_problem:
                                        uid2wrong_problem[uid], uid2wrong_gt[uid] = self._decode_sample_by_index(batch, idx)
                                if uid in all_correct_uids:
                                    uid2correct_idxs[uid].append(idx)
                                    if uid not in uid2correct_problem:
                                        uid2correct_problem[uid], uid2correct_gt[uid] = self._decode_sample_by_index(batch, idx)

                            if all_valid_count != total_count:
                                self._adjust_difficulty(batch, uid2wrong_problem, uid2wrong_gt, uid2wrong_idxs, uid2correct_problem, uid2correct_gt, uid2correct_idxs)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    # Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # Compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    
                    # Compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Compute rewards, apply kl penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)
                        
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                
                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                
                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)