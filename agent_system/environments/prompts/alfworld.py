# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
# ALFWORLD_TEMPLATE_NO_HIS = """
# You are an expert agent operating in the ALFRED Embodied Environment.
# Your current observation is: {current_observation}
# Your admissible actions of the current situation are: [{admissible_actions}].

# Now it's your turn to take an action.
# You should first reason step-by-step about the current situation.
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
# """

# ALFWORLD_TEMPLATE = """
# You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
# Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
# You are now at step {current_step} and your current observation is: {current_observation}
# Your admissible actions of the current situation are: [{admissible_actions}].

# Now it's your turn to take an action.
# You should first reason step-by-step about the current situation.
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
# """

# ---------------------------------------------------------------------------
# Function-typed thinking (tag capability menu).
# The same menu MUST be used verbatim across SFT, RL, and evaluation.
# ---------------------------------------------------------------------------

ALFWORLD_TAG_THINK_INSTRUCTION = """You should first conduct your reasoning process, enclosed within <think> </think> tags. Inside <think> </think>, you MUST organize your thinking using one or more of the following four thinking functions, each segment enclosed in its own tag:

<plan> Planning: decompose the task into sub-goals and decide what to do next. Use when: the task starts, or the previous step went well and you need to move forward.
<verify> Verification: check whether the latest observation matches what you expected your last action to achieve, and check your progress against the task requirements. Use when: every time you receive a new observation; you MUST verify before concluding the task is complete.
<reflect> Reflection: diagnose what went wrong — the action had no effect, the observation contradicts your expectation, or the object/location you need is missing; analyze the cause. Use when: any anomalous signal appears.
<backtrack> Backtracking: conclude that the current line of attack is a dead end, explicitly abandon it, and state which alternative route you will take instead (e.g., search a different location). Use when: after reflection you judge that continuing the current direction is useless.

Rules: every segment of thinking must belong to exactly one tag; you may use several tags in sequence (e.g., <reflect> followed by <backtrack>); free-form thinking outside these tags is NOT allowed."""

ALFWORLD_TAG_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
""" + ALFWORLD_TAG_THINK_INSTRUCTION + """
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TAG_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
""" + ALFWORLD_TAG_THINK_INSTRUCTION + """
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
