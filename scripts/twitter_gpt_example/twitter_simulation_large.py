# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from colorama import Back
from yaml import safe_load

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType
# --- 新增下面这一行 ---
from oasis.social_platform.task_blackboard import TaskBlackboard 
# --- 新增结束 ---
# --- 新增下面这一行 ---
from oasis.social_platform.post_stats import SharedMemory, TweetStats
# --- 新增结束 ---

social_log = logging.getLogger(name="social")
social_log.setLevel("DEBUG")

file_handler = logging.FileHandler("social.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/twitter_dataset/anonymous_topic_200_1h",
)
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "False_Business_0.csv")


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    model_configs: dict[str, Any] | None = None,
    inference_configs: dict[str, Any] | None = None,
    actions: dict[str, Any] | None = None,
) -> None:
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    if os.path.exists(db_path):
        os.remove(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if recsys_type == "reddit":
        start_time = datetime.now()
    else:
        start_time = 0
    social_log.info(f"Start time: {start_time}")
    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    # --- 新增：初始化任务黑板 ---
    task_blackboard = TaskBlackboard()
    # --- 新增结束 ---
    # --- 新增：初始化共享内存和推文统计 ---
    tweet_stats = TweetStats()
    shared_memory = SharedMemory()
    # 必须将 tweet_stats 写入 shared_memory，否则后续读取会报错
    await shared_memory.write_memory("tweet_stats", tweet_stats)
    await shared_memory.write_memory("last_tweet_stats", TweetStats())
    # --- 新增结束 ---
    infra = Platform(
        db_path=db_path,
        channel=twitter_channel,
        sandbox_clock=clock,
        start_time=start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )
    inference_channel = Channel()
    twitter_task = asyncio.create_task(infra.running())
    # --- 修改开始：先给个默认值，或者直接从配置里读 ---
    is_openai_model = inference_configs.get("is_openai_model", False)

    # 保留原有逻辑作为一种补充（如果名字是 gpt 开头，强制为 True）
    if inference_configs["model_type"][:3] == "gpt":
        is_openai_model = True
    # --- 修改结束 ---

    try:
        all_topic_df = pd.read_csv("data/twitter_dataset/all_topics.csv")
        if "False" in csv_path or "True" in csv_path:
            if "-" not in csv_path:
                topic_name = csv_path.split("/")[-1].split(".")[0]
            else:
                topic_name = csv_path.split("/")[-1].split(".")[0].split(
                    "-")[0]
            source_post_time = (
                all_topic_df[all_topic_df["topic_name"] ==
                             topic_name]["start_time"].item().split(" ")[1])
            start_hour = int(source_post_time.split(":")[0]) + float(
                int(source_post_time.split(":")[1]) / 60)
    except Exception:
        print("No real-world data, let start_hour be 1PM")
        start_hour = 13

    model_configs = model_configs or {}

    # construct action space prompt if actions is not None
    action_prompt = None
    if actions:
        # action_prompt = "# OBJECTIVE\nYou're a Twitter user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions.You can also manage your social network by following interesting users or muting those who spread fake news.\n\n"
        action_prompt = "# OBJECTIVE\nYou're a highly active Twitter user. I will present you with some posts and comments in your feed. After reading them, choose appropriate actions to interact with the content (e.g., like, dislike, comment, quote) or manage your social network (e.g., follow users you agree with, mute users spreading fake news).\n\n"
        for action_name, action_info in actions.items():
            action_prompt += f"- {action_name}: {action_info['description']}\n"
            if action_info.get('arguments'):
                action_prompt += "    - Arguments:\n"
                for arg in action_info['arguments']:
                    action_prompt += f"        \"{arg['name']}\" ({arg['type']}) - {arg['description']}\n"

    # agent_graph = await generate_agents(
    #     agent_info_path=csv_path,
    #     twitter_channel=twitter_channel,
    #     inference_channel=inference_channel,
    #     start_time=start_time,
    #     recsys_type=recsys_type,
    #     action_space_prompt=action_prompt,
    #     twitter=infra,
    #     is_openai_model=is_openai_model,
    #     **model_configs,
    # )
    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        twitter_channel=twitter_channel,
        inference_channel=inference_channel,
        # --- 新增下面这一行 ---
        detection_inference_channel=None, 
        # --- 新增结束 ---
        start_time=start_time,
        recsys_type=recsys_type,
        action_space_prompt=action_prompt,
        twitter=infra,
        is_openai_model=is_openai_model,
        # --- 新增：传递参数 ---
        task_blackboard=task_blackboard,
        # --- 新增结束 ---
        # --- 新增：传递这两个参数 ---
        tweet_stats=tweet_stats,
        shared_memory=shared_memory,
        # --- 新增结束 ---
        **model_configs,
    )
    # agent_graph.visualize("initial_social_graph.png")

    for timestep in range(1, num_timesteps + 1):
        os.environ["SANDBOX_TIME"] = str(timestep * 3)
        social_log.info(f"timestep:{timestep}")
        db_file = db_path.split("/")[-1]
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        # if you want to disable recsys, please comment this line
        await infra.update_rec_table()

        # 0.05 * timestep here means 3 minutes / timestep
        simulation_time_hour = start_hour + 0.05 * timestep
        tasks = []
        for node_id, agent in agent_graph.get_agents():
            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()
                threshold = agent.user_info.profile["other_info"][
                    "active_threshold"][int(simulation_time_hour % 24)]
                if agent_ac_prob < threshold:
                    tasks.append(agent.perform_action_by_llm())
            else:
                await agent.perform_action_by_hci()

        await asyncio.gather(*tasks)
        # agent_graph.visualize(f"timestep_{timestep}_social_graph.png")
# --- 修改：补充 agent_id 参数 (这里设为 None) ---
    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT), None)
    # --- 修改结束 ---
    # await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await twitter_task
    


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        model_configs = cfg.get("model")
        inference_configs = cfg.get("inference")
        actions = cfg.get("actions")

        asyncio.run(
            running(**data_params,
                    **simulation_params,
                    model_configs=model_configs,
                    inference_configs=inference_configs,
                    actions=actions))
    else:
        asyncio.run(running())
    social_log.info("Simulation finished.")
