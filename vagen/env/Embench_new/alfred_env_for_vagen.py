from vagen.env.base.base_env import BaseEnv
from vagen.env.Embench_new.embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
from vagen.env.Embench_new.prompt_utils_alfred import get_system_prompt
import json
import re

def parse(llm_raw_response):
    pattern = r"<\|action_start\|\>\s*\[\s*(\d+)"
    m = re.search(pattern, llm_raw_response)
    if m:
        print(f"debug: action is {m.group(1)}")
        return int(m.group(1)), None, None
    return None, None, None

def seed_to_config(seed):
    if isinstance(seed, str):
        seed = int(seed)
    eval_sets = ['base', 'spatial', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon']
    return eval_sets[seed//100], seed%100 #TODO: check if this is correct

class AlfredEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super(AlfredEnv, self).__init__()
        self.env = EBAlfEnv(*args, **kwargs)
        self.system_prompt = ""
        self.renew_system_prompt()
        self.all_steps_history = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        self.gamma = 1
        
    def get_system_prompt(self):
        return self.system_prompt
        
    def step(self, llm_raw_response):
        action_id, action_description, thinking = parse(llm_raw_response)
        obs, reward, done, info = self.env.step(action_id)
        self.total_reward += reward * self.gamma ** (info['env_step']-1) #do we need discount here
        step_history_entry = {
            "step_id": info['env_step'],
            "thinking": thinking,
            "action": [action_id, action_description],
            "env_feedback": info["env_feedback"]
        }
        interaction_history = self.all_steps_history.copy()
        self.all_steps_history.append(step_history_entry)
        user_prompt = (
                        "<image>\n " + "instruction: " + self.instruction + " \n " +
                        "interaction_history: " + json.dumps(interaction_history, indent=2) + " \n " +
                        "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    )
        image = obs['head_rgb']
        metrics = {
            "episode_elapsed_seconds": info['episode_elapsed_seconds'],
            "last_action_success": info['last_action_success']
        }
        new_obs = {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "image": image
        }
        new_info = {
            "metrics": metrics,
            "llm_raw_response": llm_raw_response,
            "llm_response": "" #Not using it currently, but we follow the original format to avoid errors
        }
        return new_obs, reward, done, new_info
        
    def close(self):
        self.env.close()
        
    def reset(self, seed):
        eval_set, episode_id = seed_to_config(seed)
        image = self.env.reset(eval_set, episode_id)
        self.renew_system_prompt()
        self.all_steps_history = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        user_prompt = (
                        "<image>\n " + "instruction: " + self.instruction + " \n " +
                        "interaction_history: []\n" +
                        "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    )
        obs = {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "image": image
        }
        metrics = {
            "episode_elapsed_seconds": 0,
            "last_action_success": None
        }
        info = {
            "metrics": metrics,
            "llm_raw_response": None,
            "llm_response": None
        }
        return obs, info
    
    def system_prompt(self):
        return self.system_prompt
    
    def compute_reward(self):
        return self.total_reward
    
    def renew_system_prompt(self):
        episode_num = self.env._current_episode_num
        eval_set = self.env._eval_set
        self.system_prompt = get_system_prompt(eval_set, episode_num)
