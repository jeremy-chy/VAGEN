from vagen.env.base.base_env import BaseEnv
from vagen.env.Embench_new.embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
import json
import re
from PIL import Image

manipulation_system_prompt = "## You are a Franka Panda robot with a parallel gripper. You can perform various tasks and output a sequence of gripper actions to accomplish a given task with images of your status. The input space, output action space and color space are defined as follows:\n\n** Input Space **\n- Each input object is represented as a 3D discrete position in the following format: [X, Y, Z]. \n- There is a red XYZ coordinate frame located in the top-left corner of the table. The X-Y plane is the table surface. \n- The allowed range of X, Y, Z is [0, 100]. \n- Objects are ordered by Y in ascending order.\n\n** Output Action Space **\n- Each output action is represented as a 7D discrete gripper action in the following format: [X, Y, Z, Roll, Pitch, Yaw, Gripper state].\n- X, Y, Z are the 3D discrete position of the gripper in the environment. It follows the same coordinate system as the input object coordinates.\n- The allowed range of X, Y, Z is [0, 100].\n- Roll, Pitch, Yaw are the 3D discrete orientation of the gripper in the environment, represented as discrete Euler Angles. \n- The allowed range of Roll, Pitch, Yaw is [0, 120] and each unit represents 3 degrees.\n- Gripper state is 0 for close and 1 for open.\n\n** Color space **\n- Each object can be described using one of the colors below:\n  [\"red\", \"maroon\", \"lime\", \"green\", \"blue\", \"navy\", \"yellow\", \"cyan\", \"magenta\", \"silver\", \"gray\", \"olive\", \"purple\", \"teal\", \"azure\", \"violet\", \"rose\", \"black\", \"white\"],\n\n** Generation Guide **\n- Include the thinking process between <|think_start|> and <|think_end|>\n- Include only the target action in <|action_start|> and <|action_end|>, i.e. the content inside <|action_start|> and <|action_end|> should be nothing more than the 7-DoF vector. Do not include any other thing, such as '\"'.\n"


def output_to_action(output_text):
    # Extract raw string between think_start and think_end
    think_match = re.search(r"<\|think_start\|\>(.*?)<\|think_end\|\>", output_text, re.DOTALL)
    if think_match:
        think_format_correct = True
        think_text = think_match.group(1).strip()
    else:
        think_format_correct = False
        think_text = "[No think block found]"

    # Extract the first action
    action_match = re.search(r"<\|action_start\|\>\[(.*?)\]<\|action_end\|\>", output_text)
    if action_match:
        action_format_correct = True
        action_list = [int(x.strip()) for x in action_match.group(1).split(',')]
    else:
        action_format_correct = False
        action_list = []

    return action_list, think_text, think_format_correct and action_format_correct

def seed_to_config(seed):
    if isinstance(seed, str):
        seed = int(seed)
    eval_sets = ['base', 'common_sense', 'complex', 'spatial', 'visual']
    return eval_sets[seed//100], seed%100 #TODO: check if this is correct

class EBManipulationEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super(EBManipulationEnv, self).__init__()
        self.env = EBManEnv(resolution=500, *args, **kwargs)
        self.system_prompt = ""
        # self.renew_system_prompt()
        self.all_steps_history = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        self.gamma = 0.9
        
    def get_system_prompt(self):
        return manipulation_system_prompt
        
    def step(self, llm_raw_response):

        action_list, think_text, format_correct = output_to_action(llm_raw_response)

        obs, reward, done, info = self.env.step(action_list)

        # TODO: calculate reward
        reward = 0
        if format_correct:
            reward += 1
        if info["task_success"]:
            reward += 20

        self.total_reward = self.total_reward * self.gamma + reward

        step_history_entry = {
            "step_id": info['env_step'],
            "thinking": thinking,
            "action": action_content,
            "env_feedback": info["env_feedback"]
        }
        self.all_steps_history.append(step_history_entry)
        interaction_history = self.all_steps_history.copy()
        user_prompt = (
                        "<image>\n " + "instruction: " + self.instruction + " \n " +
                        "interaction_history: " + json.dumps(interaction_history, indent=2) + " \n " +
                        "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    )

        image = Image.fromarray(obs['head_rgb'])
        metrics = {
            "turn_metrics": {
                "task_success": info["task_success"],
            },
            "traj_metrics": {
                "task_success": info["task_success"],
            }
        }
        obs = {
            "obs_str": user_prompt,
            "multi_modal_data": {
                "<image>": [image]
            }
        }
        info = {
            "metrics": metrics,
            "llm_raw_response": llm_raw_response
            # "llm_response": "" #Not using it currently, but we follow the original format to avoid errors
        }

        return obs, reward, done, info
        
    def close(self):
        self.env.close()
        
    def reset(self, seed): 

        eval_set, episode_id = seed_to_config(seed)
        image = Image.fromarray(self.env.reset(eval_set, episode_id)['head_rgb'])
        
        # self.renew_system_prompt()
        self.all_steps_history = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        user_prompt = (
                        "<image>\n " + "instruction: " + self.instruction + " \n " +
                        "interaction_history: []\n" +
                        "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    )
        obs = {
            "obs_str": user_prompt,
            "multi_modal_data": {
                "<image>": [image]
            }
        }
        metrics = {
            "turn_metrics": {
                "task_success": False,
            },
            "traj_metrics": {
                "task_success": False,
            }
        }

        info = {
            "metrics": metrics,
            "llm_raw_response": None,
            "llm_response": None
        }
        return obs, info
    
    # def system_prompt(self):
    #     return self.system_prompt
    
    def compute_reward(self):
        return self.total_reward
    
    # def renew_system_prompt(self):
    #     episode_num = self.env._current_episode_num
    #     eval_set = self.env._eval_set
    #     self.system_prompt = get_system_prompt(eval_set, episode_num)
