from vagen.env.base.base_env import BaseEnv
from vagen.env.Embench_new.embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
# from vagen.env.Embench_new.prompt_utils_alfred import get_system_prompt
import json
import re
from PIL import Image

alfred_system_prompt = '''## You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules
• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
• Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle. 
• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.


## The available action id (0 ~ {}) and action names are: {}.

## Guidelines
1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output.\n Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions. 
4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.\n Try to modify the action sequence because previous actions do not lead to success.
5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan.\n If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.

** Generation Guide **\n    - Include the thinking process between <|think_start|> and <|think_end|>\n    - Include only the target action in <|action_start|> and <|action_end|>, i.e. the content inside <|action_start|> and <|action_end|> should be nothing more than [action_id, 'action_name'], where the action id is an integer and the action name is the corresponding name. Do not include any other thing, such as '\"'.\n    "
'''

# def parse(llm_raw_response):
#     pattern = r"<\|action_start\|\>\s*\[\s*(\d+)"
#     m = re.search(pattern, llm_raw_response)
#     if m:
#         print(f"debug: action is {m.group(1)}")
#         return int(m.group(1)), None, None
#     return None, None, None

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
    action_match = re.search(r"<\|action_start\|\>\[(\d+),\s*(.*?)\]<\|action_end\|\>", output_text, re.DOTALL)
    # action_match = re.search(r"<\|action_start\|\>(.*?)<\|action_end\|\>", output_text, re.DOTALL)
    if action_match:
        action_format_correct = True
        action_index = int(action_match.group(1))
        action_detail = action_match.group(2).strip()     
        action_content = f"[{action_index}, {action_detail}]"
    else:
        action_format_correct = False
        action_index = 1
        action_detail = ""
        action_content = "[No action block found]"

    return action_index, action_content, think_text, think_format_correct and action_format_correct

def seed_to_config(seed):
    if isinstance(seed, str):
        seed = int(seed)
    eval_sets = ['base', 'spatial', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon']
    return eval_sets[seed//100], seed%100 #TODO: check if this is correct

class AlfredEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super(AlfredEnv, self).__init__()
        self.env = EBAlfEnv(resolution=300, *args, **kwargs)
        self.system_prompt = ""
        # self.renew_system_prompt()
        self.all_steps_history = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        self.gamma = 1
        
    def get_system_prompt(self):
        return alfred_system_prompt.format(len(self.env.language_skill_set)-1, self.env.language_skill_set)
        
    def step(self, llm_raw_response):

        action_id, action_content, thinking, format_correct = output_to_action(llm_raw_response)

        obs, reward, done, info = self.env.step(action_id)

        # TODO: calculate reward
        reward = 0
        if format_correct:
            reward += 1
        if info["task_success"]:
            reward += 20

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
                "task_progress": info["task_progress"],
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

        print("--------------------------------")
        print("start reset Env")
        print("--------------------------------")   

        eval_set, episode_id = seed_to_config(seed)
        image = Image.fromarray(self.env.reset(eval_set, episode_id)['head_rgb'])
        
        print("--------------------------------")
        print(type(image))
        print("--------------------------------")

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
                "task_progress": 0,
                "task_success": False,
            },
            "traj_metrics": {}
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
    
    # def renew_system_prompt(self):
    #     episode_num = self.env._current_episode_num
    #     eval_set = self.env._eval_set
    #     self.system_prompt = get_system_prompt(eval_set, episode_num)
