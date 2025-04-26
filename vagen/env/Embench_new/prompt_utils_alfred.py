import json
import argparse
import os

#load the avail actions from action_spaces/action_spaces_{eval_set}.json for all eval sets into a big dict
avail_actions = {}
eval_sets = ['base', 'spatial', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon']
for eval_subset in eval_sets:
    with open(f"/srv/local/xsm/alfred_all/action_spaces/action_spaces_{eval_subset}.json", "r") as f:
        action_space = json.load(f)
        avail_actions[eval_subset] = action_space

def get_system_prompt(eval_subset, episode_id):
    # print("avail actions: ", avail_actions[eval_subset])
    # print("eval subset: ", eval_subset)
    # print("episode id: ", episode_id)
    # print(avail_actions[eval_subset][str(episode_id+1)])
    action_space = avail_actions[eval_subset][str(episode_id+1)]
    num_avail_actions = len(action_space)
     
    system_prompt = f"""## You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.
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


    ## The available action id (0 ~ {num_avail_actions-1}) and action names are: {str(action_space)}.

    ## Guidelines
    1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
    2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
    3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output.\n Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions. 
    4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.\n Try to modify the action sequence because previous actions do not lead to success.
    5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
    6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan.\n If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.

    ** Generation Guide **
    - Include the thinking process between <|think_start|> and <|think_end|>
    - Include only the target action in <|action_start|> and <|action_end|>, i.e. the content inside <|action_start|> and <|action_end|> should be nothing more than the 7-DoF vector. Do not include any other thing, such as '"'.
    """
    
    return system_prompt
