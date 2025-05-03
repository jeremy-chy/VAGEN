from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from vagen.env.Embench_new.alfred_env_for_vagen import AlfredEnv

from vagen.server.serial import serialize_observation


class AlfredService(BaseService):    
    def __init__(self, serviceconfig: None):
        self.envs = {}
        self.max_workers = 1  # Default to 4 workers if not specified
        
    def create_environment(self, env_id: str, config: Dict[str, Any]) -> None:
        """
        Helper function to create a single environment.

        Args:
            env_id (str): The environment ID.
            config (Dict[str, Any]): The configuration for the environment.
        """
        # self.envs[env_id] = AlfredEnv(**config["env_config"])
        self.envs[env_id] = AlfredEnv()
        # try:
        # self.envs[env_id] = AlfredEnv(**config)  # Create environment from config
        # except Exception as e:
        #     print(f"Error creating environment {env_id}: {e}")
        #     # Handle any error gracefully (e.g., log it, attempt retry, etc.)
    
    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        """
        Create multiple environments in parallel.

        Args:
            ids2configs (Dict[Any, Any]): 
                A dictionary where each key is an environment ID and the corresponding
                value is the configuration for that environment.

        Returns:
            None

        Note:
            The implementation should create all environments concurrently.
            It should gracefully handle errors and perform cleanup of any partially created environments.
        """
        # Clear environments that are not in the new config
        for env_id in list(self.envs.keys()):
            if env_id not in ids2configs:
                self.envs[env_id].close()
                del self.envs[env_id]

        # Use ThreadPoolExecutor to create environments concurrently
        with ThreadPoolExecutor() as executor:
            futures = []
            for env_id, config in ids2configs.items():
                futures.append(executor.submit(self.create_environment, env_id, config))

            # Handle errors and cleanup
            for future in as_completed(futures):
                future.result()
                '''
                try:
                    future.result()  # Get result to catch exceptions
                except Exception as e:
                    print(f"Error in one of the environment creations: {e}")
                    # Optionally, handle cleanup here
                    pass
                '''
        

    def reset_environment(self, env_id: str, seed: Any) -> Tuple[Any, Any]:
        """
        Helper function to reset a single environment.

        Args:
            env_id (str): The environment ID.
            seed (Any): The seed for resetting, or None if default behavior is required.

        Returns:
            Tuple[Any, Any]: A tuple (observation, info) after the reset.
        """
        if seed is None:
            raise NotImplementedError("None seed is not supported in AlfredEnvService")
        
        if env_id not in self.envs:
            raise ValueError(f"Environment {env_id} not found.")
        
        try:
            # Reset the environment and return the result
            print("--------------------------------")
            print("start reset")
            print("--------------------------------")
            observation, info = self.envs[env_id].reset(seed)
            observation = serialize_observation(observation)
            return env_id, (observation, info)
        except Exception as e:
            print(f"Error resetting environment {env_id}: {e}")
            return env_id, None  # Handle the error gracefully here, e.g., return None or a default value
    
    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple environments in parallel.

        Args:
            ids2seeds (Dict[Any, Any]):
                A dictionary where each key is an environment ID and the corresponding
                value is a seed value (or None for using default seeding behavior).

        Returns:
            Dict[Any, Tuple[Any, Any]]:
                A dictionary mapping environment IDs to tuples of the form (observation, info),
                where 'observation' is the initial state after reset, and 'info' contains additional details.
        """
        print("--------------------------------")
        print("start reset batch")
        print("--------------------------------")
        return_dict = {}
        
        # Use ThreadPoolExecutor to reset environments concurrently
        with ThreadPoolExecutor() as executor:
            futures = []
            for env_id, seed in ids2seeds.items():
                futures.append(executor.submit(self.reset_environment, env_id, seed))
            
            for future in as_completed(futures):
                try:
                    env_id, result = future.result()  # Get result, raises exception if occurred
                    if result is not None:
                        return_dict[env_id] = result
                except Exception as e:
                    print(f"Error processing future for reset: {e}")
        
        return return_dict

    def step_environment(self, env_id: str, action: Any) -> Tuple[Dict, float, bool, Dict]:
        """
        Helper function to step through a single environment.

        Args:
            env_id (str): The environment ID.
            action (Any): The action to take in the environment.

        Returns:
            Tuple[Dict, float, bool, Dict]: A tuple (observation, reward, done, info) after taking the step.
        """
        print("--------------------------------")
        print(f"action: {action}")
        print("--------------------------------")
        if env_id not in self.envs:
            raise ValueError(f"Environment {env_id} not found.")

        observation, reward, done, info = self.envs[env_id].step(action)
        observation = serialize_observation(observation)
        return env_id, (observation, reward, done, info)
        # try:
        #     # Step through the environment and return the result
        #     observation, reward, done, info = self.envs[env_id].step(action)
        #     observation = serialize_observation(observation)
        #     return env_id, (observation, reward, done, info)
        # except Exception as e:
        #     print(f"Error stepping environment {env_id}: {e}")
        #     return env_id, None  # Return default values in case of error
    
    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Step through multiple environments in parallel.

        Args:
            ids2actions (Dict[Any, Any]):
                A dictionary where each key is an environment ID and the corresponding
                value is the action to execute in that environment.

        Returns:
            Dict[Any, Tuple[Dict, float, bool, Dict]]:
                A dictionary mapping environment IDs to tuples of the form 
                (observation, reward, done, info), where:
                    - 'observation' is the new state of the environment after the action,
                    - 'reward' is a float representing the reward received,
                    - 'done' is a boolean indicating whether the environment is finished,
                    - 'info' contains additional information or context.
        """
        return_dict = {}
        
        # Use ThreadPoolExecutor to step through environments concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for env_id, action in ids2actions.items():
                futures.append(executor.submit(self.step_environment, env_id, action))
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    env_id, result = future.result()  # Get result, raises exception if occurred
                    if result[0] is not None:  # Ensure observation is valid
                        return_dict[env_id] = result
                except Exception as e:
                    print(f"Error processing future for step: {e}")
        
        return return_dict

    def compute_reward(self, env_id: str) -> float:
        """
        Helper function to compute the reward for a single environment.

        Args:
            env_id (str): The environment ID.

        Returns:
            float: The computed total reward for the environment.
        """
        if env_id not in self.envs:
            raise ValueError(f"Environment {env_id} not found.")
        
        try:
            reward = self.envs[env_id].compute_reward()  # Assuming compute_reward is a method in AlfredEnv
            return reward
        except Exception as e:
            print(f"Error computing reward for environment {env_id}: {e}")
            return 0.0  # Return 0 in case of error
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute the total reward for multiple environments in parallel.

        Args:
            env_ids (List[str]): A list of environment IDs.

        Returns:
            Dict[Any, float]:
                A dictionary mapping each environment ID to its computed total reward.
        """
        return_dict = {}
        
        # Use ThreadPoolExecutor to compute rewards concurrently
        with ThreadPoolExecutor() as executor:
            futures = []
            for env_id in env_ids:
                futures.append(executor.submit(self.compute_reward, env_id))
            
            for future in as_completed(futures):
                try:
                    result = future.result()  # Get result, raises exception if occurred
                    if result is not None:
                        return_dict[env_id] = result
                except Exception as e:
                    print(f"Error processing future for reward computation: {e}")
        
        return return_dict

    def get_system_prompt(self, env_id: str) -> str:
        """
        Helper function to retrieve the system prompt for a single environment.

        Args:
            env_id (str): The environment ID.

        Returns:
            str: The system prompt string for the environment.
        """
        if env_id not in self.envs:
            raise ValueError(f"Environment {env_id} not found.")
        
        try:
            system_prompt = self.envs[env_id].get_system_prompt()  # Assuming get_system_prompt is a method in AlfredEnv
            return env_id, system_prompt
        except Exception as e:
            print(f"Error retrieving system prompt for environment {env_id}: {e}")
            return env_id, ""  # Return empty string in case of error
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Retrieve system prompts for multiple environments in parallel.

        Args:
            env_ids (List[str]): A list of environment IDs.

        Returns:
            Dict[Any, str]:
                A dictionary mapping each environment ID to its corresponding system prompt string.
        """
        return_dict = {}
        
        # Use ThreadPoolExecutor to retrieve system prompts concurrently
        with ThreadPoolExecutor() as executor:
            futures = []
            for env_id in env_ids:
                futures.append(executor.submit(self.get_system_prompt, env_id))
            
            for future in as_completed(futures):
                try:
                    env_id, system_prompt = future.result()  # Get result, raises exception if occurred
                    if system_prompt is not None:
                        return_dict[env_id] = system_prompt
                except Exception as e:
                    print(f"Error processing future for system prompt retrieval: {e}")
        
        return return_dict

    def close_environment(self, env_id: str) -> None:
        """
        Helper function to close a single environment.

        Args:
            env_id (str): The environment ID.
        """
        if env_id not in self.envs:
            print(f"Environment {env_id} not found, skipping close.")
            return
        
        try:
            self.envs[env_id].close()  # Assuming close is a method in AlfredEnv
        except Exception as e:
            print(f"Error closing environment {env_id}: {e}")
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and clean up resources in parallel.

        Args:
            env_ids (Optional[List[str]]):
                A list of environment IDs to close. If None, all environments should be closed.

        Returns:
            None
        """
        if env_ids is None:
            env_ids = list(self.envs.keys())  # Close all environments if no list provided
        
        # Use ThreadPoolExecutor to close environments concurrently
        with ThreadPoolExecutor() as executor:
            futures = []
            for env_id in env_ids:
                futures.append(executor.submit(self.close_environment, env_id))
            
            for future in as_completed(futures):
                try:
                    future.result()  # Get result, raises exception if occurred
                except Exception as e:
                    print(f"Error processing future for environment close: {e}")