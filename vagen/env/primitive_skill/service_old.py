from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from vagen.server.serial import serialize_observation

from .env import PrimitiveSkillEnv
from .env_config import PrimitiveSkillEnvConfig
from ..base.base_service_config import BaseServiceConfig
import threading
from mani_skill.utils.building.articulations.partnet_mobility import _load_partnet_mobility_dataset, PARTNET_MOBILITY
class PrimitiveSkillService(BaseService):
    """
    Service class for PrimitiveSkill environments.
    Implements batch operations with parallel processing for efficiency.
    """
    _dataset_lock = threading.Lock()
     
    def __init__(self, config: BaseServiceConfig):
        """
        Initialize the PrimitiveSkillService.
        
        Args:
            config: Service configuration
                max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_workers = config.get('max_workers', 10)
        self.environments = {}
        self.env_configs = {}
    
    def _ensure_dataset_initialized(self):
        """
        Ensure the PartNet Mobility dataset is initialized in a thread-safe manner.
        """
        with self._dataset_lock:
            if PARTNET_MOBILITY is None or "model_urdf_paths" not in PARTNET_MOBILITY:
                _load_partnet_mobility_dataset()
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple PrimitiveSkill environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
                Each config should contain:
                - env_name: Should be "primitive_skill"
                - env_config: PrimitiveSkill specific configuration
        """
        # Define worker function
        self._ensure_dataset_initialized()
        def create_single_env(env_id, config):
            # Verify environment type
            
            env_name = config.get('env_name', 'primitive_skill')
            if env_name != 'primitive_skill':
                return env_id, None, f"Expected environment type 'primitive_skill', got '{env_name}'"
            
            env_config_dict = config.get('env_config', {})
                
            # Create environment config
            env_config = PrimitiveSkillEnvConfig(**env_config_dict)
            
            # Create environment
            env = PrimitiveSkillEnv(env_config)
            
            return env_id, (env, env_config), None
        
        # Use ThreadPoolExecutor for parallel creation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all environment creation tasks
            futures = {
                executor.submit(create_single_env, env_id, config): env_id 
                for env_id, config in ids2configs.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error creating environment {env_id}: {error}")
                    continue
                
                env, env_config = result
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple PrimitiveSkill environments in parallel.
        
        Args:
            ids2seeds: A dictionary where each key is an environment ID and the corresponding
                     value is a seed value (or None for using default seeding behavior).
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}
        
        # Define worker function
        def reset_single_env(env_id, seed):
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            return env_id, (serialized_observation, info), None
        
        # Use ThreadPoolExecutor for parallel reset
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all reset tasks
            futures = {
                executor.submit(reset_single_env, env_id, seed): env_id 
                for env_id, seed in ids2seeds.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error resetting environment {env_id}: {error}")
                    results[env_id] = ({}, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple PrimitiveSkill environments in parallel.
        
        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                       value is the action to execute in that environment.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, reward, done, info)
        """
        results = {}
        
        # Define worker function
        def step_single_env(env_id, action):
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
            return env_id, (serialized_observation, reward, done, info), None
        
        # Use ThreadPoolExecutor for parallel step
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all step tasks
            futures = {
                executor.submit(step_single_env, env_id, action): env_id 
                for env_id, action in ids2actions.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error stepping environment {env_id}: {error}")
                    results[env_id] = ({}, 0.0, True, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """
        Compute the total reward for multiple PrimitiveSkill environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its computed total reward
        """
        results = {}
        
        # Define worker function
        def compute_reward_single_env(env_id):
            env = self.environments[env_id]
            return env_id, env.compute_reward(), None
        
        # Use ThreadPoolExecutor for parallel computation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all computation tasks
            futures = {
                executor.submit(compute_reward_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error computing reward for environment {env_id}: {error}")
                    results[env_id] = 0.0
                else:
                    results[env_id] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """
        Get system prompts for multiple PrimitiveSkill environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its corresponding system prompt string
        """
        results = {}
        
        # Define worker function
        def get_system_prompt_single_env(env_id):
            env = self.environments[env_id]
            return env_id, env.system_prompt(), None
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_system_prompt_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error getting system prompt for environment {env_id}: {error}")
                    results[env_id] = ""
                else:
                    results[env_id] = result
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple PrimitiveSkill environments and clean up resources in parallel.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all environments
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        # Define worker function
        def close_single_env(env_id):
            env = self.environments[env_id]
            env.close()
        
        # Use ThreadPoolExecutor for parallel closing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all closing tasks
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                error = future.result()
                if error:
                    print(f"Error closing environment: {error}")
        
        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)