import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class MultiAgentBinaryOptionsTradingEnvironment(py_environment.PyEnvironment):
    def __init__(self, lookback_sequences, num_agents, initial_balance=10000, 
                 option_payout=0.8, option_loss=1, epsilon=0.1):
        """
        Initialize the environment for multiple agents trading binary options.

        Args:
            lookback_sequences (np.array): A 3D numpy array of shape 
                                            (num_sequences, lookback, num_features) 
                                            representing financial data prepared using the lookback period.
            num_agents (int): Number of agents interacting with the environment.
            initial_balance (float): The initial account balance for each agent.
            option_payout (float): The payout percentage for a correct option prediction.
            option_loss (float): The loss incurred for an incorrect prediction (usually 1, meaning full stake is lost).
            epsilon (float): Probability of taking a random action for exploration.
        """
        super(MultiAgentBinaryOptionsTradingEnvironment, self).__init__()

        # Initialize data and environment parameters
        self._lookback_sequences = lookback_sequences
        self._num_agents = num_agents
        self._current_step = 0
        self._total_steps = len(lookback_sequences) - 1
        self._balances = np.full(num_agents, initial_balance)  # Balance for each agent
        self._initial_balance = initial_balance
        self._option_payout = option_payout
        self._option_loss = option_loss
        self.epsilon = epsilon  # Epsilon for exploration
        
        # Action space: 0 (Put) or 1 (Call)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(num_agents,), dtype=np.int32, minimum=0, maximum=1, name='action') 
        
        # Observation space: sequence of data + balance for each agent
        num_features = lookback_sequences.shape[2]  # Number of features per sequence step
        self._observation_spec = array_spec.ArraySpec(
            shape=(num_agents, lookback_sequences.shape[1], num_features + 1), dtype=np.float32, name='observation')
        
        # Initial state for each agent
        self._states = self._get_observations(self._current_step)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_observations(self, step):
        """
        Helper function to create observations for all agents: lookback data + account balance.

        Args:
            step (int): The current step in the environment.
        
        Returns:
            np.array: The observations for all agents at the current step.
        """
        lookback_data = self._lookback_sequences[step]  # Extract lookback data
        balance_data = np.tile(self._balances[:, np.newaxis], (1, lookback_data.shape[1], 1))  # Balance added as a feature
        observation = np.concatenate((lookback_data, balance_data), axis=2)
        return observation

    def _reset(self):
        """
        Reset the environment to its initial state for all agents.
        """
        self._current_step = 0
        self._balances = np.full(self._num_agents, self._initial_balance)  # Reset balances
        self._episode_ended = False
        self._states = self._get_observations(self._current_step)
        return ts.restart(self._states)

    def _step(self, actions):
        """
        Advance the environment by one step based on each agent's actions.
        
        Args:
            actions (np.array): The actions taken by each agent.
        """
        if self._episode_ended:
            return self.reset()

        # Check if current step is within bounds
        if self._current_step >= self._total_steps or np.any(self._balances <= 0):
            self._episode_ended = True

        rewards = np.zeros(self._num_agents)  # Initialize rewards for each agent
        
        # Check the next price movement (binary representation) for reward
        if self._current_step < self._total_steps:
            next_movement = self._lookback_sequences[self._current_step + 1][0][0]  # Extract next movement
            
            # Calculate stake as a percentage of the current balance (dynamic stake)
            stakes = 0.25 * self._balances  # Stake for each agent
            
            for agent in range(self._num_agents):
                if self._balances[agent] <= 0:
                    continue  # Skip agents with no balance

                # Calculate immediate reward based on the action taken
                if actions[agent] == 1:  # Call (predict price will go up)
                    immediate_reward = self._option_payout * stakes[agent] if next_movement == 1 else -self._option_loss * stakes[agent]
                else:  # Put (predict price will go down)
                    immediate_reward = self._option_payout * stakes[agent] if next_movement == 0 else -self._option_loss * stakes[agent]

                # Update balance based on immediate reward
                self._balances[agent] += immediate_reward
                
                # Prevent negative balances
                if self._balances[agent] <= 0:
                    self._balances[agent] = self._initial_balance  # Reset balance to initial value
                    self._episode_ended = True

                # Calculate financial growth reward
                percentage_growth = (self._balances[agent] - self._initial_balance) / self._initial_balance
                financial_reward = 10 * percentage_growth 

                # Risk adjustment: penalize large losses
                risk_penalty = -0.1 * abs(immediate_reward) if immediate_reward < 0 else 0  # Penalty for losses

                # Normalize rewards
                total_reward = immediate_reward + financial_reward + risk_penalty
                rewards[agent] = total_reward / (abs(total_reward) + 1)  # Normalization

        # Update observations with new states for all agents
        self._current_step += 1
        self._states = self._get_observations(self._current_step)

        # Return appropriate time step for each agent
        if self._episode_ended:
            return ts.termination(self._states, rewards)
        else:
            return ts.transition(self._states, rewards, discount=0.99)

