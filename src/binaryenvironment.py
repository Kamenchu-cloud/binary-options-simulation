import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class BinaryOptionsTradingEnvironment(py_environment.PyEnvironment):
    def __init__(self, lookback_sequences, initial_balance=10000, option_payout=0.8, option_loss=1, epsilon=0.1):
        """
        Initialize the environment for binary options trading.

        Args:
            lookback_sequences (np.array): A 3D numpy array of shape 
                                            (num_sequences, lookback, num_features) 
                                            representing financial data prepared using the lookback period.
            initial_balance (float): The initial account balance.
            option_payout (float): The payout percentage for a correct option prediction.
            option_loss (float): The loss incurred for an incorrect prediction (usually 1, meaning full stake is lost).
            epsilon (float): Probability of taking a random action for exploration.
        """
        super(BinaryOptionsTradingEnvironment, self).__init__()

        # Initialize data and environment parameters
        self._lookback_sequences = lookback_sequences
        self._current_step = 0
        self._total_steps = len(lookback_sequences) - 1  
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._option_payout = option_payout
        self._option_loss = option_loss
        self.epsilon = epsilon  # Epsilon for exploration
        
        # Action space: 0 (Put) or 1 (Call)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action') 
        
        # Observation space: sequence of data + balance
        num_features = lookback_sequences.shape[2]  # Number of features per sequence step
        self._observation_spec = array_spec.ArraySpec(
            shape=(lookback_sequences.shape[1], num_features + 1), dtype=np.float32, name='observation')
        
        # Initial state
        self._state = self._get_observation(self._current_step)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_observation(self, step):
        """
        Helper function to create the observation: lookback data + account balance.

        Args:
            step (int): The current step in the environment.
        
        Returns:
            np.array: The observation at the current step.
        """
        lookback_data = self._lookback_sequences[step]  # Extract lookback data
        balance = np.full((lookback_data.shape[0], 1), self._balance)  # Balance added as a feature
        observation = np.concatenate((lookback_data, balance), axis=1)
        return observation

    def _reset(self):
        """
        Reset the environment to its initial state.
        """
        self._current_step = 0
        self._balance = self._initial_balance
        self._episode_ended = False
        self._state = self._get_observation(self._current_step)
        return ts.restart(self._state)

    def _step(self, action):
        """
        Advance the environment by one step based on the agent's action.
        """
        if self._episode_ended:
            return self.reset()

        # Check if current step is within bounds
        if self._current_step >= self._total_steps or self._balance <= 0:
            self._episode_ended = True

        # Check the next price movement (binary representation) for reward
        if self._current_step < self._total_steps:
            next_movement = self._lookback_sequences[self._current_step + 1][0][0]  # Extract next movement
            
            # Calculate stake as a percentage of the current balance (dynamic stake)
            stake = 0.25 * self._balance
            
            # Calculate immediate reward based on the action taken
            if action == 1:  # Call (predict price will go up)
                immediate_reward = self._option_payout * stake if next_movement == 1 else -self._option_loss * stake
            else:  # Put (predict price will go down)
                immediate_reward = self._option_payout * stake if next_movement == 0 else -self._option_loss * stake

            # Update balance based on immediate reward
            self._balance += immediate_reward
            
            # Prevent negative balances
            if self._balance <= 0:
                self._balance = self._initial_balance  # Reset balance to initial value
                self._episode_ended = True

            # Calculate the reward based on financial implications
            # Reward for balance increase (using percentage growth)
            percentage_growth = (self._balance - self._initial_balance) / self._initial_balance
            financial_reward = 10 * percentage_growth  # Scale this reward as necessary

            # Risk adjustment: penalize large losses
            risk_penalty = -0.1 * abs(immediate_reward) if immediate_reward < 0 else 0  # Penalty for losses

            # Normalize rewards: ensuring total reward does not dominate immediate reward
            total_reward = immediate_reward + financial_reward + risk_penalty
            normalized_total_reward = total_reward / (abs(total_reward) + 1)  # Normalization

        else:
            normalized_total_reward = 0.0

        # Update observation with new state
        self._current_step += 1
        self._state = self._get_observation(self._current_step)

        # Return appropriate time step
        if self._episode_ended:
            return ts.termination(self._state, normalized_total_reward)
        else:
            return ts.transition(self._state, normalized_total_reward, discount=0.99)

    def select_action(self, model):
        """
        Select an action using epsilon-greedy strategy.

        Args:
            model: The trained model used to predict the action.
        
        Returns:
            int: The action to take (0 for Put, 1 for Call).
        """
        if np.random.rand() < self.epsilon:  # Epsilon-greedy exploration
            return np.random.choice([0, 1])  # Random action
        else:
            # Use model to predict the best action
            action_probabilities = model.predict(self._state) 
            return np.argmax(action_probabilities)  

# Example usage (the following lines should be used in a separate execution context)
lookback_sequences = np.random.randint(0, 2, (100, 10, 5))  # Sample binary sequences
env = BinaryOptionsTradingEnvironment(lookback_sequences)
time_step = env.reset()
action = 1  # Example action
next_time_step = env.step(action)
