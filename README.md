# binary-options-simulation
# -*- coding: utf-8 -*-

## Overview
This project is set to showcase Reinforcement Learning capabilities on financial time series data.
The data used is retrieved from IQ options and is of symbol "EURUSD"
The goal is to train an agent to predict the direction of the price movement and make a decision to maximize the account balance in the shortest period of time. Here, we will be using a csv file with 980 datapoints after data preparation with columns : 'timestamp', 'open', 'close', 'volume', 'binary_representation', 'MACD', 'Signal_Line', 'RSI', 'Upper_Band', 'Lower_Band'.
The binary_representation column is the target variable which is 1 if the price closes above the open price.

