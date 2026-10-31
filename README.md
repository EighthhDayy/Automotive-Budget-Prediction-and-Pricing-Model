# Automotive-Budget-Prediction-and-Pricing-Model


1.Project Overview   

The Automotive Budget Prediction and Pricing Model combines machine learning and reinforcement learning methods to predict consumers’ car budgets and generate corresponding pricing and sales strategies.

The model can help potential car buyers estimate their budget and provide car sales personnel with pricing and sales strategy suggestions, aiming to increase conversion rates while maximizing profits.

2.Background

This project was inspired by my previous work experience at a technology company under a car manufacturer. During that time, I participated in the development of an automotive marketing model designed to help car manufacturers:

Develop precise marketing strategies; Increase brand awareness and market share; Improve sales conversion rates and overall sales performance

The commercial model was mainly based on market research and analysis data, used to identify consumer demand patterns and market trends to support sales decisions.
Inspired by this model, I created this open-source project using publicly available datasets.

3.Project Goals

The project aims to achieve the following goals:

Predict consumers’ car budgets

Generate reasonable pricing and sales strategies based on budgets and user features

Optimize pricing decisions through reinforcement learning, balancing profits and conversion rates

4.Use Cases

For Car Buyers: 

When users are interested in buying a car but are unsure of their budget, the model can estimate a recommended budget based on user data such as Annual Salary, Net Worth, and Age.
    
For Car Sales Personnel:
  
It can predict potential customers’ budgets, reference recommended pricing and sales strategies and help improve the likelihood of closing a sale

5.Methodology

This project uses a combination of machine learning and reinforcement learning methods:

5.1 Customer Clustering 

Use K-Means to cluster consumersand explore purchase preferences and behavior characteristics of different user groups
  
5.2 Budget Prediction

Use XGBoost to predict the budget range based on user features and historical data
  
5.3 Car Recommendation (KNN)

Based on the predicted budget, use KNN to find the three cars with prices closest to the budget
  
5.4 Pricing Strategy Optimization (Reinforcement Learning)

Build a Sales Simulation Environment
  
Train a Q-Learning agent
  
Learn optimal pricing strategies over 1000+ iterations
  
Balance profit maximization and conversion rate
  
6.Data Source

  From publicly available datasets: 
  
  url = 'https://raw.githubusercontent.com/pritom02bh/Car-Purchase-Amount-Prediction-Project-for-Car-Industry/master/Car_Purchasing_Data.csv'


7.Model Outputs

The model can produce the following outputs:

Predicted consumer car budgets
  
Consumer clustering labels
  
Pricing and sales strategy suggestions for different user groups
  
Performance of different pricing strategies in the simulation environment in terms of profit and conversion rate

8.Model Performance Evaluation

The budget prediction model (XGBoost) was evaluated using $R^2$ Score and Mean Absolute Error (MAE):

<img width="343" height="55" alt="image" src="https://github.com/user-attachments/assets/8bbb223d-d673-4fae-9765-f6a3e4eb1678" />

9.Limitations

Public datasets may have limitations in accuracy and completeness and reinforcement learning is based on a simulated market environment.
  
10.Future Work

So, future work can consider the following:

Use larger-scale, real-time market data
  
Introduce more complex consumer decision behavior modeling
  
Explore deep reinforcement learning methods
  
Extend the model to real-world car models and brand pricing scenarios
