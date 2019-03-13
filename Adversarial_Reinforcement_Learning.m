%% Reset Workspace
clc
clear
close all

%% Create environment

env = environment();

% Cliff environment
% env.rowDim = 4;
% env.colDim = 12;
% env.layout = zeros(env.rowDim, env.colDim);
% env.startState = [4, 1, 3];  % Up
% env.endState = [4, 12, 1];
% env.layout(4,2:end-1) = -1;

% Line environment
% env.rowDim = 1;
% env.colDim = 10;
% env.layout = zeros(env.rowDim, env.colDim);
% env.startState = [1, 1, 1];  % Up
% env.layout(1,end) = 2;

% Pit environment
env.rowDim = 9;
env.colDim = 9;
env.layout = zeros(env.rowDim, env.colDim);
env.layout(5,5) = -1;
env.startState = [5, 1, 1];  % Up
env.layout(5,end) = 2;  % End state

% Divide environment
env.rowDim = 9;
env.colDim = 9;
env.layout = zeros(env.rowDim, env.colDim);
env.layout(5,5) = -1;
env.startState = [5, 1, 1];  % Up
env.layout(5,end) = 2;  % End state



% Action space
env.agentActions = [0, 1;        % Right, 1
                    0, -1;       % Left,  2
                   -1, 0;        % Up,    3
                    1, 0];       % Down,  4
                
env.adversaryActions = [0, 1;        % Right, 1
                        0, 0;        % None,  2
                       -1, 0;        % Up,    3
                        1, 0;        % Down,  4
                        0, 0];       % None,  5  

%            
% Rewards
env.stepReward = -1;
env.cliffReward = -100;

numRuns = 1;
numEpisodes = 10000;

%% Q-Learning
qLearningResults = QLearning(env,1,0, numRuns, numEpisodes);

%% Evaluate
qLearningResults.eps = 0;
results = QLearningEval(env,qLearningResults,1);

%% SARSA

trainAgent = 1;
trainAdversary = 1;
useAdversary = 1;

% Initialize untrained actors
qAgent = zeros(env.rowDim, env.colDim, env.numAgentActions());
qAdv = zeros(env.rowDim, env.colDim, env.numAgentActions(), env.numAdversaryActions());

sarsaResults = SARSA(env, qAgent, qAdv, trainAgent, trainAdversary, useAdversary, numRuns, numEpisodes);
trainedAgent = sarsaResults.Q_agent;

% Train the adversary and agent
% trainAgent = 1;
% trainAdversary = 1;
% useAdversary = 1;
% numEpisodes = 10000;
% sarsaResults = SARSA(env, trainedAgent, qAdv, trainAgent, trainAdversary, useAdversary, numRuns, numEpisodes);

%% Evaluate
sarsaResults.eps = 0;
output = SARSAeval(env, sarsaResults, 1);
