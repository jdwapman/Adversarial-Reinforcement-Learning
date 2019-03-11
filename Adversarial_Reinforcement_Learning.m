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
env.rowDim = 6;
env.colDim = 9;
env.layout = zeros(env.rowDim, env.colDim);
env.layout(6,5) = -1;
env.startState = [1, 1, 1];  % Up
env.layout(:,end) = 2;  % End state

% Split environment
env.rowDim = 7;
env.colDim = 5;
env.layout = zeros(env.rowDim, env.colDim);

env.startState = [1, 1, 1];  % Up
env.layout(:,end) = 2;  % End state
env.layout(2,3) = 1;
env.layout(2:4,3:4) = 1;
% Action space
env.agentActions = [0, 1;        % Right, 1
                    0, -1;       % Left,  2
                   -1, 0;        % Up,    3
                    1, 0];       % Down,  4
                
env.adversaryActions = [0, 1;        % Right, 1
                        0, -1;       % Left,  2
                       -1, 0;        % Up,    3
                        1, 0;
                        0, 0];       % Down,  4
  
% env.adversaryActions = [0, 0;        % Right, 1
%                         0, 0;       % Left,  2
%                         0, 0;        % Up,    3
%                         0, 0;        % Down,  4
%                         0, 0];       

%            
% Rewards
env.stepReward = -1;
env.cliffReward = -100;

numRuns = 1;
numEpisodes = 4000;

%% Q-Learning
% qLearningResults = QLearning(env,1,0, numRuns, numEpisodes);

%% Evaluate
% qLearningResults.eps = 0;
% results = QLearningEval(env,qLearningResults,0);

%% SARSA

% trainAgent = 1;
% trainAdversary = 0;
% useAdversary = 0;
% 
% % Initialize untrained actors
% qAgent = zeros(env.rowDim, env.colDim, env.numAgentActions());
% qAdv = zeros(env.rowDim, env.colDim, env.numAgentActions(), env.numAdversaryActions());
% 
% % Train the agent
% numRuns = 1;
% numEpisodes = 4000;
% sarsaResults = SARSA(env, qAgent, qAdv, trainAgent, trainAdversary, useAdversary, numRuns, numEpisodes);
% trainedAgent = sarsaResults.Q_agent;
% 
% 
%% Evaluate
% sarsaResults.eps = 0;
% output = SARSAeval(env, sarsaResults, 1);

%% Iteration

% Initialize untrained actors
qAgent = zeros(env.rowDim, env.colDim, env.numAgentActions());
qAdv = zeros(env.rowDim, env.colDim, env.numAgentActions(), env.numAdversaryActions());

rewards = [];
for i = 1:1:1000
i
% First, alternate training
numEpisodes = 1;
numRuns = 1;

% Only train the agent
trainAgent = 1;
trainAdversary = 0;
useAdversary = 1;
sarsaResultsAgent = SARSA(env, qAgent, qAdv, trainAgent, trainAdversary, useAdversary, numRuns, numEpisodes, 0);
qAgent = sarsaResultsAgent.Q_agent;

rewards = [rewards; sarsaResultsAgent.runRewards];

% Only train the adversary
trainAgent = 0;
trainAdversary = 1;
useAdversary = 1;
sarsaResultsAdv = SARSA(env, qAgent, qAdv, trainAgent, trainAdversary, useAdversary, numRuns, numEpisodes, 0);
qAdv = sarsaResultsAdv.Q_adversary;
end

%% Evaluate
sarsaResults.eps = 0;
output = SARSAeval(env, sarsaResultsAgent, 1);
