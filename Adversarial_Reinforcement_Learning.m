%% Reset Workspace
clc
clear
close all

%% Create environment

env = environment();

% Cliff environment
env.rowDim = 4;
env.colDim = 12;
env.layout = zeros(env.rowDim, env.colDim);
env.startState = [4, 1, 3];  % Up
env.endState = [4, 12, 1];
env.layout(4,2:end-1) = -1;

% Line environment
% env.rowDim = 3;
% env.colDim = 20;
% env.layout = zeros(env.rowDim, env.colDim);
% env.startState = [2, 1, 1];  % Up
% env.endState = [2, 20, 1];

% Pit environment
% env.rowDim = 5;
% env.colDim = 5;
% env.layout = zeros(env.rowDim, env.colDim);
% env.layout(3,3) = -1;
% env.startState = [1, 1, 1];  % Up
% env.endState = [5, 5, 1];


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
numEpisodes = 1000;

%% Q-Learning
% qLearningResults = QLearning(env,1,1, numRuns, numEpisodes);
% % Evaluate
% qLearningResults.eps = 0;
% results = QLearningEval(env,qLearningResults,1);

%% SARSA

sarsaResults = SARSA(env, 1, 1, numRuns, numEpisodes);

% trainedAdv = SARSA(env, alg, 1,1, numRuns, numEpisodes);
%% Evaluate
sarsaResults.eps = 0;
output = SARSAeval(env, sarsaResults, 1);

load("Results.mat")