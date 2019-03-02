%% Reset Workspace
clc
clear
close all

%% Create environment

env = environment();

% State space. Notes: 0 for allowable state, +1 for obstacle, -1 for
% cliff/trap.
env.rowDim = 10;
env.colDim = 10;
env.layout = zeros(env.rowDim, env.colDim);

% Add a cliff at the bottom row
env.layout(end,2:end-1) = -1;
env.layout(1,2) = 1;  % Obstacle

env.startState = [1, 1];
env.endState = [10, 10];

% Action space
env.actions = [0, 1;        % Right, 1
               0, -1;       % Left,  2
              -1, 0;        % Up,    3
               1, 0];       % Down,  4
     
           
% Rewards
env.stepReward = -1;
env.cliffReward = -100;

