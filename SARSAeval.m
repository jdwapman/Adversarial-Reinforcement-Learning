function [results] = SARSAeval(env, alg, useAdversary)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here


% Initialize and store S0 != terminal
state = env.startState;

agentStates = [];
agentActions = [];
agentRewards = [0];

advStates = [];
advActions = [];
advRewards = [0];

while 1
    
    
    % Take action At
    agentActionNum = getAgentAction(env, alg, state);
    [agentNextState, agentReward, agentDone] = env.stepAgent(state, agentActionNum);
    
    % Adversary's turn to modify the position
    if agentDone % Do nothing
        advNextState = agentNextState;
        advReward = -agentReward;
        advDone = agentDone;
    else
        advActionNum = getAdversaryAction(env, alg, agentNextState);
        
        if ~useAdversary
            advActionNum = 5;
        end
        
        [advNextState, advReward, advDone] = env.stepAdversary(agentNextState, advActionNum);
    end
    
    if agentReward == -100
        advReward = 100;
    end
    
    % Observe and store the next reward as Rt+1 and the next state as
    % S_t+1
    agentStates = [agentStates; state];  % Update with state it's moved to
    agentStates(end,3) = agentActionNum;
    advStates = [advStates; agentNextState];
    
    agentActions = [agentActions; agentActionNum];
    agentRewards = [agentRewards; -advReward];
    
    
    advActions = [advActions; advActionNum];
    advRewards = [advRewards; advReward];
    
    state = advNextState;
    state(3) = agentActionNum;
    
    if agentDone || advDone
        agentStates = [agentStates; state];  % Update with state it's moved to
        agentStates(end,3) = agentActionNum;
        advStates = [advStates; agentNextState];
        break
    end
end
results.agentStates = agentStates;
results.agentRewards = agentRewards;
results.agentActions = agentActions;
results.advStates = advStates;
results.advRewards = advRewards;
results.advActions = advActions;

end

function[chosenAction] = getAgentAction(env, alg, state)
% Take the greedy action with probability 1-epsilon
greedy = binornd(1, 1-alg.eps);

actions = env.validActionsAgent(state);

% Choose A from S using policy derived from Q (eps-greedy)
currQ = alg.Q_agent(state(1), state(2), actions);

% Take the greedy action with probability 1-epsilon
greedy = binornd(1, 1-alg.eps);

if greedy
    maxQ = max(currQ); % Find the maximum value
    maxIdx = find(currQ == maxQ);
    validActionNum = maxIdx(randi([1,length(maxIdx)]));
else
    validActionNum = randi([1,length(actions)]);
end

chosenAction = actions(validActionNum);

end

function[chosenAction] = getAdversaryAction(env, alg, state)
% Take the greedy action with probability 1-epsilon
greedy = binornd(1, 1-alg.eps);

actions = env.validActionsAdversary(state);

% Choose A from S using policy derived from Q (eps-greedy)
currQ = alg.Q_adversary(state(1), state(2), state(3), actions);

% Take the greedy action with probability 1-epsilon
greedy = binornd(1, 1-alg.eps);

if greedy
    maxQ = max(currQ); % Find the maximum value
    maxIdx = find(currQ == maxQ);
    validActionNum = maxIdx(randi([1,length(maxIdx)]));
else
    validActionNum = randi([1,length(actions)]);
end

chosenAction = actions(validActionNum);

end

