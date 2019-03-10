function[results] = QLearning(env, alg, useAdversary)

% Initialize and store S0 != terminal
state = env.startState;

agentStates = [];
agentActions = [];
agentRewards = [0];

advStates = [];
advActions = [];
advRewards = [0];

numSteps = 0;
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
            advActionNum = 5;  % No change in agent's pos
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
    
    % Agent is now in the advNextState location
    agentActionNumNext = getAgentAction(env,alg,advNextState);
    agentActionValNext = alg.Q_agent(advNextState(1), advNextState(2), agentActionNumNext);
    
    % Wind's next action depends on the agent's next state
    moveAgentState = env.stepAgent(advNextState, agentActionNumNext);
    advActionNumNext = getAdversaryAction(env,alg,moveAgentState);
    advActionValNext = alg.Q_adversary(moveAgentState(1), moveAgentState(2), moveAgentState(3), advActionNumNext);
       
    state = advNextState;
    state(3) = agentActionNum;
    
    numSteps = numSteps + 1;
    if numSteps == 1000
        break;
    end
    
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