function[] = SARSA(env)

% Algorithm Parameters
alg.alpha = 0.5;
alg.eps = 0.1;
alg.gamma = 1;  % No discounting

alg.numEpisodes = 100;

alg.Q_agent = zeros(env.rowDim, env.colDim, env.numActions);
alg.Q_adversary = zeros(env.rowDim, env.colDim, env.numActions);

alg.n = 1;

% Start the episode
totalRewards = [];
for epNum = 1:1:alg.numEpisodes
    epNum
    
    % Initialize and store S0 != terminal
    state = env.startState;
    
    states = [state];
    actions = [];
    rewards = [0];
    
    while 1
        agentActionNum = getAgentAction(env, alg, state);
        
        % Take action At
        [nextState, reward, done] = env.stepAgent(state, agentActionNum);
        
        % Observe and store the next reward as Rt+1 and the next state as
        % S_t+1
        states = [states; nextState];
        actions = [actions; agentActionNum];
        rewards = [rewards; reward];
        
        agentActionNumNext = getAgentAction(env,alg,nextState);
        agentActionValNext = alg.Q_agent(nextState(1), nextState(2), agentActionNumNext);
        
        alg.Q_agent(state(1), state(2), agentActionNum) = ...
            alg.Q_agent(state(1), state(2), agentActionNum) + ...
            alg.alpha * ...
            (reward + alg.gamma * agentActionValNext - ...
            alg.Q_agent(state(1), state(2), agentActionNum));
        
        state = nextState;
        
        % Make sure terminal state stays at 0
        alg.Q_agent(env.endState(1), env.endState(2), :) = 0;
        
        
        if done
           break 
        end
    end
    
    totalRewards = [totalRewards; sum(rewards)];
    
end

save("Results.mat")

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