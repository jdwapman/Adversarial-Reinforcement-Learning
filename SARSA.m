function[alg] = SARSA(env, trainAgent, trainAdversary)
runRewards = [];
for numRuns = 1:1:1
    % Algorithm Parameters
    alg.alpha = 0.5;
    alg.eps = 0.1;
    alg.gamma = 1;  % No discounting
    
    alg.numEpisodes = 100;
    
    alg.Q_agent = zeros(env.rowDim, env.colDim, env.numAgentActions());
    alg.Q_adversary = zeros(env.rowDim, env.colDim, env.numAgentActions(), env.numAdversaryActions());
    
    alg.n = 1;
    
    % Start the episode
    totalRewards = [];
    numRuns
    for epNum = 1:1:alg.numEpisodes
        
        
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
                
                if ~trainAdversary
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
            
            
            if trainAgent
                alg.Q_agent(state(1), state(2), agentActionNum) = ...
                    alg.Q_agent(state(1), state(2), agentActionNum) + ...
                    alg.alpha * ...
                    (-advReward + alg.gamma * agentActionValNext - ...
                    alg.Q_agent(state(1), state(2), agentActionNum));
            end
            
            if trainAdversary
                alg.Q_adversary(agentNextState(1), agentNextState(2), agentNextState(3), advActionNum) = ...
                    alg.Q_adversary(agentNextState(1), agentNextState(2), agentNextState(3), advActionNum) + ...
                    alg.alpha * ...
                    (advReward + alg.gamma * advActionValNext - ...
                    alg.Q_adversary(agentNextState(1), agentNextState(2), agentNextState(3), advActionNum));
            end
            
            state = advNextState;
            state(3) = agentActionNum;
            
            % Make sure terminal state stays at 0
            alg.Q_agent(env.endState(1), env.endState(2), :) = 0;
            alg.Q_adversary(env.endState(1), env.endState(2), :, :) = 0;
            
            if agentDone || advDone
                agentStates = [agentStates; state];  % Update with state it's moved to
                agentStates(end,3) = agentActionNum;
                advStates = [advStates; agentNextState];
                break
            end
        end
        
        %     if sum(agentRewards < -50) && (epNum > 250)
        %         asdf = 1
        %     end
        
        totalRewards = [totalRewards; sum(agentRewards)];
        
    end
    
    runRewards = [runRewards totalRewards];
end
plot(mean(runRewards,2))
ylim([-100,0])
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