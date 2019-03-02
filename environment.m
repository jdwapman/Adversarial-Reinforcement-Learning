classdef environment
    %environment Simple Gridworld Environment
    %   Simple Gridworld environment used for adversarial reinforcement
    %   learning. Features include cliffs and obstacles.
    
    properties
        layout
        rowDim
        colDim
        startState
        endState
        
        actions
        
        stepReward  % Agent's reward for each time step where it has not reached the edge
        cliffReward % Agent's reward for falling off a cliff
    end
    
    methods
        function obj = environment()
        end
        
        function[count] = numActions(obj)
            count = size(obj.actions,1);
        end
        
        function [nextState, reward] = stepAgent(obj, state, actionNum)
            %step Determine the agent's next state
            %   Detailed explanation goes here
            nextState = [state(1:2) + obj.actions(actionNum,:), actionNum];
            
            reward = obj.stepReward; % Default
            
            % Check if the agent has gone off a cliff
            if obj.inTrap(nextState)
                reward =  obj.cliffReward;
            end
        end
        
        function [nextState, reward] = stepAdversary(obj, state, actionNum)
            %step Determine the agent's next state and return the reward
            %   After the agent moves, the adversary can apply an action to
            %   modify the agent's state. The reward is evaluated after
            %   both actors have completed their action.
            nextState = [state(1:2) + obj.actions(actionNum,:), actionNum];
            
            reward = obj.stepReward; % Default
            
            % Check if the agent has gone off a cliff
            if obj.inTrap(nextState)
                reward =  obj.cliffReward;
            end
        end
        
        function [inTrap] = inTrap(obj, state)
            if obj.layout(state(1), state(2)) == -1
                inTrap = 1;
            else
                inTrap = 0;
            end
        end
        
        function[actions] = validActionsAgent(obj,state)
            nextStates = state(1:2) + obj.actions;
            
            actions = [];
            
            for i = 1:1:obj.numActions
                if ~obj.blocked(nextStates(i,:))
                    actions = [actions, i];
                end
            end
        end
        
        function[actions] = validActionsAdversary(obj,state)
            % First, find where it is possible for the agent to move
            % (within bounds and not into an obstacle)
            actions = obj.validActionsAgent(state)
            
            % Next, the wind cannot move the agent against its last
            % direction of motion
            
            % Find the opposite action
            action = state(3);
            opposites = [2,1,4,3];
            opposite = opposites(action);
            actions(actions == opposite) = [];
        end
        
        function[isBlocked] = blocked(obj, state)
            %outOfBounds Determine whether a given state is out of bounds
            %or blocked by an obstacle
            
            isBlocked = false;
            
            if state(1) < 1 || state(1) > obj.rowDim;
                isBlocked = true;
                return
            end
            
            if state(2) < 1 || state(2) > obj.colDim;
                isBlocked =true;
                return
            end
            
            % Need to return before checking within layout if out of bounds
            if obj.layout(state(1), state(2)) == 1
                isBlocked = true;
                return
            end
        end
    end
end

