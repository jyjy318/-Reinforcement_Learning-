%% Create Environment Interface
MDP = createMDP(8,["up";"down"]);
MDP.T(1,2,1) = 1;
MDP.R(1,2,1) = 3;
MDP.T(1,3,2) = 1;
MDP.R(1,3,2) = 1;
% State 2 transition and reward
MDP.T(2,4,1) = 1;
MDP.R(2,4,1) = 2;
MDP.T(2,5,2) = 1;
MDP.R(2,5,2) = 1;
% State 3 transition and reward
MDP.T(3,5,1) = 1;
MDP.R(3,5,1) = 2;
MDP.T(3,6,2) = 1;
MDP.R(3,6,2) = 4;
% State 4 transition and reward
MDP.T(4,7,1) = 1;
MDP.R(4,7,1) = 3;
MDP.T(4,8,2) = 1;
MDP.R(4,8,2) = 2;
% State 5 transition and reward
MDP.T(5,7,1) = 1;
MDP.R(5,7,1) = 1;
MDP.T(5,8,2) = 1;
MDP.R(5,8,2) = 9;
% State 6 transition and reward
MDP.T(6,7,1) = 1;
MDP.R(6,7,1) = 5;
MDP.T(6,8,2) = 1;
MDP.R(6,8,2) = 1;
% State 7 transition and reward
MDP.T(7,7,1) = 1;
MDP.R(7,7,1) = 0;
MDP.T(7,7,2) = 1;
MDP.R(7,7,2) = 0;
% State 8 transition and reward
MDP.T(8,8,1) = 1;
MDP.R(8,8,1) = 0;
MDP.T(8,8,2) = 1;
MDP.R(8,8,2) = 0;
MDP.TerminalStates = ["s7";"s8"];
env = rlMDPEnv(MDP);
env.ResetFcn = @() 1;

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0)

%% Create AC Agent
qTable = rlTable(obsInfo, actInfo);
criticNetwork = [
    imageInputLayer([1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(1,'Name','CriticFC')];

criticOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);

critic = rlRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);

actorNetwork = [
    imageInputLayer([1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(2,'Name','action')];

actorOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},actorOpts);

agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',32, ...
    'DiscountFactor',0.99);

agent = rlACAgent(actor,critic,agentOpts);


%% Train Agent

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',500,...
    'MaxStepsPerEpisode',50,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',10,...
    'ScoreAveragingWindowLength',30); 

doTraining = true;

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts)
else
    % Load pretrained agent for the example.
    load('genericMDPQAgent.mat','agent');
end

%% Simulate AC Agent

simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)

