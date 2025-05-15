from gymnasium.envs.registration import register

register(
    id="MatsimGraphEnvGNN-v0", 
    entry_point="rlevmatsim.envs.matsim_graph_env_gnn:MatsimGraphEnvGNN",
)

register(
    id="MatsimGraphEnvMlp-v0", 
    entry_point="rlevmatsim.envs.matsim_graph_env_mlp:MatsimGraphEnvMlp",
)

register(
    id="FlowMatsimGraphEnvMlp-v0", 
    entry_point="rlevmatsim.envs.flow_matsim_graph_env:FlowMatsimGraphEnv",
)

register(
    id="StatFlowMatsimGraphEnvMlp-v0", 
    entry_point="rlevmatsim.envs.stat_flow_matsim_graph_env:StatFlowMatsimGraphEnv",
)

register(
    id="ClusterFlowMatsimGraphEnv-v0", 
    entry_point="rlevmatsim.envs.cluster_flow_matsim_graph_env:ClusterFlowMatsimGraphEnv",
)

register(
    id="FlowSimEnv-v0", 
    entry_point="rlevmatsim.envs.flowsim_env:FlowSimEnv",
)