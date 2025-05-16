from gymnasium.envs.registration import register

register(
    id="MatsimGraphEnvGNN-v0", 
    entry_point="rlevmatsim.envs.rl_ocp_env:RLOCPEnv",
)
