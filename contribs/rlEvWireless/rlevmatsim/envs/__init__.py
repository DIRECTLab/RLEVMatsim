from gymnasium.envs.registration import register

register(
    id="RLOCP", 
    entry_point="rlevmatsim.envs.rl_ocp_env:RLOCPEnv",
)
