cd ../../

mvn exec:java -Dexec.mainClass="org.matsim.contrib.rlEvWireless.OCPRewardServer" -Dexec.args="1 8000"
python -m rlevmatsim.sim_learner \
"./scenario_examples/tinytown_scenario_example_1agent/ev_tiny_town_config.xml" \
"--results_dir" "./tinytownsimlearner_results" \
"--epochs" "10000"