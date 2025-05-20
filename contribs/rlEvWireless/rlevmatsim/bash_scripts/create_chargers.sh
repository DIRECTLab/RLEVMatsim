cd ..
# python -m rlevmatsim.scripts.create_chargers "./scenario_examples/utahev_scenario_example/utahevnetwork.xml" \
# "./scenario_examples/utahev_scenario_example/utahevchargers.xml" "--percent" "0.5" "--percent_dynamic" "0.5" 
python -m rlevmatsim.scripts.create_chargers "./scenario_examples/i-15-scenario/i-15-network.xml" \
"./scenario_examples/i-15-scenario/i-15-chargers.xml" "--percent" "0.5" "--percent_dynamic" "0.5" 