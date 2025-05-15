cd ../contribs/cadytsIntegration

export MAVEN_OPTS="-Xmx61G"

mvn exec:java -Dexec.mainClass="org.matsim.contrib.cadyts.run.RunCadyts4CarExample" -Dexec.args="../evWireless/scenario_examples/utah_flow_scenario_example/utahconfig.xml"
