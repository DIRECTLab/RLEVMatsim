cd ../../

num_threads=$1
port=$2

mvn exec:java -Dexec.mainClass="org.matsim.contrib.rlEvWireless.OCPRewardServer" -Dexec.args="$num_threads $port"
