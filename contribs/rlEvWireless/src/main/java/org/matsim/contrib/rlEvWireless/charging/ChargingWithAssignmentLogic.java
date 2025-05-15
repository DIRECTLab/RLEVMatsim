package org.matsim.contrib.rlEvWireless.charging;

import org.matsim.contrib.rlEvWireless.fleet.ElectricVehicle;

import java.util.Collection;

public interface ChargingWithAssignmentLogic extends ChargingLogic {
	void assignVehicle(ElectricVehicle ev);

	void unassignVehicle(ElectricVehicle ev);

	Collection<ElectricVehicle> getAssignedVehicles();
}
