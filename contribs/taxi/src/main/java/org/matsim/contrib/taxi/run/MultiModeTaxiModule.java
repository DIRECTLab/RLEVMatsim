/*
 * *********************************************************************** *
 * project: org.matsim.*
 * *********************************************************************** *
 *                                                                         *
 * copyright       : (C) 2019 by the members listed in the COPYING,        *
 *                   LICENSE and WARRANTY file.                            *
 * email           : info at matsim dot org                                *
 *                                                                         *
 * *********************************************************************** *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *   See also COPYING, LICENSE and WARRANTY file                           *
 *                                                                         *
 * *********************************************************************** *
 */

package org.matsim.contrib.taxi.run;

import org.matsim.contrib.drt.fare.DrtFareParams;
import org.matsim.contrib.drt.optimizer.constraints.DrtOptimizationConstraintsSetImpl;
import org.matsim.contrib.drt.run.DrtConfigGroup;
import org.matsim.contrib.drt.run.DrtModeModule;
import org.matsim.contrib.drt.run.DrtModeQSimModule;
import org.matsim.contrib.taxi.analysis.TaxiModeAnalysisModule;
import org.matsim.contrib.taxi.optimizer.TaxiModeOptimizerQSimModule;
import org.matsim.core.controler.AbstractModule;

import com.google.inject.Inject;

/**
 * @author Michal Maciejewski (michalm)
 */
public class MultiModeTaxiModule extends AbstractModule {

	@Inject
	private MultiModeTaxiConfigGroup multiModeTaxiCfg;

	@Override
	public void install() {
		for (TaxiConfigGroup taxiCfg : multiModeTaxiCfg.getModalElements()) {
			var drtCfg = convertTaxiToDrtCfg(taxiCfg);
			install(new DrtModeModule(drtCfg));
			installQSimModule(new DrtModeQSimModule(drtCfg, new TaxiModeOptimizerQSimModule(taxiCfg)));
			install(new TaxiModeAnalysisModule(taxiCfg));
		}
	}

	public static DrtConfigGroup convertTaxiToDrtCfg(TaxiConfigGroup taxiCfg) {

		// Taxi specific settings, not applicable directly to DRT
		// - destinationKnown
		// - vehicleDiversion
		// - onlineVehicleTracker
		// - breakSimulationIfNotAllRequestsServed
		// - taxiOptimizerParams

		var drtCfg = new DrtConfigGroup();

		drtCfg.setMode(taxiCfg.getMode());
		drtCfg.setUseModeFilteredSubnetwork(taxiCfg.useModeFilteredSubnetwork);
		drtCfg.setStopDuration(Double.NaN);//used only inside the DRT optimiser

		// Taxi optimisers do not reject, so time constraints are only used for routing plans (DrtRouteCreator).
		// Using some (relatively high) values as we do not know what values should be there. They can be adjusted
		// manually after the TaxiAsDrtConfigGroup config is created.
		DrtOptimizationConstraintsSetImpl defaultConstraintsSet =
                drtCfg.addOrGetDrtOptimizationConstraintsParams()
                        .addOrGetDefaultDrtOptimizationConstraintsSet();
		defaultConstraintsSet.setMaxWaitTime(3600);
		defaultConstraintsSet.setMaxTravelTimeAlpha(2);
		defaultConstraintsSet.setMaxTravelTimeBeta(3600);
		defaultConstraintsSet.setMaxAbsoluteDetour(Double.MAX_VALUE);

		defaultConstraintsSet.setRejectRequestIfMaxWaitOrTravelTimeViolated(false);
		drtCfg.setChangeStartLinkToLastLinkInSchedule(taxiCfg.changeStartLinkToLastLinkInSchedule);
		drtCfg.setIdleVehiclesReturnToDepots(false);
		drtCfg.setOperationalScheme(DrtConfigGroup.OperationalScheme.door2door);
		defaultConstraintsSet.setMaxWalkDistance(Double.MAX_VALUE);
		drtCfg.setVehiclesFile(taxiCfg.taxisFile);
		drtCfg.setTransitStopFile(null);
		drtCfg.setDrtServiceAreaShapeFile(null);
		drtCfg.setPlotDetailedCustomerStats(taxiCfg.detailedStats || taxiCfg.timeProfiles);
		drtCfg.setNumberOfThreads(taxiCfg.numberOfThreads);
		drtCfg.setStoreUnsharedPath(false);

		taxiCfg.getTaxiFareParams().ifPresent(taxiFareParams -> {
			var drtFareParams = new DrtFareParams();
			drtFareParams.setBaseFare(taxiFareParams.getBasefare());
			drtFareParams.setDistanceFare_m(taxiFareParams.getDistanceFare_m());
			drtFareParams.setTimeFare_h(taxiFareParams.getTimeFare_h());
			drtFareParams.setDailySubscriptionFee(taxiFareParams.getDailySubscriptionFee());
			drtFareParams.setMinFarePerTrip(taxiFareParams.getMinFarePerTrip());
			drtCfg.addParameterSet(drtFareParams);
		});

		// DRT specific settings, not existing in taxi
		// - drtCfg.drtInsertionSearchParams
		// - drtCfg.zonalSystemParams
		// - drtCfg.rebalancingParams
		// - drtCfg.drtSpeedUpParams
		// - drtCfg.drtRequestInsertionRetryParams

		return drtCfg;
	}
}
