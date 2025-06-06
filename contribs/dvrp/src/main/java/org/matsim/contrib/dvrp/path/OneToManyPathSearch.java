/* *********************************************************************** *
 * project: org.matsim.*
 *                                                                         *
 * *********************************************************************** *
 *                                                                         *
 * copyright       : (C) 2016 by the members listed in the COPYING,        *
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
 * *********************************************************************** */

package org.matsim.contrib.dvrp.path;

import static org.matsim.core.router.util.LeastCostPathCalculator.Path;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import org.matsim.api.core.v01.IdMap;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Node;
import org.matsim.core.router.speedy.LeastCostPathTree;
import org.matsim.core.router.speedy.SpeedyGraph;
import org.matsim.core.router.util.TravelDisutility;
import org.matsim.core.router.util.TravelTime;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

public class OneToManyPathSearch {
	public static OneToManyPathSearch createSearch(SpeedyGraph graph, TravelTime travelTime,
			TravelDisutility travelDisutility, boolean lazyPathCreation) {
		return new OneToManyPathSearch(new LeastCostPathTree(graph, travelTime, travelDisutility), travelTime,
				lazyPathCreation);
	}

	public static class PathData {
		public static final PathData EMPTY = new PathData(0, 0);
		public static final PathData INFEASIBLE = new PathData(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);

		private final Supplier<Path> pathSupplier;
		private Path path;

		private final double travelTime;

		private PathData(double travelTime, double travelCost) {
			this.pathSupplier = null;
			this.path = new Path(null, null, travelTime, travelCost);
			this.travelTime = travelTime;
		}

		public PathData(Path path, double firstAndLastLinkTT) {
			this.pathSupplier = null;
			this.path = new Path(path.nodes!= null ? ImmutableList.copyOf(path.nodes) : null, ImmutableList.copyOf(path.links), path.travelTime, path.travelCost);
			this.travelTime = path.travelTime + firstAndLastLinkTT;
		}

		public PathData(Supplier<Path> pathSupplier, double pathTravelTime, double firstAndLastLinkTT) {
			this.pathSupplier = pathSupplier;
			this.travelTime = pathTravelTime + firstAndLastLinkTT;
		}

		public double getTravelTime() {
			return travelTime;
		}

		//package visibility only (path.nodes is null)
		public Path getPath() {
			if (path == null) {
				path = pathSupplier.get();
			}
			return path;
		}
	}

	private final LeastCostPathTree dijkstraTree;
	private final TravelTime travelTime;
	private final boolean lazyPathCreation;

	private OneToManyPathSearch(LeastCostPathTree dijkstraTree, TravelTime travelTime,
			boolean lazyPathCreation) {
		this.dijkstraTree = dijkstraTree;
		this.travelTime = travelTime;
		this.lazyPathCreation = lazyPathCreation;
	}

	public PathData[] calcPathDataArray(Link fromLink, List<Link> toLinks, double startTime, boolean forward) {
		return calcPathDataArray(fromLink, toLinks, startTime, forward, Double.POSITIVE_INFINITY);
	}

	public PathData[] calcPathDataArray(Link fromLink, List<Link> toLinks, double startTime, boolean forward,
			double maxTravelTime) {
		OneToManyPathCalculator pathConstructor = new OneToManyPathCalculator(dijkstraTree, travelTime,
				forward, fromLink, startTime);
		pathConstructor.calculateDijkstraTree(toLinks, maxTravelTime);
		return createPathDataArray(toLinks, pathConstructor);
	}

	public Map<Link, PathData> calcPathDataMap(Link fromLink, Collection<Link> toLinks, double startTime,
			boolean forward) {
		return calcPathDataMap(fromLink, toLinks, startTime, forward, Double.POSITIVE_INFINITY);
	}

	public Map<Link, PathData> calcPathDataMap(Link fromLink, Collection<Link> toLinks, double startTime,
			boolean forward, double maxTravelTime) {
		OneToManyPathCalculator pathCalculator = new OneToManyPathCalculator(dijkstraTree, travelTime, forward,
				fromLink, startTime);
		pathCalculator.calculateDijkstraTree(toLinks, maxTravelTime);
		return createPathDataMap(toLinks, pathCalculator);
	}

	private PathData[] createPathDataArray(List<Link> toLinks, OneToManyPathCalculator pathCalculator) {
		PathData[] pathDataArray = new PathData[toLinks.size()];
		for (int i = 0; i < pathDataArray.length; i++) {
			pathDataArray[i] = createPathData(pathCalculator, toLinks.get(i));
		}
		return pathDataArray;
	}

	private Map<Link, PathData> createPathDataMap(Collection<Link> toLinks, OneToManyPathCalculator pathCalculator) {
		Map<Link, PathData> pathDataMap = Maps.newHashMapWithExpectedSize(toLinks.size());
		for (Link toLink : toLinks) {
			pathDataMap.put(toLink, createPathData(pathCalculator, toLink));
		}
		return pathDataMap;
	}

	private PathData createPathData(OneToManyPathCalculator pathCalculator, Link toLink) {
		return lazyPathCreation ?
				pathCalculator.createPathDataLazily(toLink) :
				pathCalculator.createPathDataEagerly(toLink);
	}
}
