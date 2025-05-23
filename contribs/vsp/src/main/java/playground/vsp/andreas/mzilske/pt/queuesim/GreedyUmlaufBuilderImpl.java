package playground.vsp.andreas.mzilske.pt.queuesim;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.pt.Umlauf;
import org.matsim.pt.UmlaufBuilder;
import org.matsim.pt.UmlaufImpl;
import org.matsim.pt.UmlaufInterpolator;
import org.matsim.pt.UmlaufStueck;
import org.matsim.pt.UmlaufStueckI;
import org.matsim.pt.transitSchedule.api.Departure;
import org.matsim.pt.transitSchedule.api.TransitLine;
import org.matsim.pt.transitSchedule.api.TransitRoute;
import org.matsim.pt.transitSchedule.api.TransitRouteStop;
import org.matsim.pt.transitSchedule.api.TransitStopFacility;

import com.google.inject.Inject;

public final class GreedyUmlaufBuilderImpl implements UmlaufBuilder {
	private static final Logger log = LogManager.getLogger( GreedyUmlaufBuilderImpl.class );

	public static class UmlaufKey {

		private final Id<TransitStopFacility> stopFacilityId;
		private final double lastArrivalTime;
		private final Id<Umlauf> umlaufId;


		Id<Umlauf> getUmlaufId() {
			return umlaufId;
		}

		UmlaufKey( Id<TransitStopFacility> stopFacilityId, double lastArrivalTime, Id<Umlauf> umlaufId ) {
			super();
			this.stopFacilityId = stopFacilityId;
			this.lastArrivalTime = lastArrivalTime;
			this.umlaufId = umlaufId;
		}

		public Id<TransitStopFacility> getStopFacility() {
			return stopFacilityId;
		}

		double getLastArrivalTime() {
			return lastArrivalTime;
		}

		@Override
		public String toString() {
			return stopFacilityId + " at " + lastArrivalTime + "(" + getUmlaufId() + ")";
		}


	}

	private static final Comparator<UmlaufStueck> departureTimeComparator = new Comparator<>() {

		@Override
		public int compare(UmlaufStueck o1, UmlaufStueck o2) {
			return Double.compare(o1.getDeparture().getDepartureTime(), o2.getDeparture().getDepartureTime());
		}

	};

	private static final Comparator<UmlaufKey> umlaufKeyComparator = new Comparator<>() {

		@Override
		public int compare(UmlaufKey o1, UmlaufKey o2) {
			int c = o1.getStopFacility().compareTo(o2.getStopFacility());
			if (c != 0) {
				return c;
			} else {
				int cc = o1.getUmlaufId().compareTo(o2.getUmlaufId());
				if (cc != 0) {
					return cc;
				} else {
					return Double.compare(o1.getLastArrivalTime(), o2.getLastArrivalTime());
				}
			}
		}

	};

	private final Collection<TransitLine> transitLines;
	private final SortedMap<UmlaufKey,Umlauf> umlaeufe = new TreeMap<UmlaufKey,Umlauf>(umlaufKeyComparator);
	private ArrayList<UmlaufStueck> umlaufStuecke;

	private final UmlaufInterpolator interpolator;

	public GreedyUmlaufBuilderImpl(UmlaufInterpolator interpolator, Collection<TransitLine> transitLines) {
		this.interpolator = interpolator;
		this.transitLines = transitLines;
	}

	@Inject
	GreedyUmlaufBuilderImpl( Scenario scenario ) {
		interpolator = new UmlaufInterpolator( scenario.getNetwork(), scenario.getConfig().scoring() );
		this.transitLines = scenario.getTransitSchedule().getTransitLines().values();
	}

	private static boolean canBuild() {
		return true;
	}

	@Override
	public Collection<Umlauf> build() {
		if (!canBuild()) {
			throw new IllegalArgumentException();
		}
		createUmlaufStuecke();
		int id = 0;
		for (UmlaufStueck umlaufStueck : umlaufStuecke) {
			Umlauf umlauf = findFittingUmlauf(umlaufStueck);
			if (umlauf == null) {
				umlauf = new UmlaufImpl(Id.create(id++, Umlauf.class));
			} else {
				umlaeufe.remove(getKey(umlauf));
			}
			interpolator.addUmlaufStueckToUmlauf(umlaufStueck, umlauf);
			umlaeufe.put(getKey(umlauf), umlauf);
		}
		return umlaeufe.values();
	}

	private static double getLastArrivalTime( Umlauf umlauf ) {
		TransitRouteStop previousStop = getLastStop(umlauf);
		double previousDepartureTime = getLastDeparture(umlauf).getDepartureTime();
		double arrivalOffset = previousStop.getArrivalOffset().seconds();
		double previousArrivalTime = previousDepartureTime + arrivalOffset;
		return previousArrivalTime;
	}

	private static Departure getLastDeparture( Umlauf umlauf ) {
		return getLastUmlaufStueck(umlauf).getDeparture();
	}

	private static TransitRouteStop getLastStop( Umlauf umlauf ) {
		UmlaufStueckI lastUmlaufStueck = getLastUmlaufStueck(umlauf);
		return getLastStop(lastUmlaufStueck);
	}

	private static TransitRouteStop getLastStop( UmlaufStueckI umlaufStueck ) {
		List<TransitRouteStop> stops = umlaufStueck.getRoute().getStops();
		return stops.getLast();
	}

	private static UmlaufStueckI getLastUmlaufStueck( Umlauf umlauf ) {
		return umlauf.getUmlaufStuecke().getLast();
	}

	private static String getLastStopPostAreaId( Umlauf umlauf ) {
		return getLastStop(umlauf).getStopFacility().getStopAreaId().toString();
	}

	private UmlaufKey getKey(Umlauf umlauf) {
		return new UmlaufKey(Id.create(getLastStopPostAreaId(umlauf), TransitStopFacility.class), getLastArrivalTime(umlauf), umlauf.getId());
	}

	private Umlauf findFittingUmlauf(UmlaufStueck umlaufStueck) {
		String firstStopPostAreaId = umlaufStueck.getRoute().getStops().getFirst().getStopFacility().getStopAreaId().toString();
		UmlaufKey earliestAtPoint = new UmlaufKey(Id.create(firstStopPostAreaId, TransitStopFacility.class), 0.0, Id.create(0, Umlauf.class));
		UmlaufKey latestAtPoint = new UmlaufKey(Id.create(firstStopPostAreaId, TransitStopFacility.class), umlaufStueck.getDeparture().getDepartureTime(), Id.create(0, Umlauf.class));
		log.info("Looking between " + earliestAtPoint + " and " + latestAtPoint);
		SortedMap<UmlaufKey,Umlauf> fittingUmlaeufe = umlaeufe.subMap(earliestAtPoint, latestAtPoint);
		Umlauf fittingUmlauf;
		if (fittingUmlaeufe.isEmpty()) {
			fittingUmlauf = null;
		} else {
			fittingUmlauf = fittingUmlaeufe.get(fittingUmlaeufe.firstKey());
			log.info("Found " + getKey(fittingUmlauf));
		}
		return fittingUmlauf;
	}

	private void createUmlaufStuecke() {
		this.umlaufStuecke = new ArrayList<>();
		for (TransitLine line : transitLines) {
			for (TransitRoute route : line.getRoutes().values()) {
				for (Departure departure : route.getDepartures().values()) {
					UmlaufStueck umlaufStueck = new UmlaufStueck(line, route, departure);
					umlaufStuecke.add(umlaufStueck);
				}
			}
		}
		this.umlaufStuecke.sort(departureTimeComparator);
	}




}
