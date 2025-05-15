package org.matsim.contrib.rlEvWireless.charging;

import org.matsim.core.events.handler.EventHandler;

public interface ChargingStartEventHandler extends EventHandler {
    void handleEvent(ChargingStartEvent event);
}
