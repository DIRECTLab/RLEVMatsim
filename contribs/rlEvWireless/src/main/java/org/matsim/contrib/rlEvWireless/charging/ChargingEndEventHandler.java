package org.matsim.contrib.rlEvWireless.charging;

import org.matsim.core.events.handler.EventHandler;

public interface ChargingEndEventHandler extends EventHandler {
    void handleEvent(ChargingEndEvent event);
}
