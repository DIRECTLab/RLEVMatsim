package org.matsim.contrib.rlEvWireless.charging;

import org.matsim.core.events.handler.EventHandler;

public interface QueuedAtChargerEventHandler extends EventHandler {
	void handleEvent(QueuedAtChargerEvent event);
}
