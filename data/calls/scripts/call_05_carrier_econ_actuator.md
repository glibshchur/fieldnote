# Call 05: Carrier technical support, Economizer X actuator alarm

TECH: I just replaced the damper actuator on a rooftop with the Economizer X, the W7220 controller. New actuator's an MS3105. I put it in test mode to check the stroke and it worked fine, but now the W7220 shows an actuator alarm and the damper won't move.
REP: Where's the mode select dial on the actuator right now?
TECH: Uh, still on six I think.
REP: Six is test mode. With the W7220 it has to be on position one, that's address eleven. Put it back to one.
TECH: Okay, done. Still alarming.
REP: The S-bus is a two-wire network, polarity doesn't matter, and the controller discovers devices on it. After you set the dial, cycle the twenty-four volt power to the W7220 so it rediscovers the actuator.
TECH: Okay, cycling now.
REP: And while it's driving, put your meter on ACT 24V plus and ACT COM. You want at least about twenty-two volts AC. A sagging transformer gives you the same actuator alarm.
TECH: Twenty-three and change. And the alarm cleared.

<!-- CALL-ONLY: after setting the dial to position 1, power-cycle the W7220 so it rediscovers the actuator; confirm at least ~22 VAC at ACT 24V+/ACT COM while driving. -->
