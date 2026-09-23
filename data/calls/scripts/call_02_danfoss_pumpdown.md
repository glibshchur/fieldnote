# Call 02: Danfoss technical support, SZ scroll short-cycling on pump-down

TECH: Hey, I've got a Danfoss SZ scroll on R-134a, it's a pump-down system, and it's short-cycling on the low-pressure pump-down switch. Several starts in just a few minutes.
REP: Okay. First, the numbers from the application guide. Minimum low-pressure safety switch setting is seven psig, and the minimum pump-down setting for R-134a is also seven psig. We recommend pump-down about one bar below the nominal evaporating pressure. And the LP safety is never bypassed, and it has no time delay.
TECH: Right. How often can it start?
REP: No more than twelve starts per hour, and we recommend a three-minute anti-short-cycle timer, a hundred and eighty seconds.
TECH: I've got the pump-down switch dialed to what the scale says should be fine.
REP: Don't trust the dial scale on those adjustable switches. They can be off by several psi. Put a compound gauge on the suction port and watch the actual cut-out while the system slowly pumps down.
TECH: Okay.
REP: And then set your cut-in well above the cut-out. Roughly fifteen to twenty psi above. If the cut-in is too close, the pressure rebounds after pump-down and the switch rapid-cycles the compressor. That's probably what you're seeing.
TECH: That tracks. The differential is barely anything right now.
REP: Open it up, add the timer if it doesn't have one, and you should be good.

<!-- CALL-ONLY: verify actual cut-out with a compound gauge during slow pump-down instead of the dial scale; set cut-in roughly 15–20 psi above cut-out. -->
