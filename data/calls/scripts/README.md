# Support-call scripts

Scripted technician ↔ manufacturer-support calls, recorded for Fieldnote's audio modality.
Each call is grounded in a real manual page and adds one piece of field advice that is
**not in any manual** (marked `CALL-ONLY` below the script), so some eval questions can
only be answered from the recording.

Recordings are synthetic: `make calls` renders each script with macOS `say`, two voices
per call (rotated across calls, including Australian and British rep voices), into
`data/calls/<script name>.wav` at 16 kHz mono. The WAVs are gitignored and regenerated on
demand. Gold timestamps in `eval/questions.jsonl` were pinned from Whisper transcripts of
these renders. To swap in human recordings, keep the file stems and re-pin the `t` values.

The advice in these scripts is written for a portfolio eval, not as service guidance.

| File | Manual it's grounded in | Call-only fact |
|---|---|---|
| call_01_copeland_coresense | copeland_ae1385_coresense.pdf p.7 | ~30°F condensing split check before blaming the switch; photo the LED first |
| call_02_danfoss_pumpdown | danfoss_sm_sy_sz_scroll.pdf p.33 | verify cut-out on a gauge; cut-in 15–20 psi above cut-out |
| call_03_mitsubishi_u1 | mitsubishi_puz_puy_nha3_service.pdf p.37 | clamp fan amps, check blade slipping on shaft |
| call_04_carrier_38au_astp | carrier_38au_07-14_iss.pdf p.21 | discharge-line thermocouple 6 in out, shut down above ~225°F |
| call_05_carrier_econ_actuator | carrier_economizer_x_llecon.pdf p.14 | power-cycle W7220 to rediscover; ≥22 VAC while driving |
| call_06_bryant_569j_lowambient | bryant_569j_condensing_iss.pdf p.56 | jumper LAS ~2 min to test motors; LAS sensor out of sun/discharge air |
| call_07_trane_precedent_lpc | trane_precedent_yhj_rt-svx075a.pdf p.9 | run supply fan alone ~15 min to clear frost before reset |
| call_08_goodman_coresense_lockedrotor | goodman_gsz14_service.pdf p.98 | clamp common; kill power if LRA lasts > ~3 s |
