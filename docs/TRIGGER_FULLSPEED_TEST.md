# Hardware trigger at full speed: bench checklist (EXPERIMENTAL)

**Status: built and mock-tested only. Not yet run on the rig.** This is the checklist for the first run.

The DLPC350 (DLP LightCrafter 4500 EVM, driven from the 4500 MKII GUI) plays a pattern pack from its own flash. On every pattern exposure it raises **TRIG_OUT_1**. That line is wired to a trigger input of the Blackfly S BFS-U3-16S2M. The camera is in hardware trigger mode (FrameStart, RisingEdge) and takes **exactly one full frame per pattern**. No soft sync and no averaging over free-run frames are involved.

Each step below is marked **HUMAN** (done by a person at the bench or in the GUI) or **SOFTWARE** (an engine command, which opens the camera only).

## Safety rules (all steps)

- **The HDMI path is untouched.** Nothing here changes presets, the HDMI scan or the projector window.
- **Pattern mode is entered only by a human, in the GUI.** The engine never sends the DLPC350 a USB / HID command. `scan-trigger` refuses to run without `--pattern-mode-set-by-human`.
- **Never write flash without the firmware backup** (step 1). The GUI cannot read the flash back.
- **LED currents stay within the continuous limits** (R120 G180 B160). Start at the HDMI bench values. See [FLASH_FASTSCAN_4500.md §5.5](FLASH_FASTSCAN_4500.md).
- **Measure the trigger level before wiring** (step 4). Connect grounds first and disconnect them last. Make all wiring changes with the sequence stopped.

## Timing at a glance

A set is the 16 one-bit fast-scan slots. The 1-bit pattern exposure equals the period.

| Pattern period | Rate | 16-pattern set | Camera at full frame (1440×1080 Mono8, ~226 fps → ≥ 4425 µs) | Default camera exposure |
|---|---|---|---|---|
| 10 000 µs | 100 Hz | **160 ms** | OK | 3000 µs |
| 5 000 µs | 200 Hz | **80 ms** | OK | 3000 µs |
| 4 500 µs | 222 Hz | **72 ms** | OK, with about 75 µs of margin | 3000 µs |
| < 4 425 µs | — | — | **refused**: the camera cannot read out a full frame | — |

The pack tool refuses any combination that breaks one of these rules. The message names the rule.

- **DLPC350 rules:**
  - 1-bit exposure ≥ 235 µs.
  - Exposure = period, or period − exposure ≥ 230 µs.
  - 8-bit patterns need exposure ≥ 8333 µs, so the 8-bit hybrid pack cannot run at 5 or 4.5 ms.
- **Camera rules:**
  - Period ≥ 1e6 / 226 µs.
  - 1-bit: camera exposure ≤ pattern exposure − 120 µs guard.
  - 8-bit: pattern exposure ≤ camera exposure ≤ period − 120 µs.

The 226 fps figure is FLIR's full-frame number for this model. It has not been measured on this rig with a trigger. With TriggerOverlap = ReadOut, a period near 4.43 ms leaves almost no margin. Expect 4.5 ms to be the edge.

## 0. Preconditions — HUMAN

- [ ] The projector and the camera have been on for **≥ 20 min** (warm-up). The HDMI video mode shows the desktop on the DMD.
- [ ] A normal HDMI scan works today (`scan-fast` or `scan-metric`). A **stereo calibration** exists at this geometry, because `scan-trigger` refuses to run without one (rule 6).
- [ ] The LED currents are noted (LED Driver Control) and are within the continuous limits.
- [ ] **No engine capture is running.** Modes are changed only in the GUI and never mid-scan.
- [ ] The fast-scan pack's plane check ([FLASH_FASTSCAN_4500.md §5.4](FLASH_FASTSCAN_4500.md)) passed on this unit, if it was done before.

## 1. Firmware backup — HUMAN (nothing proceeds without it)

Follow **[FLASH_FASTSCAN_4500.md §5.1](FLASH_FASTSCAN_4500.md)**:

1. Note the Firmware Version / Tag and the LED currents.
2. Save Solution (`.ini`).
3. Use the **firmware link** in the Image / Firmware tab to save the firmware `.bin`. Its version must equal the unit's. If it does not, use the supplier's `.bin`.

Keep the `.bin` and the `.ini` together. **Never overwrite them.** Every new build is saved under a new name.

## 2. Build the packs — SOFTWARE (files only; touches nothing)

```
scan --preset <the HDMI scan's preset> fastscan-pack --trigger-mode trigout --period-us 10000 --label trig10ms
scan --preset <...> fastscan-pack --trigger-mode trigout --period-us 5000  --label trig5ms
scan --preset <...> fastscan-pack --trigger-mode trigout --period-us 4500  --label trig4p5ms
```

The same builds are available as `python tools/make_flash_pack.py --fastscan --trigger-mode trigout --period-us 5000` (add `--zone-ring …` for a zone image). The optional 8-bit hybrid pack runs at ≥ 8.6 ms only: `scan flash-pack --period-us 10000`.

Each pack's `manifest.json` records `timing.hardware_trigger`:

- the period
- the pattern and camera exposure
- the trigger line
- `set_ms`
- the rules

Its `procedure.txt` adds a section on the hardware-triggered scan. The pack contract checks (`--unpack-check`, `fastscan_manifest_problems`) still pass.

The 16 bit-planes are the same in all three packs. **Only the GUI's period and exposure differ.** A single flash image therefore serves every period: the pack is loaded once (step 3), and only the Sequence Settings change between 10 / 5 / 4.5 ms.

## 3. Load the pack and set the sequence — HUMAN

1. **Load the images.** Follow [FLASH_FASTSCAN_4500.md §5.2](FLASH_FASTSCAN_4500.md): Firmware Build from the **backup** `.bin`, add the pack BMPs, Save Updates under a **new** name, then Upload.
2. **Operating Mode:** Pattern Sequence.
3. **Sequence Settings:**
   - Pattern Source: **Flash**
   - Trigger Mode: **Internal**
   - Bit Depth **1**
   - Pattern Exposure = Pattern Period = the pack's period (10000 first)
   - The 16 rows from `procedure.txt`
   - Play Once **off**
4. **Trigger Controls:** **Trigger Out 1 = normal polarity** (active high), rising delay **0**, falling delay **0**.
5. **Send**, then **Validate Sequence**: every indicator should be clear.
6. **LED Current** at the start values, then **Set**.
7. **Play.**

## 4. Measure TRIG_OUT_1 before wiring — HUMAN

With the sequence playing and **nothing connected** to the camera:

1. Use a meter (DC) or, better, a scope between EVM **J14 pin 2 (Trigger Out 1)** and **J14 pin 3 (GND)**.
   - A meter shows the average: about level × duty. With exposure = period the pulse is high for almost the whole period, so the meter reads close to the high level.
   - A scope shows the pulses and the high level directly.
2. Compare the reading with the camera's input thresholds:

| Input | Low | High | Source |
|---|---|---|---|
| **Line0** (opto-isolated input) | 0 – 1.4 V | **2.6 – 30 V**, 3.5 – 7 mA | FLIR Blackfly S GPIO page (below) |
| **Line3** (non-isolated input) | 0 – 1.4 V | **2.6 – 24 V** | same page |

The EVM's trigger outputs are **1.8 V or 3.3 V**, selected by a jumper (**J13** for Trigger Out 1: pins 3–4 = 3.3 V, pins 5–6 = 1.8 V; DLPU011E §1.5, Fig. 1-11).

- **A 1.8 V output will not trigger either camera line** (high ≥ 2.6 V). Set J13 to **3.3 V** (power off to move a jumper), or use a level shifter.
- **At 3.3 V the opto input (Line0) may still fail.** The opto LED needs 3.5–7 mA, and TRIG_OUT_1 is a logic output whose drive current is not specified in the EVM guide. If step 6 shows NO PULSES at 3.3 V, use **Line3** (a plain logic-level input) or put a buffer / level shifter (e.g. a 5 V logic buffer able to source ≥ 10 mA) in front of Line0.

## 5. Wiring — HUMAN

**CONFIRM every pin against your cable's colour chart and the manuals before connecting.** Wire colours differ between cable batches and third-party cables. Use a meter to check continuity from each wire to its connector pin.

### Camera side: Blackfly S USB3 6-pin GPIO (Hirose HR10A-7R-6PB on the camera; cable plug HR10A-7P-6S)

| Pin | Line | Function | Wire colour (FLIR GPIO cable, per FLIR page) |
|---|---|---|---|
| 1 | **Line3** | non-isolated input; also VAUX (auxiliary power input) | Green |
| 2 | **Line0** | **opto-isolated input** | Black |
| 3 | Line2 | non-isolated bidirectional / 3.3 V VOUT (120 mA) | Red |
| 4 | Line1 | opto-isolated **output** (do not drive it) | White |
| 5 | — | **opto ground** (return for Line0 / Line1) | Blue |
| 6 | — | **GND** (camera ground; return for Line2 / Line3) | Brown |

Sources and confidence:

- **Pin table and electrical levels:** FLIR / Teledyne Blackfly S Installation Guide, *Physical Interface → General Purpose Input/Output (GPIO)*, https://softwareservices.flir.com/BFS-U3-200S6/latest/Family/GPIOConnector.htm. This is the shared *Family* page; the same page under a BFS-U3-16S2 URL was not reachable. The 16S2 spec page (https://softwareservices.flir.com/BFS-U3-16S2/latest/Model/spec.html) lists the same I/O set: 1 opto input, 1 opto output, 1 non-isolated bidirectional, 1 non-isolated input and 3.3 V auxiliary out.
  - Confidence: **single official source**, family-level.
- **Wire colours:** as printed on that FLIR page. **Not verified** against the colour chart of the actual cable (e.g. FLIR's 6-pin GPIO cable accessory). Check with a meter.
- **Line3 pull-up:** whether Line3 needs an external pull-up when used as an input is **not stated** in what was found. The non-isolated lines are open-drain as outputs.
- **Board-level variant:** a board-level BFS-U3-16S2M-BD2 uses a different (JST) GPIO connector. Check that the bench camera is the cased model with the round Hirose connector.

**Use one of these two wirings:**

- **A. Line0 (opto-isolated):** EVM TRIG_OUT_1 → camera **pin 2** (Line0, black). EVM GND → camera **pin 5** (opto ground, blue). The opto ground is isolated from the camera ground, so this wiring needs **both** wires. Command option: `--line Line0` (default).
- **B. Line3 (non-isolated):** EVM TRIG_OUT_1 → camera **pin 1** (Line3, green). EVM GND → camera **pin 6** (GND, brown). Command option: `--line Line3`.

**Never connect the trigger to pin 3 (Line2 / 3.3 V out) or pin 4 (Line1 output).** Leave unused wires insulated.

### Projector side: LightCrafter 4500 EVM output trigger connector J14

| J14 pin | Signal |
|---|---|
| 1 | Trigger Out 1 supply |
| **2** | **Trigger Out 1 (TRIG_OUT_1)** |
| **3** | **GND** |
| 4 | Trigger Out 2 supply |
| 5 | Trigger Out 2 |
| 6 | GND |

- **Level:** jumper **J13** sets Trigger Out 1 (J15 sets Trigger Out 2): pins 3–4 = 3.3 V, pins 5–6 = 1.8 V.
- **Mating connector:** 6-pin, 1.25 mm pitch, housing Molex 51021-0600 with crimp terminals Molex 50079-8100.
- **Trigger inputs:** J11 (Trigger In 1 on pin 2, GND on pin 3), with level jumpers J10 / J12. These are used only for `--trigger-mode external`.
- **Sources:**
  - TI **DLPU011E**, *DLP LightCrafter 4500 EVM User's Guide*: §7.2 Table 7-2 (Output Trigger Connector, J14), §7.1 Table 7-1 (J11), §1.5 / Fig. 1-11 (trigger level jumpers), Figs. 1-7 / 1-8 (connector locations). A local copy is in `docs/` (gitignored); online at https://www.ti.com/lit/pdf/dlpu011.
  - Confidence: **verified in the document** (pin tables, jumper settings, mating parts). The board-photo locations were not checked visually.
- **Not verified for the MKII:** DLPU011E does not mention a "4500 MKII". Its applicability to this unit's board revision is unverified. **Check the silkscreen** for J14 / J13 before connecting.
- **Discrepancy in DLPU011E:** §1.4 says the trigger inputs support 5 V, 3.3 V and 1.8 V, while the jumper table lists only 3.3 V and 1.8 V. This does not matter for TRIG_OUT_1.

## 6. Pulse check — SOFTWARE (camera only)

The sequence must be playing (step 3, Repeat on) and the wiring done (step 5).

```
scan trigger-test --line Line0 --seconds 5 --expect-period-ms 10
```

The command:

1. Puts the camera in hardware trigger on the line (FrameStart, RisingEdge, 1 ms exposure).
2. Counts the triggered frames for 5 s. Timeouts count as zero.
3. Prints the count, the rate (Hz), the period mean / std / min / max from the **camera timestamps**, the estimated missed pulses and frame-id gaps, and a verdict.
4. **Always** sets TriggerMode back to Off.

Exit codes: 0 = OK, 1 = NO PULSES or RATE MISMATCH.

| Verdict | Meaning / what to do |
|---|---|
| **OK** | The rate matches the GUI period and nothing was missed. Go to step 7. |
| **NO PULSES** | No trigger reached the camera. Check that the sequence plays, the J14 pin 2/3 wiring, the camera pins (Line0 = pin 2 + opto GND pin 5), and the level (J13 at 3.3 V). Then try `--line Line3` (pins 1 / 6) or a level shifter. |
| **RATE MISMATCH** | The pulse period differs from the expected period (the GUI period differs from the pack), pulses are missed (level near the threshold, or a period too short for the camera), pulses come in doubles (ringing: shorten the wires, use a common ground), or frame ids skip (USB). |

## 7. First triggered scan at 10 ms — SOFTWARE

Same scene as an HDMI reference scan. The sequence is playing (Repeat on).

```
scan --preset <...> scan-trigger trig10 --pack <trig10ms pack folder> --variant full --line Line0 --repeats 2 --pattern-mode-set-by-human
```

What happens:

1. The camera grabs exactly (repeats + 1) × 16 frames. The extra loop is used to find the start. Missing frames, extra frames, spacing ±50 % of the period and the first-trigger timeout (30 s) are checked by `TriggeredSequenceCapture`.
2. The frames are aligned on the loop's sync markers (sync_code = horizontal bars, sync_white just before it). A G/R plane swap in the GUI only rotates the loop, which the alignment absorbs.
3. The loops are averaged per slot.
4. The result goes through the unchanged multifreq decode and 'plane' triangulation, into a metric scan (meta `mode: flash_trigger`, `trigger.timing`: frames, total ms, per-frame ms, drops).

Check:

- [ ] Frames = 48 (16 × 3), 0 dropped, per frame ≈ 10.000 ms.
- [ ] Valid % and point count are close to the HDMI scan.
- [ ] Repeat 3×: point count and plane rms are stable.

Refusals: exit 2 with the reason (e.g. "got k of N frames", "frame ids jump", "spacing", "no sync_code frame stands out").

## 8. Step down: 5 ms, then 4.5 ms — HUMAN + SOFTWARE

1. **HUMAN:** Stop, set Pattern Exposure = Pattern Period = 5000 µs, Send, Validate, Play.
2. **SOFTWARE:** `scan trigger-test --expect-period-ms 5`, then `scan-trigger trig5 --pack <trig5ms pack> …` (the 5 ms pack: its manifest carries the 5 ms timing that the capture checks against).
3. Repeat at **4500 µs** with the 4.5 ms pack. The first sign of the camera's limit is frame-id jumps or RATE MISMATCH: stay at 5 ms if that happens.

## 9. Compare with the HDMI fast scan — SOFTWARE

On the same scene, without touching anything between the scans:

- [ ] **Point count:** triggered vs `scan-fast` (HDMI) with the same zone.
- [ ] **Plane rms** on a flat reference (e.g. `scan accuracy` or a plane fit on the flat).
- [ ] **A known step height** (gauge block or the bench step): the triggered result should agree with the HDMI result within its repeatability.
- [ ] **Capture time:** 160 / 80 / 72 ms per set vs ~5 s HDMI.

The 1-bit fine level reads a constant offset of about 0.04 column against HDMI 8-bit fringes on the diamond DMD. Relative heights are unaffected; see FLASH_FASTSCAN_4500.md.

## 10. Back to video mode / rollback — HUMAN

1. **Stop** the sequence.
2. Unplug the trigger wires (signal first, ground last).
3. Return to video mode as in [FLASH_FASTSCAN_4500.md §5.8](FLASH_FASTSCAN_4500.md): Operating Mode → Video Mode, Source Parallel RGB, LED currents back to the step-0 values.
4. Run a normal HDMI scan to confirm.
5. **Rollback** (only if the unit misbehaves): [§5.9](FLASH_FASTSCAN_4500.md). Upload the backup `.bin`, then Apply Solution with the `.ini`.

The camera needs nothing: every command sets TriggerMode back to Off when it exits. If a crash leaves it on, the next normal scan reopens the camera with the preset settings. SpinView can also set TriggerMode Off.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| NO PULSES, scope shows pulses at J14 | Level too low for the opto (1.8 V, or 3.3 V without current) | J13 → 3.3 V; use Line3 (pins 1/6); or a buffer / level shifter |
| NO PULSES, no pulses at J14 | Sequence not playing, Trigger Out 1 disabled, or the wrong connector | Play; Trigger Controls; check J14 on the silkscreen |
| Line0 never fires, Line3 works | Opto ground (pin 5) not connected to EVM GND | Connect pin 5 to J14 pin 3 |
| RATE MISMATCH, period = 2× expected | GUI period ≠ pack period, or every other pulse missed | Match the GUI to the pack; check the level |
| Double pulses (period ≪ expected) | Ringing, long wires, no common ground | Short twisted pair, common ground |
| scan-trigger: "frame ids jump" | Camera dropped frames (USB, buffers) or the period is at the camera's limit | Longer period; dedicated USB3 port |
| scan-trigger: "spacing" | The GUI timing differs from the pack's manifest | Use the pack built for the period the GUI runs |
| scan-trigger: "no sync_code frame stands out" | Not the fast-scan loop, or the trigger is not one frame per pattern | Check the sequence rows (procedure.txt), Trigger Out 1 polarity, plane check |
| scan-trigger: black not darkest | Wrong plane order in the GUI | Plane check (FLASH_FASTSCAN_4500.md §5.4) |
| Refused: no stereo calibration | Rule 6 | `calibrate-stereo` at this geometry first |

## Not verified yet

- Nothing in this document has run on the rig: not the PySpin trigger code path, the pulse levels, the opto drive current, or the 4.5 ms limit.
- FLIR pin table: verified on the Blackfly S family page only. The cable wire colours are not checked against the actual cable.
- TI J14 / J13 pinout: from DLPU011E. Its applicability to the 4500 MKII board revision is not checked.
- The camera's full-frame rate under hardware trigger with TriggerOverlap = ReadOut (226 fps is FLIR's free-run figure).

## Code

- `engine/capture/hwtrigger.py`: pulse statistics and verdict, pulse counting, 1-bit loop alignment, and the fake triggered camera.
- `engine/reconstruct/trigger_scan.py`: `trigger_test`, `scan_trigger` (camera-only context manager; TriggerMode Off in `finally`).
- `engine/capture/patternmode.py`: `TriggeredSequenceCapture` guards, the PySpin trigger setup.
- `tools/make_flash_pack.py`: `triggered_timing`, `trigger_timing_errors`, `fastscan_triggered_timing`, `--trigger-mode`, `--period-us`.
- Tests: `tests/test_trigger_fullspeed_mock.py`.
