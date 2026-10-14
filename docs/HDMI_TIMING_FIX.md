# HDMI timing fix for the "blue lines" failure

**Symptom:** after a DLPC350 USB display-mode or input-source command, the DMD stops showing the HDMI desktop and drops to its internal test pattern (blue lines). Only a power cycle has recovered it so far.

**Scope:** this procedure and `engine/capture/dmd_control.py` are for diagnostics of the video input only. **They are not part of the scan path.** The scan path (`first_scan_v10.py`, `engine/capture/projector.py`) never sends USB display-mode or input-source commands (ENGINE_ARCHITECTURE §9 rule 1). The engine also refuses to project in any process where the USB driver is loaded.

**What the tool sends:** one status read, and one write (Input Source Selection). It cannot switch the projector display (video / pattern sequence): it has no such command, and the driver under it refuses that write always. The write commands (`--relock`, `--source`, `--bitdepth`) make the projector re-acquire its video input, so the picture leaves the wall and may not come back without a power cycle. They need `--yes`; without it the tool exits 2 and sends nothing. Every command, `--status` included, takes the rig lock and exits 2 while a scan holds the rig.

## Root cause (working hypothesis)

| Claim | Source |
|---|---|
| Video display mode assumes streaming input "up to 1280 × 800 up to 120 Hz". Pattern display mode assumes 912 × 1140 and bypasses image processing. | DLPU010G §2.4.1, Table 2-49; §2.4.2 |
| The EVM's HDMI input accepts 912 × 1140 in video mode (scaled to native). | DLPU011E, connector 20 |
| Buffer-swap errors "can occur if the DLP LightCrafter 4500 is set to Video Mode and the vertical backporch timing is too small." | DLPU011E §3.2, "Forced Swap" status indicator |
| The input-signal status read is the debugging aid "to detect if the controller reads the resolution correctly". | DLPU010G §2.1.5 |
| This laptop's 912 × 1140 custom mode was made with **CVT-RB** (reduced blanking), giving a small vertical back porch. | bench / Intel Graphics Command Center |
| Timing must meet DLPC350 datasheet §6.7/6.8, and parallel RGB must be 24-bit, not 30-bit. | TI E2E guidance (reported by the operator; not a TI document in hand) |

**Chain of events:** the marginal CVT-RB timing locks at boot. A USB mode or source command forces the controller to re-acquire the input, the re-lock fails, and the controller falls back to its internal test pattern.

Useful fingerprints:
- **CVT-RB** has exactly **160 px horizontal blanking** and a minimum vertical blank of about 460 µs.
- **Standard CVT** has much larger horizontal blanking and at least 550 µs of vertical sync plus back porch.

The status readout shows both, so you can tell which mode is live.

## Commands used (all DLPU010G)

| Command | CMD2 | CMD3 | Kind | Section |
|---|---|---|---|---|
| Input Video Signal Detection Status | **0x07** | **0x1C** | read, 28 bytes, firmware ≥ 2.0.0 | §2.1.5, Table 2-5 (p.13–14). Appendix quick reference p.68 lists `0x071C`, 28 bytes. The §2.1.5 header also shows a stray "USB: 0x04" that appears nowhere else; treated as a typo. |
| Input Source Selection | 0x1A | 0x00 | write | §2.3.4.2, Table 2-19. Bits 2:0 are the source (0 parallel/HDMI, 1 test pattern, 2 flash, 3 FPD-link). Bits 5:3 are the parallel bit depth (0=30, **1=24**, 2=20, 3=16, 4=10, 5=8). |
| Display Mode Selection | 0x1A | 0x1B | **not sent by the tool** (named because it is the other command that makes the controller re-acquire its input) | §2.4.1, Table 2-49 (0 video, 1 pattern) |

The wire format is `swiss_army_sensor.drivers.dlp_driver`, which is byte-verified against pycrafter4500: CMD3 goes on the wire before CMD2. The tool closes the HID handle **without** parking the DMD.

## Procedure

Use the `scan` conda env. From the repo root:

```
set PY=python
```

**0. Preconditions.**
- The LightCrafter GUI is **closed** while the tool runs, because both open the same HID device.
- No scan is running (the tool takes the rig lock for every command and is refused while a scan holds it).

**1. Record the BEFORE timing,** with the current CVT-RB mode still active:

```
%PY% -m engine.capture.dmd_control --status
```

Copy the table into "Before" below.

**2. Re-create the mode (human).** In Intel Graphics Command Center → Display → Custom resolution:
- Delete the CVT-RB 912 × 1140 mode.
- Re-add **912 × 1140 @ 60 Hz** with **standard CVT** timing (not reduced blanking).
- Re-apply the EVM display at desktop offset (1920, 0).

**3. Set the bit depth to 24.** Use one of these:
- **LightCrafter GUI:** Video Mode tab → Parallel RGB bit depth = **24** → close the GUI.
- **The tool:** `%PY% -m engine.capture.dmd_control --bitdepth 24 --yes`. This re-asserts source 0 at 24-bit.

**4. Record the AFTER timing.** Expect:
- **Detected** at **912x1140**
- larger vertical blanking and first line than before
- horizontal blanking no longer exactly 160 px

```
%PY% -m engine.capture.dmd_control --status
```

**5. Recovery test.** This deliberately causes the failure, then recovers without a power cycle:

```
%PY% -m engine.capture.dmd_control --source 1 --yes     (blue lines / test pattern should appear)
%PY% -m engine.capture.dmd_control --relock --yes       (expect "Detected", exit code 0)
```

If `--relock` ends in **Lock Failed**, the output shows the resolution, vertical blanking and first line the controller saw. Record them below. Power-cycling the EVM is still the fallback.

**6. Scan-path check.** Run the engine's `check_signal` (a separate process) and confirm gc_conf is back to about 55 with the vertical-stripe preset.

## Results (fill in on the bench)

| Field | Before (CVT-RB) | After (standard CVT, 24-bit) |
|---|---|---|
| signal status | | |
| resolution | | |
| vertical frequency (Hz) | | |
| total pixels / line | | |
| horizontal blanking (px) | | |
| total lines / frame | | |
| **vertical blanking (lines)** | | |
| **first line (VSYNC + back porch)** | | |
| implied pixel clock (MHz) | | |
| `--source 1` → `--relock` recovers without a power cycle? | | |

## Notes

- **First line.** Table 2-5 defines "First Line" as the beginning of the active line in the frame, so it equals VSYNC width plus back porch. The guide gives no separate VSYNC width, so the tool reports this sum. It's the number that should grow when CVT-RB is replaced by CVT.
- **Units.** Table 2-5's unit wording for pixel clock and horizontal frequency is ambiguous, so those are printed raw. The tool also prints the pixel clock implied by `total_px × total_lines × vfreq` as a cross-check. The raw vertical frequency is in 1/100 Hz (6000 = 60.00 Hz).
- **Pattern mode (build step 10).** Once the relock is reliable, pattern-sequence mode via the video port becomes a candidate: it's pixel-accurate and has no degamma. It must remain a separate, explicitly entered mode.
