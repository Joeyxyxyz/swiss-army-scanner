# Flash / hardware-trigger pattern mode (fast path)

**Status:** the tools exist and are mock-tested. **Nothing has run on the real rig yet.** The HDMI video path (about 15 s per scan) is still the scan path and still the default.

This is the separate, explicitly entered, guarded mode from ENGINE_ARCHITECTURE §9 step 10. The engine **never** switches the DLPC350 into pattern mode. A human does that in the TI LightCrafter 4500 GUI.

**For the no-cable fast scan** (1-bit patterns from one flash image on the internal timer, free-running camera, soft sync), use `docs/FLASH_FASTSCAN_4500.md` and `tools/make_flash_pack.py --fastscan` instead. That doc also has the full 4500 MKII GUI procedure: backup through the GUI's firmware link, firmware build and upload, sequence settings, return to video mode and rollback.

## What changes

| | HDMI video mode (today) | Pattern mode (this doc) |
|---|---|---|
| Pattern source | desktop window over HDMI, one image per 60 Hz frame | 24-bit images in DLPC350 flash, or bit-planes of 912x1140 HDMI frames (video port) |
| Sync | software: hold, flush, then grab | TRIG_OUT_1 goes to camera Line0. One hardware-triggered frame per pattern. |
| Per pattern | about 0.35 s hold plus averaging | 4.5 to 8.8 ms |
| One axis (16 patterns) | about 15 s | about 0.14 s (flash, uniform timing) |

The DMD can run 1-bit patterns at kHz rates, but the camera is the limit. A full-frame Blackfly S Mono8 tops out at about 226 fps, so the pattern period stays at 4.5 ms or more. In practice the rate is about 110 to 200 patterns per second. Averaging is gone: each pattern is one exposure. To get SNR back, repeat the sequence and average in software. That is not built yet.

## Files

- `tools/make_flash_pack.py` builds the pack: BMPs, `manifest.json` and `procedure.txt`. It writes to `<library>\output\flashpacks\<ts>_<name>\` (or under `--out`) and never overwrites.
- `engine/capture/patternmode.py` has `TriggeredSequenceCapture`, which is guarded. It grabs exactly N frames, detects missing frames and returns them in HybridBasis order. The bench-only PySpin trigger setup is in the same file.
- `tests/test_patternmode_mock.py` holds the mock proof.

## 1. Make a pack

```
python tools/make_flash_pack.py                  # preset cell_wall_2p7ft_vstripe_fullrange, x axis, flash
python tools/make_flash_pack.py --axes y         # horizontal stripes (projector rows)
python tools/make_flash_pack.py --video-port     # HDMI Pattern Sequence frames
python tools/make_flash_pack.py --unpack-check <pack dir>
```

The default preset gives this layout: 16 patterns per axis, which are white, black, 10 gray planes and 4 fringes.

- **Flash image 0:** the 1-bit patterns go on G0..G7 and R0..R3. The first fringe is 8-bit and goes on B0..B7.
- **Flash image 1:** the other three fringes go on G, R and B.

This is the TI SDK packing order (`lcr4500.cpp` CreateFirmwareImages). The packed int's bits 0-7 go to the BMP green byte, bits 8-15 to red and bits 16-23 to blue. The GUI names the planes G0-G7, R0-R7 and B0-B7.

**Make one pack per axis.** The DLPC350 holds two 24-bit buffers, so 48 bit-planes can be preloaded. Loading another image from flash takes up to about 200 ms (DLPU011E §3.3.3). An x+y pack has 4 images, and each image would show for only about 115 ms, so the sequence stalls. The tool writes a warning in this case.

Timing rules are checked by `validate_timing` (DLPU010G §2.4.3.4.3):

- 1-bit exposure must be at least 235 µs. 8-bit exposure must be at least 8333 µs.
- Exposure must equal the period, or the period must be at least 230 µs longer than the exposure.
- In VSYNC modes, exposure must equal the period.

These are the suggested values:

- **Flash, uniform timing:** exposure 8333 µs, period 8833 µs. Set the camera ExposureTime to 8433 µs. The extra 100 µs sits inside the dark gap, so the camera integrates the whole PWM sequence of an 8-bit fringe.
- **Flash, variable timing (Mode 3):** 1-bit patterns use 4000/4500 µs.
- **Video port:** Mode 4 (VSYNC, variable exposure). Each frame carries at most 3 one-bit patterns (the camera is the limit) or 2 eight-bit patterns. That makes 6 frames, about 100 ms. The camera exposure is 150 µs shorter than the pattern exposure.

## 2. Flash it (human, TI GUI)

1. **Back up first.** Download the firmware `.bin` through the firmware link on the GUI's Image / Firmware tab, and keep the unit's original firmware image from its supplier if you have one. Note the firmware version shown in System Control. The GUI cannot read the flash back, so a bad flash is only recovered by uploading a known-good `.bin`. See `docs/FLASH_FASTSCAN_4500.md` §5.1 and §5.9.
2. Go to Image/Firmware, then Firmware Build:
   - Choose the base `.bin`.
   - Enter a tag of 32 characters or fewer.
   - Add `images/flash_00.bmp` and `flash_01.bmp` in order.
   - Set illumination to Monochrome, one LED.
   - Click Save Updates, which writes a `.bin`.

   If you keep the images that are already in the firmware, the flash indices shift. Add that offset to every index in `procedure.txt`.
3. Go to Firmware Upload and upload the new `.bin`. The flash is erased and rewritten, so do not interrupt it.
4. On the Pattern Sequence tab:
   - Set Source to Flash and Trigger to Internal.
   - Set the exposure and period to the values in `procedure.txt`.
   - Uncheck Repeat.
   - Add the patterns in manifest order: flash index, then the bit-plane(s) to click.
   - Click Validate.

`procedure.txt` in the pack lists every click in order.

## 3. Trigger wiring and camera

- **Projector:** in Trigger Controls, set Trigger Out 1 to normal polarity (active high) with 0 delay. TRIG_OUT_1 is high for the length of each pattern's exposure.
- **Wiring:** connect TRIG_OUT_1 to the camera's **Line0** (opto-isolated input on the GPIO cable) and connect the grounds. **Check the voltage levels before wiring.** This has not been verified on this rig. If the opto input does not register the EVM's trigger level, use a level shifter or the non-isolated Line3 (`source="Line3"`).
- **Camera:** set these nodes in SpinView, or with `patternmode.configure_pyspin_trigger(cam, exposure_us=...)`, in this order:
  1. TriggerMode Off
  2. TriggerSelector FrameStart
  3. TriggerSource Line0
  4. TriggerActivation RisingEdge
  5. TriggerOverlap ReadOut
  6. ExposureMode Timed
  7. ExposureTime
  8. StreamBufferHandlingMode **OldestFirst**, with the buffer count at N+8 or more
  9. TriggerMode On

  The free-run path uses NewestOnly. Here every frame counts, so the buffer must be OldestFirst.
- **Start order:** arm the camera first (BeginAcquisition), then press Play in the GUI.

## 4. Capture (engine, bench)

```python
from engine.capture import patternmode as pm
cam = pm.PySpinTriggeredCamera(pyspin_cam, exposure_us=manifest["timing"]["uniform"]["camera_exposure_us"])
cap = pm.TriggeredSequenceCapture(cam, manifest, acknowledge=pm.ACK_PHRASE).capture()
field = basis.decode(cap.frames_for("x"), geometry_id)
```

`TriggeredSequenceCapture` refuses to run in these cases:

- `acknowledge` is not exactly `ACK_PHRASE`.
- The DLPC350 USB driver or `hid` is loaded in the process.
- The camera has no `grab_triggered`, which means it is the free-run scan camera.

After capture it raises `MissingFrames` in these cases:

- The first trigger does not arrive within 30 s, or a later one within about 10 periods.
- Frame IDs jump.
- Trigger spacing is more than ±50 % off the manifest period.
- Extra frames arrive, for example because the sequence was left on repeat.

The frames go through the existing `HybridBasis.decode` unchanged. The mock test shows that the triggered decode is identical to the video-path decode.

## 5. Risks

- **Changing the display mode is the blue-lines failure.** Switching Video to Pattern Sequence and back makes the DLPC350 re-acquire its input. With the marginal CVT-RB 912x1140 HDMI timing, it can drop to the internal test pattern, and so far only a power cycle recovers it. Read `docs/HDMI_TIMING_FIX.md` first. Fix the HDMI timing with standard CVT before trying the video-port variant. The flash variant does not need HDMI during the sequence, but switching back to video mode can still hit the same failure.
- **Flashing can brick the EVM** if the upload is interrupted or the base `.bin` is wrong. Keep the backup.
- **Polarity:** the HDMI path lands inverted (`invert: true`) and pattern mode may not. The decode measures polarity from white and black on every capture, so either works.
- **Single exposure per pattern:** there is no averaging, so noise is about 2.8x the 8-average HDMI scan. Clipped-fringe handling (`sat_level`) still applies.
- **Not yet built:** `api.py` has no verb for pattern mode, there is no LED-current handling in pattern mode, TRIG_OUT_2 (sequence start) is not used for alignment, and there is no software averaging over repeats.
