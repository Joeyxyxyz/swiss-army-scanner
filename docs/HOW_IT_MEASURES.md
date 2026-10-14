# How it measures

One page for an engineer who knows other inspection systems, such as a laser line sensor over a moving part, and has not seen this one. Details: the [README](../README.md).

## The method

The projector draws known stripes on the part; the camera, looking from the side, sees where each stripe lands; a stripe that lands shifted means the surface is higher or lower there.

This is triangulation, as in a laser line sensor. The stripes come as a short series of patterns, coarse to fine. Together they give every projector column its own code, so each camera pixel can tell which column lit it.

## Compared with a laser line sensor

| | Laser line sensor | This scanner |
|---|---|---|
| Light | one line | many coded stripes at once, over the whole view |
| Motion | the part or the sensor moves | the part stands still |
| Result | one height profile per capture, under the line | a height for every camera pixel that read the stripes |
| Sections | where the line ran | in any direction, afterwards, from the stored scan |

## What it needs

- **A calibration** with a printed board, shown to the rig in 12 to 20 positions. Without it there are no mm.
- **A fixed camera and projector:** position, zoom and focus as at calibration.
- **The part in view of both.**
- **A short moment without movement:** the patterns are shown one after another. A normal scan takes about 5 to 15 seconds on the README's bench rig.

## What limits it

- **Shadows.** Where only the projector or only the camera sees the surface, there is no reading.
- **Mirror-like and very dark surfaces** send the light past the camera, straight into it, or hardly back at all: the reading there is missing or poor.
- **Movement during the scan.** A part that moves between the patterns is read wrongly.
- **A rig bumped since calibration** gives wrong mm. The rig check (the HOME button) finds that: it projects a test pattern, compares the camera picture with the one stored after calibration, and answers OK, MOVED or UNKNOWN. MOVED also says what moved: the camera (the scene and the pattern on it moved together in its picture), the projector or its picture (the pattern moved over a still scene), or both. A small turn of the camera can be corrected in software; a projector move needs the projector put back, or a new calibration. Between checks, every scan that reads the stripes in both directions checks itself: when the two directions disagree far more than they did when the calibration was made, it warns that the calibration may not fit the rig any more.
- **Fine detail.** See *Limits & sizing* in the README.

## The words on the operator page

| Word | Meaning |
|---|---|
| **Program** | What is checked on one kind of part: where to look, which checks, which limits. In files: recipe, profile. |
| **Limit** | The allowed range of a value. In files: threshold, tolerance. |
| **Judgment** | PASS, FAIL or PULL. When a result cannot be trusted (too little was read, the rig moved), the part is pulled for inspection by a person, never passed. In files: verdict, disposition. |
| **Trigger** | The start of one inspection. START on the page is a manual trigger. |
| **Part ID** | The code read from the part; empty when none was read. |

**Event log:** one line per inspection that ran, with its values and limits, and links to a picture and, where the inspection has them, to height profiles and 3D data. It is kept as table files, one pair per month (the inspections, and their values), in the library's `events` folder.

**Bad parts:** a part judged FAIL or PULL is kept for tracking - copied to the bad-part output when one is set - and is never used as a reference: the scanner only ever compares a part with good parts. A START that the station refuses before it scans anything (its setup cannot be used, say none of the zones it inspects is switched on) pulls the part through the line's signals only: it has no line in the event log and no copy in the bad-part output.

## Two ways the projector shows stripes

- **Video.** The projector is a second screen of this PC. The software shows each pattern as a picture, and the camera takes its frames while the picture stands. Every normal scan works this way.
- **Pattern sequence (fast scan).** The projector plays the patterns by itself, a whole set in a fraction of a second, with a sync pulse for each; the camera takes one frame per pattern on that pulse (other pages here call it a hardware trigger).

A person switches the projector between these two display modes; in pattern sequence normal scans do not work. The fast scan is off unless enabled for the rig. Its first version reports how well the stripes were read: the camera pixels decoded, and whether every frame and sync pulse arrived. It gives no mm and no judgment yet: mm need a calibration made in the same display mode.
