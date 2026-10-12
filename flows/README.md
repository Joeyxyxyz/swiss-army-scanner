# Flows (work states)

A flow is the **work states** of a station: the ordered steps and permissions that let the
line continue to the next thing. It is not the station configuration: the station
configuration (`stations/*.json`) holds the parameters and the profile library set; a flow
says what is claimed, checked and permitted, step by step. A station configuration names its
flow inline (`"flow": [...]`) or as a file relative to itself (`"flow":
"../flows/example_tube_line.json"`), so several stations (or several profile sets) can share
one.

```json
{
  "schema_version": 2,
  "name": "example_tube_line",
  "steps": [
    {"id": "loaded", "title": "Tube loaded?",
     "expects": {"part_present": true},
     "verify": {"recipe": "presence", "observe": {"part_present": "seat/presence"}},
     "permit": "LOAD_DONE",
     "on_mismatch": ["project_fault",
                     {"prompt": "no tube found: load or clear the sensor",
                      "when_dlp": {"part_present": false}},
                     {"prompt": "a tube is on the seat but the sensor says none: check the sensor",
                      "when_dlp": {"part_present": true}},
                     "hold"]},
    {"id": "aligned", "title": "Tube aligned?",
     "expects": {"part_present": true},
     "verify": {"recipe": "alignment", "observe": {"part_present": "seat/presence"}},
     "permit": "CLAMP",
     "on_mismatch": ["project_fault",
                     {"prompt": "no tube found: load or clear the sensor",
                      "when_dlp": {"part_present": false}},
                     {"prompt": "a tube is on the seat but the sensor says none: check the sensor",
                      "when_dlp": {"part_present": true}},
                     "hold"],
     "on_defect": ["project_fault", "prompt tube off its seat: re-seat it", "retry"], "retries": 2},
    {"id": "end_face", "title": "End face OK?",
     "expects": {"part_present": true},
     "verify": {"recipe": "end_face", "observe": {"part_present": "seat/presence"}},
     "permit": "TRANSFER",
     "on_mismatch": ["project_fault",
                     {"prompt": "no tube found: load or clear the sensor",
                      "when_dlp": {"part_present": false}},
                     {"prompt": "a tube is on the seat but the sensor says none: check the sensor",
                      "when_dlp": {"part_present": true}},
                     "hold"],
     "on_defect": ["project_fault", "kickout REJECT end face not OK"]}
  ]
}
```

- `expects`: what the line's own sensors claim at this step. The DLP confirms or corrects
  it: is that part really there, or not? Every step of the example asks it, so a tube that
  went missing at *Tube aligned?* or *End face OK?* is a missing tube (CORRECTED: the sensor
  still says present), never a "dent" and never a REJECT.
- `verify`: the DLP check. `recipe` is a **key** of the active profile's `recipes`, so the
  same flow checks a long tube against the long-tube recipes and a short tube against the
  short-tube ones. `observe` names, for **every** expected signal, the check whose answer IS
  that signal (`"<zone>/<check>"`); there is no default - a signal no check was named for is
  refused, never "confirmed" by whatever presence check the recipe happens to have.
- A sensor mismatch has two directions (the sensor says present and the seat is empty, or
  the reverse): give each its own prompt with `when_dlp`, as above.
- `permit`: the permission (bus output `PERMIT_<NAME>`), granted only when the work state is
  CONFIRMED - and every required work state before it is - never on a doubt.
- At least one work state must be `required` (the default). A work state left in DOUBT or
  CORRECTED is never passed over, required or not: it is verified again (`hold` / `retry`)
  or the cycle ends PULL; `on_doubt` takes no `goto`.

Every key, the outcomes (CONFIRMED / CORRECTED / NOT_MET / DEFECT / DOUBT), the actions
(`project_fault`, `prompt`, `hold`, `retry`, `kickout`, `goto`) and the result document are
in `stations/README.md`; `scan flow-show <station config>` prints a flow as the engine reads
it and lists every problem. A flow file may also set `defaults` (`retries`, `timeout_s`,
`trigger`) for its steps.
