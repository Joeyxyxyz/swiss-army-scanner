# swiss_army_sensor tests

The seed package's own tests. They import the package `__init__`, which loads the
DLPC350 USB-HID driver module (no device is opened; the mock driver is used). The
scanner engine refuses to scan once that module is loaded (rule 9.1), so these run
in their **own** pytest process, never together with `tests/`:

```bash
python -m pytest -q tests_sensor
```
