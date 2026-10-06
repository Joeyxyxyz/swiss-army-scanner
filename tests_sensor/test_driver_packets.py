"""
Cross-validate the DLPC350 wire format against pycrafter4500.

pycrafter4500 is the established, hardware-proven LightCrafter 4500
control library. Its `command()` builds the exact byte stream known to
work on real DLPC350 silicon. This test rebuilds the same commands with
our pure `build_packet()` and compares byte-for-byte (modulo the hidapi
report-id byte, which pycrafter omits because it writes raw pyusb
endpoints).

This is what caught the v0.4 CMD2/CMD3 byte-order swap.
"""
import sys; sys.path.insert(0, ".")

from swiss_army_sensor.drivers.dlp_driver import build_packet, parse_reply, CMD


def pycrafter_wire(mode, sequence_byte, com1, com2, data):
    """Reimplementation of pycrafter4500.dlpc350.command() byte layout
    (first packet only, single-packet commands), kept literal to the
    published source so divergence is loud."""
    buffer = []
    flagstring = 0xC0 if mode == "r" else 0x40
    payload_len = len(data) + 2
    buffer.append(flagstring)
    buffer.append(sequence_byte)
    buffer.append(payload_len & 0xFF)          # bits_to_bytes(..., reverse=True) -> LSB first
    buffer.append((payload_len >> 8) & 0xFF)
    buffer.append(com2)                        # CMD3 on the wire first
    buffer.append(com1)                        # then CMD2
    buffer.extend(data)
    while len(buffer) < 64:
        buffer.append(0x00)
    return bytes(buffer)


# (name, mode, CMD2, CMD3, data) — taken from pycrafter4500 call sites
CASES = [
    ("display_mode_pattern", "w", 0x1A, 0x1B, [0x01]),
    ("pattern_source_video", "w", 0x1A, 0x22, [0x00]),
    ("pattern_trigger_vsync", "w", 0x1A, 0x23, [0x00]),
    ("start_sequence", "w", 0x1A, 0x24, [0x02]),
    ("validation", "w", 0x1A, 0x1A, [0x00]),
    ("dmd_park", "w", 0x06, 0x09, [0x01]),
    ("main_status_read", "r", 0x1A, 0x0C, []),
    ("lut_entry", "w", 0x1A, 0x34, [0b000000, 0b00101000, 0x00]),
]


def test_wire_format_matches_pycrafter4500():
    for name, mode, cmd2, cmd3, data in CASES:
        seq = 7
        ours = build_packet((cmd2, cmd3), bytes(data),
                            read=(mode == "r"), reply=True, seq=seq)
        assert len(ours) == 1, f"{name}: unexpectedly multi-packet"
        ours_wire = ours[0][1:]  # strip hidapi report-id byte
        ref = pycrafter_wire(mode, seq, cmd2, cmd3, data)
        assert ours_wire == ref, (
            f"{name}: wire mismatch\n ours: {ours_wire[:12].hex()}\n  ref: {ref[:12].hex()}")
    print(f"[OK] {len(CASES)} commands byte-identical to pycrafter4500")


def test_multipacket_split():
    data = bytes(range(200))
    pkts = build_packet((0x1A, 0x34), data, seq=1)
    assert all(len(p) == 65 for p in pkts)
    assert len(pkts) == 1 + ((200 - 58) + 63) // 64
    # payload reassembles
    body = pkts[0][7:] + b"".join(p[1:] for p in pkts[1:])
    assert body[:200] == data
    print(f"[OK] 200-byte payload split into {len(pkts)} packets, reassembles")


def test_reply_parsing():
    # device reply: flags, seq, len LSB/MSB, data...
    raw = bytes([0x40, 0x07, 0x01, 0x00, 0x12]) + b"\x00" * 59
    assert parse_reply(raw) == b"\x12"
    raw3 = bytes([0x40, 0x07, 0x03, 0x00, 0xAA, 0xBB, 0xCC]) + b"\x00" * 57
    assert parse_reply(raw3) == b"\xaa\xbb\xcc"
    assert parse_reply(b"") == b""
    print("[OK] reply data extracted from byte 4 with LE length (v0.4 read byte 0)")


def test_validation_bit_semantics():
    from swiss_army_sensor.drivers.dlp_driver import MockDLPDriver
    d = MockDLPDriver().open()
    v = d.validate()
    assert v["valid"] and not v["busy"]
    print("[OK] validate() interprets Table 2-52 bits (bit7=busy)")


if __name__ == "__main__":
    test_wire_format_matches_pycrafter4500()
    test_multipacket_split()
    test_reply_parsing()
    test_validation_bit_semantics()
