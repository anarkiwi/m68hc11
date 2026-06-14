"""Memory model: flat 64 KB, fill, load, big-endian 16-bit access, I/O hooks."""

import m68hc11


def test_ram_fill_default_zero():
    c = m68hc11.HC11()
    assert c.read8(0x0000) == 0x00
    assert c.read8(0xFFFF) == 0x00


def test_ram_fill_ff():
    c = m68hc11.HC11(ram_fill=0xFF)
    assert c.read8(0x1234) == 0xFF


def test_load_blob():
    c = m68hc11.HC11()
    c.load(b"\xde\xad\xbe\xef", 0x4000)
    assert c.read(0x4000, 4) == b"\xde\xad\xbe\xef"


def test_load_wraps_at_top():
    c = m68hc11.HC11()
    c.load(b"\x11\x22\x33", 0xFFFE)  # wraps 0xFFFE,0xFFFF,0x0000
    assert c.read8(0xFFFE) == 0x11
    assert c.read8(0xFFFF) == 0x22
    assert c.read8(0x0000) == 0x33


def test_big_endian_16():
    c = m68hc11.HC11()
    c.write16(0x2000, 0x1234)
    assert c.read8(0x2000) == 0x12  # high byte at lower address
    assert c.read8(0x2001) == 0x34
    assert c.read16(0x2000) == 0x1234


def test_write_int_vs_bytes():
    c = m68hc11.HC11()
    c.write(0x100, 0xAB)  # single int
    assert c.read8(0x100) == 0xAB
    c.write(0x200, b"\x01\x02\x03")  # bytes
    assert c.read(0x200, 3) == b"\x01\x02\x03"


def test_address_wraps_on_access():
    c = m68hc11.HC11()
    c.write8(0x1_0000, 0x42)  # wraps to 0x0000
    assert c.read8(0x0000) == 0x42


def test_self_modifying_code_allowed():
    # No region is read-only: code can rewrite itself.
    c = m68hc11.HC11()
    # STAA overwrites the immediate operand of the following LDAA instruction.
    c.load(b"\xb7\x01\x04\x86\x00", 0x100)  # STAA $0104 ; LDAA #$00
    c.set_regs(pc=0x100, a=0x99)
    c.step()  # STAA $0104 writes 0x99 over the LDAA immediate operand at 0x0104
    assert c.read8(0x0104) == 0x99
    c.step()  # LDAA #$99 (operand was just patched)
    assert c.a == 0x99


# --------------------------------------------------------------------------- #
# I/O hooks
# --------------------------------------------------------------------------- #
def test_read_hook_overrides_memory():
    c = m68hc11.HC11()
    c.write8(0x1003, 0x00)  # underlying RAM
    c.on_read(0x1000, 0x100F, lambda addr: (addr & 0xFF) ^ 0xA5)
    assert c.read8(0x1003) == (0x03 ^ 0xA5)
    assert c.read8(0x2000) == 0x00  # outside the hook range


def test_write_hook_intercepts():
    log = []
    c = m68hc11.HC11()
    c.on_write(0x1000, 0x1000, lambda addr, val: log.append((addr, val)))
    c.write8(0x1000, 0x5A)
    assert log == [(0x1000, 0x5A)]
    assert c.read8(0x1000) == 0x00  # hook swallowed the write; RAM untouched


def test_hook_used_during_execution():
    # A LDAA from a hooked "register" returns the modelled value.
    state = {"reg": 0x37}
    c = m68hc11.HC11()
    c.on_read(0x1004, 0x1004, lambda addr: state["reg"])
    c.load(b"\xb6\x10\x04", 0x0100)  # LDAA $1004
    c.set_regs(pc=0x0100)
    c.step()
    assert c.a == 0x37


def test_write_hook_during_execution():
    captured = []
    c = m68hc11.HC11()
    c.on_write(0x1008, 0x1008, lambda addr, val: captured.append(val))
    c.load(b"\x86\xc3\xb7\x10\x08", 0x0100)  # LDAA #$C3 ; STAA $1008
    c.set_regs(pc=0x0100)
    c.step()
    c.step()
    assert captured == [0xC3]


def test_open_bus_stub():
    # Unmodelled peripheral returns a fixed "open bus" value.
    c = m68hc11.HC11()
    c.on_read(0x4000, 0x5FFF, lambda addr: 0xFF)
    assert c.read8(0x4123) == 0xFF
    assert c.read8(0x5FFF) == 0xFF
