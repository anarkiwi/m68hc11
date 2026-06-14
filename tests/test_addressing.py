"""Addressing-mode resolution, prefix decoding, and per-mode cycle counts."""

import m68hc11


def make(prog, base=0x0100, **regs):
    c = m68hc11.HC11()
    c.load(bytes(prog), base)
    c.set_regs(pc=base, **regs)
    return c


# --------------------------------------------------------------------------- #
# Effective-address resolution per mode (using LDAA variants)
# --------------------------------------------------------------------------- #
def test_immediate():
    c = make([0x86, 0x42])  # LDAA #$42
    st = c.step()
    assert c.a == 0x42 and st.ea is None and st.cycles == 2
    assert st.mnemonic == "LDAA #$42"


def test_direct():
    c = make([0x96, 0x40])  # LDAA $40
    c.write8(0x0040, 0x99)
    st = c.step()
    assert c.a == 0x99 and st.ea == 0x0040 and st.cycles == 3


def test_extended():
    c = make([0xB6, 0x12, 0x34])  # LDAA $1234
    c.write8(0x1234, 0x77)
    st = c.step()
    assert c.a == 0x77 and st.ea == 0x1234 and st.cycles == 4


def test_indexed_x():
    c = make([0xA6, 0x05], x=0x2000)  # LDAA $05,X
    c.write8(0x2005, 0x55)
    st = c.step()
    assert c.a == 0x55 and st.ea == 0x2005 and st.cycles == 4
    assert st.mnemonic == "LDAA $05,X"


def test_indexed_x_offset_unsigned():
    c = make([0xA6, 0xFF], x=0x2000)  # offset is unsigned 0..255
    c.write8(0x20FF, 0xEE)
    c.step()
    assert c.a == 0xEE


def test_indexed_y():
    c = make([0x18, 0xA6, 0x05], y=0x3000)  # LDAA $05,Y
    c.write8(0x3005, 0x44)
    st = c.step()
    assert c.a == 0x44 and st.ea == 0x3005 and st.cycles == 5
    assert st.mnemonic == "LDAA $05,Y"


def test_relative_forward_and_back():
    c = make([0x20, 0x7F])  # BRA +0x7F from 0x0102 -> 0x0181
    st = c.step()
    assert c.pc == 0x0181 and st.cycles == 3
    c = make([0x20, 0xFE])  # BRA -2 -> back to itself (0x0100)
    c.step()
    assert c.pc == 0x0100


# --------------------------------------------------------------------------- #
# Prefix pages: $18, $1A, $CD
# --------------------------------------------------------------------------- #
def test_page2_cpy_immediate():
    c = make([0x18, 0x8C, 0x12, 0x34], y=0x1234)  # CPY #$1234
    st = c.step()
    assert c.z is True and st.cycles == 5
    assert st.mnemonic == "CPY #$1234"
    assert st.opcode == b"\x18\x8c\x12\x34"


def test_page3_cpd_immediate():
    c = make([0x1A, 0x83, 0x00, 0x10], d=0x0010)  # CPD #$0010
    st = c.step()
    assert c.z is True and st.cycles == 5
    assert st.mnemonic == "CPD #$0010"


def test_page3_ldy_indexed_x():
    # $1A $EE = LDY indexed by X
    c = make([0x1A, 0xEE, 0x02], x=0x4000)
    c.write16(0x4002, 0xCAFE)
    st = c.step()
    assert c.y == 0xCAFE and st.cycles == 6
    assert st.mnemonic == "LDY $02,X"


def test_page4_cpd_indexed_y():
    # $CD $A3 = CPD indexed by Y
    c = make([0xCD, 0xA3, 0x02], y=0x4000, d=0x1111)
    c.write16(0x4002, 0x1111)
    st = c.step()
    assert c.z is True and st.cycles == 7
    assert st.mnemonic == "CPD $02,Y"


def test_page4_ldx_indexed_y():
    # $CD $EE = LDX indexed by Y
    c = make([0xCD, 0xEE, 0x00], y=0x5000)
    c.write16(0x5000, 0xBEEF)
    st = c.step()
    assert c.x == 0xBEEF and st.cycles == 6
    assert st.mnemonic == "LDX $00,Y"


def test_page4_cpx_indexed_y():
    c = make([0xCD, 0xAC, 0x00], y=0x5000, x=0x0042)
    c.write16(0x5000, 0x0042)
    st = c.step()
    assert c.z is True and st.cycles == 7


# --------------------------------------------------------------------------- #
# A representative sweep of cycle counts straight from the opcode map
# --------------------------------------------------------------------------- #
CYCLE_CASES = [
    # (program bytes, expected cycles, label)
    ([0x01], 2, "NOP"),
    ([0x3D], 10, "MUL"),
    ([0x02], 41, "IDIV"),
    ([0x03], 41, "FDIV"),
    ([0x3B], 12, "RTI"),
    ([0x39], 5, "RTS"),
    ([0x8D, 0x00], 6, "BSR"),
    ([0x9D, 0x00], 5, "JSR dir"),
    ([0xBD, 0x00, 0x00], 6, "JSR ext"),
    ([0xAD, 0x00], 6, "JSR idx"),
    ([0x7E, 0x00, 0x00], 3, "JMP ext"),
    ([0x6E, 0x00], 3, "JMP idx"),
    ([0x86, 0x00], 2, "LDAA imm"),
    ([0xB6, 0x00, 0x00], 4, "LDAA ext"),
    ([0xCC, 0x00, 0x00], 3, "LDD imm"),
    ([0xFC, 0x00, 0x00], 5, "LDD ext"),
    ([0xC3, 0x00, 0x00], 4, "ADDD imm"),
    ([0xF3, 0x00, 0x00], 6, "ADDD ext"),
    ([0x70, 0x00, 0x00], 6, "NEG ext"),
    ([0x14, 0x00, 0x00], 6, "BSET dir"),
    ([0x1C, 0x00, 0x00], 7, "BSET idx,X"),
    ([0x12, 0x00, 0x00, 0x00], 6, "BRSET dir"),
    ([0x1E, 0x00, 0x00, 0x00], 7, "BRSET idx,X"),
    # prefixed forms (+1 cycle vs base)
    ([0x18, 0x08], 4, "INY"),
    ([0x18, 0xCE, 0x00, 0x00], 4, "LDY imm"),
    ([0x18, 0x3C], 5, "PSHY"),
    ([0x18, 0x38], 6, "PULY"),
    ([0x18, 0xA6, 0x00], 5, "LDAA idx,Y"),
    ([0x18, 0x60, 0x00], 7, "NEG idx,Y"),
    ([0x18, 0x1C, 0x00, 0x00], 8, "BSET idx,Y"),
    ([0x18, 0x1E, 0x00, 0x00, 0x00], 8, "BRSET idx,Y"),
]


def test_cycle_counts():
    for prog, cyc, label in CYCLE_CASES:
        c = make(prog)
        st = c.step()
        assert st.cycles == cyc, f"{label}: got {st.cycles}, want {cyc}"


def test_step_opcode_bytes_capture_full_instruction():
    c = make([0xCD, 0xEE, 0x07], y=0x100)
    st = c.step()
    assert st.opcode == b"\xcd\xee\x07"
    assert st.pc == 0x0100
