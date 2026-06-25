"""HD6301 / HD6303 CPU-variant decode and execution.

The HD6303 is a CMOS 6801/6803 with six extra instructions (XGDX, SLP and the
AIM/OIM/EIM/TIM immediate-with-memory ops) and without the 68HC11-only
additions.  These tests pin the six extensions, the opcodes that must now decode
as illegal, and confirm the base 68HC11 decode is left untouched.
"""

import pytest

import m68hc11

# Bytes that are 68HC11 page-1 instructions (or prefixes) but are undefined on
# the HD6303 and must raise rather than silently mis-decode.
HC11_ONLY = [0x00, 0x02, 0x03, 0x12, 0x13, 0x14, 0x15, 0x1C, 0x1D, 0x1E, 0x1F, 0x8F, 0xCF]


def _cpu(program, base=0x0100):
    cpu = m68hc11.HD6303()
    cpu.load(bytes(program), base)
    cpu.pc = base
    return cpu


# --------------------------------------------------------------------------- #
# XGDX ($18) and SLP ($1A) -- not prefix bytes on the 6303
# --------------------------------------------------------------------------- #
def test_xgdx_18_executes_and_is_not_a_prefix():
    cpu = _cpu([0x18])  # XGDX
    cpu.d = 0x1234
    cpu.x = 0xABCD
    step = cpu.step()
    assert step.mnemonic == "XGDX"
    assert len(step.opcode) == 1
    assert step.cycles == 2
    assert cpu.d == 0xABCD and cpu.x == 0x1234


def test_slp_1a_sets_waiting():
    cpu = _cpu([0x1A])  # SLP
    step = cpu.disassemble(0x0100)
    assert step.mnemonic == "SLP" and len(step.opcode) == 1 and step.cycles == 4
    assert cpu.run(max_steps=5) == "wai"
    assert cpu.waiting is True


def test_18_and_1a_are_prefixes_on_plain_hc11():
    # Regression: the base 68HC11 still treats $18/$1A/$CD as prefix bytes.
    hc11 = m68hc11.HC11()
    hc11.load(bytes([0x18, 0x08]), 0x0100)  # $18 $08 -> INY
    assert hc11.disassemble(0x0100).mnemonic == "INY"
    hc11.load(bytes([0x1A, 0x83, 0x00, 0x00]), 0x0100)  # $1A $83 -> CPD #
    assert hc11.disassemble(0x0100).mnemonic.startswith("CPD")


# --------------------------------------------------------------------------- #
# AIM / OIM / EIM / TIM
# --------------------------------------------------------------------------- #
def test_aim_direct():
    cpu = _cpu([0x71, 0x0F, 0x40])  # AIM #$0F,$40
    cpu.write8(0x0040, 0xA5)
    step = cpu.step()
    assert step.mnemonic == "AIM #$0F,$40"
    assert len(step.opcode) == 3 and step.cycles == 6
    assert cpu.read8(0x0040) == 0x05  # A5 & 0F
    assert cpu.n is False and cpu.z is False and cpu.v is False


def test_oim_direct_sets_bits():
    cpu = _cpu([0x72, 0x81, 0x40])  # OIM #$81,$40
    cpu.write8(0x0040, 0x24)
    cpu.step()
    assert cpu.read8(0x0040) == 0xA5
    assert cpu.n is True and cpu.z is False


def test_eim_indexed():
    cpu = _cpu([0x65, 0xFF, 0x04])  # EIM #$FF,$04,X
    cpu.x = 0x0050
    cpu.write8(0x0054, 0x0F)
    step = cpu.step()
    assert step.mnemonic == "EIM #$FF,$04,X"
    assert len(step.opcode) == 3 and step.cycles == 7
    assert cpu.read8(0x0054) == 0xF0


def test_tim_direct_does_not_write_back():
    cpu = _cpu([0x7B, 0x0F, 0x40])  # TIM #$0F,$40
    cpu.write8(0x0040, 0xF0)
    step = cpu.step()
    assert step.mnemonic == "TIM #$0F,$40"
    assert step.cycles == 4
    assert cpu.read8(0x0040) == 0xF0  # unchanged
    assert cpu.z is True  # F0 & 0F == 0


def test_tim_indexed_cycles_and_flags():
    cpu = _cpu([0x6B, 0x80, 0x02])  # TIM #$80,$02,X
    cpu.x = 0x0050
    cpu.write8(0x0052, 0x80)
    step = cpu.step()
    assert step.cycles == 5
    assert cpu.read8(0x0052) == 0x80  # unchanged
    assert cpu.n is True and cpu.z is False


@pytest.mark.parametrize(
    "op,mode_bytes,mnem",
    [
        (0x61, [0x0F, 0x04], "AIM #$0F,$04,X"),
        (0x62, [0x0F, 0x04], "OIM #$0F,$04,X"),
        (0x6B, [0x0F, 0x04], "TIM #$0F,$04,X"),
        (0x71, [0x0F, 0x40], "AIM #$0F,$40"),
        (0x72, [0x0F, 0x40], "OIM #$0F,$40"),
        (0x75, [0x0F, 0x40], "EIM #$0F,$40"),
        (0x7B, [0x0F, 0x40], "TIM #$0F,$40"),
    ],
)
def test_immmem_disassembly(op, mode_bytes, mnem):
    cpu = _cpu([op] + mode_bytes)
    step = cpu.disassemble(0x0100)
    assert step.mnemonic == mnem
    assert len(step.opcode) == 3


# --------------------------------------------------------------------------- #
# Removed 68HC11-only opcodes now raise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", HC11_ONLY)
def test_hc11_only_opcodes_are_illegal_on_6303(op):
    cpu = _cpu([op, 0x00, 0x00, 0x00])
    with pytest.raises(m68hc11.IllegalOpcode):
        cpu.step()


def test_no_y_prefix_pages_on_6303():
    # $CD is a prefix on the HC11 but undefined on the 6303.
    cpu = _cpu([0xCD, 0xEE, 0x00])
    with pytest.raises(m68hc11.IllegalOpcode):
        cpu.step()


# --------------------------------------------------------------------------- #
# Integration: a tiny pointer-math routine using the 6303 extensions
# --------------------------------------------------------------------------- #
def test_call_routine_with_xgdx_and_aim():
    # Mask the low nibble off [$40], then return that base address in X.
    #   AIM #$F0,$40 ; LDX #$0040 ; RTS
    prog = [0x71, 0xF0, 0x40, 0xCE, 0x00, 0x40, 0x39]
    cpu = _cpu(prog, base=0x0200)
    cpu.write8(0x0040, 0x3C)
    st = cpu.call(0x0200)
    assert cpu.read8(0x0040) == 0x30
    assert st.x == 0x0040


def test_base_instructions_still_work_on_6303():
    # LDAA #$07 ; LDAB #$06 ; MUL ; RTS  -> D = 42  (shared 6801 base set)
    cpu = _cpu([0x86, 0x07, 0xC6, 0x06, 0x3D, 0x39], base=0x0300)
    st = cpu.call(0x0300)
    assert st.d == 42
