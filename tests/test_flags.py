"""Spec section 8.1 -- hand-checked flag / arithmetic self-test.

Exercises the fiddly condition-code cases: MUL C-bit, ASLD/LSRD, DAA,
ADDD/SUBD overflow & carry, NEG/COM carry, signed branches, ABX/XGDX.
Every expected value here is computed by hand from the M68HC11 reference.
"""

import m68hc11


def run1(prog, base=0x0100, **regs):
    """Assemble nothing -- just load ``prog`` at ``base`` and single-step once."""
    c = m68hc11.HC11()
    c.load(bytes(prog), base)
    c.set_regs(pc=base, **regs)
    st = c.step()
    return c, st


# --------------------------------------------------------------------------- #
# MUL: C = bit 7 of the 16-bit product
# --------------------------------------------------------------------------- #
def test_mul_carry_set():
    # 0x12 * 0x34 = 0x03A8; low byte 0xA8 has bit7 set -> C = 1
    c, st = run1([0x3D], a=0x12, b=0x34)
    assert c.d == 0x03A8
    assert c.c is True
    assert st.cycles == 10


def test_mul_carry_clear():
    # 0x10 * 0x10 = 0x0100; low byte 0x00 bit7 clear -> C = 0
    c, _ = run1([0x3D], a=0x10, b=0x10)
    assert c.d == 0x0100
    assert c.c is False


def test_mul_zero():
    c, _ = run1([0x3D], a=0x00, b=0xFF)
    assert c.d == 0
    assert c.c is False  # bit 7 of 0 is 0


# --------------------------------------------------------------------------- #
# ASLD / LSRD (16-bit shifts): C from shifted-out bit, V = N ^ C
# --------------------------------------------------------------------------- #
def test_asld():
    # 0x8000 << 1 = 0x0000, carry out = 1, N = 0 -> V = N^C = 1
    c, _ = run1([0x05], d=0x8000)
    assert c.d == 0x0000
    assert c.c is True and c.z is True and c.n is False and c.v is True


def test_asld_into_sign():
    # 0x4000 << 1 = 0x8000, C = 0, N = 1 -> V = 1
    c, _ = run1([0x05], d=0x4000)
    assert c.d == 0x8000
    assert c.c is False and c.n is True and c.v is True


def test_lsrd():
    # 0x0003 >> 1 = 0x0001, C = 1, N = 0 (always), V = C = 1
    c, _ = run1([0x04], d=0x0003)
    assert c.d == 0x0001
    assert c.c is True and c.n is False and c.v is True and c.z is False


# --------------------------------------------------------------------------- #
# DAA
# --------------------------------------------------------------------------- #
def test_daa_simple():
    # A = 0x15 + 0x27 = 0x3C ; DAA -> 0x42
    c = m68hc11.HC11()
    c.load(b"\x8b\x27\x19", 0x100)  # ADDA #$27 ; DAA
    c.set_regs(a=0x15, pc=0x100)
    c.step()
    c.step()
    assert c.a == 0x42
    assert c.c is False


def test_daa_carry_out():
    # 0x99 + 0x01 = 0x9A ; H=0,C=0,hi=9,lo>9 -> +0x66 = 0x100 -> 0x00, C=1
    c = m68hc11.HC11()
    c.load(b"\x8b\x01\x19", 0x100)  # ADDA #$01 ; DAA
    c.set_regs(a=0x99, pc=0x100)
    c.step()
    c.step()
    assert c.a == 0x00
    assert c.c is True
    assert c.z is True


def test_daa_with_half_carry():
    # 0x28 + 0x19 (BCD) raw = 0x41, H set by the add (8+9>0xF). DAA -> 0x47.
    c = m68hc11.HC11()
    c.load(b"\x8b\x19\x19", 0x100)  # ADDA #$19 ; DAA
    c.set_regs(a=0x28, pc=0x100)
    c.step()
    assert c.h is True  # half carry from 0x8 + 0x9
    c.step()
    assert c.a == 0x47


# --------------------------------------------------------------------------- #
# ADDD / SUBD overflow & carry (16-bit)
# --------------------------------------------------------------------------- #
def test_addd_carry():
    # 0xFFFF + 0x0001 = 0x0000, carry out, no signed overflow
    c = m68hc11.HC11()
    c.load(b"\xc3\x00\x01", 0x100)  # ADDD #$0001
    c.set_regs(d=0xFFFF, pc=0x100)
    c.step()
    assert c.d == 0x0000
    assert c.c is True and c.z is True and c.v is False


def test_addd_signed_overflow():
    # 0x7FFF + 0x0001 = 0x8000: signed overflow (pos+pos->neg), no carry
    c = m68hc11.HC11()
    c.load(b"\xc3\x00\x01", 0x100)
    c.set_regs(d=0x7FFF, pc=0x100)
    c.step()
    assert c.d == 0x8000
    assert c.v is True and c.c is False and c.n is True


def test_subd_borrow():
    # 0x0000 - 0x0001 = 0xFFFF: borrow (C=1), N=1, no overflow
    c = m68hc11.HC11()
    c.load(b"\x83\x00\x01", 0x100)  # SUBD #$0001
    c.set_regs(d=0x0000, pc=0x100)
    c.step()
    assert c.d == 0xFFFF
    assert c.c is True and c.n is True and c.v is False


def test_subd_signed_overflow():
    # 0x8000 - 0x0001 = 0x7FFF: neg - pos -> pos = overflow
    c = m68hc11.HC11()
    c.load(b"\x83\x00\x01", 0x100)
    c.set_regs(d=0x8000, pc=0x100)
    c.step()
    assert c.d == 0x7FFF
    assert c.v is True and c.c is False and c.n is False


# --------------------------------------------------------------------------- #
# NEG / COM carry behaviour
# --------------------------------------------------------------------------- #
def test_nega_nonzero_sets_carry():
    # NEG of a nonzero value sets C (borrow); 0x01 -> 0xFF
    c, _ = run1([0x40], a=0x01)  # NEGA
    assert c.a == 0xFF
    assert c.c is True and c.n is True and c.v is False


def test_nega_zero_clears_carry():
    c, _ = run1([0x40], a=0x00)  # NEGA
    assert c.a == 0x00
    assert c.c is False and c.z is True


def test_nega_0x80_overflow():
    # 0x80 negates to itself -> V set, C set
    c, _ = run1([0x40], a=0x80)
    assert c.a == 0x80
    assert c.v is True and c.c is True and c.n is True


def test_coma_sets_carry():
    # COM always sets carry; 0x0F -> 0xF0
    c, _ = run1([0x43], a=0x0F)  # COMA
    assert c.a == 0xF0
    assert c.c is True and c.v is False and c.n is True


# --------------------------------------------------------------------------- #
# Signed branches
# --------------------------------------------------------------------------- #
def _branch_taken(opcode, **regs):
    """Run a single branch with offset +2 and report whether it was taken."""
    c = m68hc11.HC11()
    c.load(bytes([opcode, 0x02]), 0x100)  # B?? $0104 (PC after = 0x102, +2)
    c.set_regs(pc=0x100, **regs)
    c.step()
    return c.pc == 0x0104


def test_bgt_blt_bge_ble_signed():
    # A = 5, compare to 3: 5 > 3 (signed). Z=0, N^V=0.
    def after_cmp(a, imm):
        c = m68hc11.HC11()
        c.load(bytes([0x81, imm]), 0x100)  # CMPA #imm
        c.set_regs(a=a, pc=0x100)
        c.step()
        return c

    c = after_cmp(0x05, 0x03)
    assert c.z is False and (bool(c.n) == bool(c.v))  # BGT condition holds

    # -128 (0x80) vs +1: -128 < 1, with overflow making N^V true
    c = after_cmp(0x80, 0x01)
    assert c.v is True and (bool(c.n) != bool(c.v))  # BLT condition holds


def test_branch_dispatch():
    # BMI taken when N set: load 0x80 then BMI
    c = m68hc11.HC11()
    c.load(b"\x86\x80\x2b\x02", 0x100)  # LDAA #$80 ; BMI $0106
    c.set_regs(pc=0x100)
    c.step()  # LDAA sets N
    c.step()  # BMI
    assert c.pc == 0x0106

    # BCC not taken when C set
    assert _branch_taken(0x24, ccr=m68hc11._C) is False
    # BCC taken when C clear
    assert _branch_taken(0x24, ccr=0x00) is True


# --------------------------------------------------------------------------- #
# ABX / ABY (no flags) and XGDX / XGDY
# --------------------------------------------------------------------------- #
def test_abx_no_flags():
    c = m68hc11.HC11()
    c.load(b"\x3a", 0x100)  # ABX
    c.set_regs(x=0x1000, b=0xFF, ccr=0x00, pc=0x100)
    c.step()
    assert c.x == 0x10FF
    assert c.ccr == 0x00  # ABX affects no flags


def test_abx_wraps():
    c, _ = run1([0x3A], x=0xFFFF, b=0x02)  # ABX
    assert c.x == 0x0001


def test_xgdx():
    c = m68hc11.HC11()
    c.load(b"\x8f", 0x100)  # XGDX
    c.set_regs(d=0x1234, x=0xABCD, ccr=0x00, pc=0x100)
    c.step()
    assert c.d == 0xABCD and c.x == 0x1234
    assert c.ccr == 0x00  # no flags


def test_xgdy():
    c = m68hc11.HC11()
    c.load(b"\x18\x8f", 0x100)  # XGDY
    c.set_regs(d=0x1234, y=0xABCD, pc=0x100)
    st = c.step()
    assert c.d == 0xABCD and c.y == 0x1234
    assert st.cycles == 4
