"""Spec sections 8.2 and 8.3.

* ``call()`` runs a self-contained subroutine that reads an input from a
  register and/or memory and returns its result in D / X / memory, with
  ``cycles`` reported -- using only the public API.
* Cross-validation style checks: small hand-assembled routines whose final
  registers, CCR, and *exact* cycle counts are computed by hand from the
  M68HC11 reference manual.  (Diff these against ``sim/m68hc11`` or MAME.)
"""

import m68hc11


# --------------------------------------------------------------------------- #
# 8.3 -- the call() harness
# --------------------------------------------------------------------------- #
def test_call_reads_register_returns_in_d():
    # subroutine: D = A * B (MUL), then RTS.  Inputs come in via call(a=, b=).
    c = m68hc11.HC11()
    c.load(b"\x3d\x39", 0x2000)  # MUL ; RTS
    st = c.call(0x2000, a=7, b=6)
    assert st.d == 42
    assert st.pc == 0xFFFE  # popped the sentinel
    assert st.cycles == 10 + 5  # MUL + RTS


def test_call_reads_memory_returns_in_memory():
    # subroutine doubles the byte at $00C0 and writes it back; result also in A.
    #   LDAA $C0 ; ASLA ; STAA $C0 ; RTS
    c = m68hc11.HC11()
    c.load(b"\x96\xc0\x48\x97\xc0\x39", 0x2000)
    c.write8(0x00C0, 0x09)
    st = c.call(0x2000)
    assert c.read8(0x00C0) == 0x12
    assert st.a == 0x12
    # cycles: LDAA dir(3) + ASLA(2) + STAA dir(3) + RTS(5) = 13
    assert st.cycles == 13


def test_call_returns_result_in_x():
    # subroutine: XGDX so that the D passed in comes back in X.
    c = m68hc11.HC11()
    c.load(b"\x8f\x39", 0x2000)  # XGDX ; RTS
    st = c.call(0x2000, d=0x1234)
    assert st.x == 0x1234


def test_call_isolated_cycle_count():
    # cycles reported by call() are only those elapsed during the call.
    c = m68hc11.HC11()
    c.cycles = 999  # pre-existing total
    c.load(b"\x01\x39", 0x2000)  # NOP ; RTS
    st = c.call(0x2000)
    assert st.cycles == 2 + 5
    assert c.cycles == 999 + 7  # cumulative total still advanced


def test_call_nested_subroutines():
    # outer calls inner; inner adds 1 to B; both return cleanly via the sentinel.
    #   outer @ 0x2000: BSR inner ; INCB ; RTS
    #   inner @ 0x2010: INCB ; RTS
    c = m68hc11.HC11()
    c.load(b"\x8d\x0e\x5c\x39", 0x2000)  # BSR +0x0E (->0x2010) ; INCB ; RTS
    c.load(b"\x5c\x39", 0x2010)  # INCB ; RTS
    st = c.call(0x2000, b=0x40)
    assert st.b == 0x42  # incremented twice
    assert st.pc == 0xFFFE


# --------------------------------------------------------------------------- #
# 8.2 -- cross-validation routines (hand-computed final state + cycles)
# --------------------------------------------------------------------------- #
def test_xval_straightline_aba():
    #   LDAA #$05  (2)
    #   LDAB #$03  (2)
    #   ABA        (2)
    #   RTS        (5)
    c = m68hc11.HC11()
    c.load(b"\x86\x05\xc6\x03\x1b\x39", 0x1000)
    st = c.call(0x1000)
    assert st.a == 0x08 and st.b == 0x03 and st.d == 0x0803
    assert st.cycles == 11
    assert st.n is False and st.z is False and st.v is False and st.c is False


def test_xval_sum_loop():
    # Sum `count` bytes at X into B (mod 256).
    #   CLRB           (2)        5F
    # loop:
    #   ADDB 0,X       (4)        EB 00
    #   INX            (3)        08
    #   DECA           (2)        4A
    #   BNE loop       (3)        26 FA
    #   RTS            (5)        39
    c = m68hc11.HC11()
    c.load(b"\x5f\xeb\x00\x08\x4a\x26\xfa\x39", 0x1000)
    data = bytes([10, 20, 30, 40])
    c.load(data, 0x0050)
    st = c.call(0x1000, a=len(data), x=0x0050)
    assert st.b == 100  # 10+20+30+40
    assert st.x == 0x0054  # advanced past the data
    # cycles: CLRB(2) + 4*(4+3+2+3) + RTS(5) = 2 + 48 + 5 = 55
    assert st.cycles == 55


def test_xval_16bit_add_via_addd():
    #   LDD $40    (4)   DC 40
    #   ADDD $42   (5)   D3 42
    #   STD $44    (4)   DD 44
    #   RTS        (5)
    c = m68hc11.HC11()
    c.load(b"\xdc\x40\xd3\x42\xdd\x44\x39", 0x1000)
    c.write16(0x0040, 0x1234)
    c.write16(0x0042, 0x1111)
    st = c.call(0x1000)
    assert c.read16(0x0044) == 0x2345
    assert st.d == 0x2345
    assert st.cycles == 18


def test_xval_countdown_with_carry_chain():
    # 16-bit decrement of a memory word until zero, counting iterations in B.
    # DEX must be the last flag-setter before BNE so the branch follows X, not B.
    #   LDX $40       (4)  DE 40
    # loop:
    #   INCB          (2)  5C
    #   DEX           (3)  09
    #   BNE loop      (3)  26 FC
    #   STX $42       (4)  DF 42
    #   RTS           (5)
    c = m68hc11.HC11()
    c.load(b"\xde\x40\x5c\x09\x26\xfc\xdf\x42\x39", 0x1000)
    c.write16(0x0040, 0x0003)
    st = c.call(0x1000, b=0)
    assert st.x == 0x0000
    assert st.b == 3  # three iterations
    assert c.read16(0x0042) == 0x0000
    # cycles: LDX(4) + 3*(INCB 2 + DEX 3 + BNE 3) + STX(4) + RTS(5) = 4+24+4+5 = 37
    assert st.cycles == 37


def test_call_respects_max_steps_on_runaway():
    # An infinite loop returns after max_steps without hanging.
    c = m68hc11.HC11()
    c.load(b"\x20\xfe", 0x1000)  # BRA * (self)
    st = c.call(0x1000, max_steps=100)
    assert st.pc == 0x1000  # never reached the sentinel
