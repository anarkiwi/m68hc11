"""Broad behavioural coverage of loads/stores, ALU, shifts, stack, transfers."""

import m68hc11


def make(prog, base=0x0100, **regs):
    c = m68hc11.HC11()
    c.load(bytes(prog), base)
    c.set_regs(pc=base, **regs)
    return c


# --------------------------------------------------------------------------- #
# Loads / stores set N,Z and clear V; carry untouched
# --------------------------------------------------------------------------- #
def test_ldaa_flags():
    c = make([0x86, 0x00], ccr=m68hc11._C)  # LDAA #$00 with C preset
    c.step()
    assert c.z is True and c.n is False and c.v is False
    assert c.c is True  # loads don't touch carry

    c = make([0x86, 0x80])
    c.step()
    assert c.n is True and c.z is False


def test_ldd_std_roundtrip():
    c = make([0xCC, 0xBE, 0xEF, 0xFD, 0x20, 0x00])  # LDD #$BEEF ; STD $2000
    c.step()
    assert c.d == 0xBEEF
    c.step()
    assert c.read16(0x2000) == 0xBEEF
    assert c.n is True and c.z is False  # 0xBEEF has bit15 set


def test_staa_indexed():
    c = make([0xA7, 0x10], a=0x5A, x=0x0400)  # STAA $10,X
    c.step()
    assert c.read8(0x0410) == 0x5A


def test_ldx_stx():
    c = make([0xCE, 0x12, 0x34, 0xDF, 0x50])  # LDX #$1234 ; STX $50
    c.step()
    c.step()
    assert c.x == 0x1234 and c.read16(0x0050) == 0x1234


def test_lds_sts():
    c = make([0x8E, 0x01, 0xFF])  # LDS #$01FF
    c.step()
    assert c.sp == 0x01FF


def test_ldy_sty_extended():
    c = make([0x18, 0xCE, 0xAB, 0xCD, 0x18, 0xFF, 0x30, 0x00])  # LDY #$ABCD ; STY $3000
    c.step()
    c.step()
    assert c.y == 0xABCD and c.read16(0x3000) == 0xABCD


# --------------------------------------------------------------------------- #
# 8-bit ALU
# --------------------------------------------------------------------------- #
def test_adda_half_carry_and_carry():
    c = make([0x8B, 0x01], a=0xFF)  # ADDA #$01
    c.step()
    assert c.a == 0x00
    assert c.c is True and c.h is True and c.z is True


def test_adca_uses_carry():
    c = make([0x89, 0x00], a=0x00, ccr=m68hc11._C)  # ADCA #$00 with C=1
    c.step()
    assert c.a == 0x01 and c.c is False


def test_suba_borrow():
    c = make([0x80, 0x01], a=0x00)  # SUBA #$01
    c.step()
    assert c.a == 0xFF and c.c is True and c.n is True


def test_sbca_uses_carry():
    c = make([0x82, 0x00], a=0x05, ccr=m68hc11._C)  # SBCA #$00 with borrow
    c.step()
    assert c.a == 0x04


def test_cmpa_sets_flags_only():
    c = make([0x81, 0x05], a=0x05)  # CMPA #$05
    c.step()
    assert c.a == 0x05  # unchanged
    assert c.z is True and c.c is False


def test_logic_ops():
    c = make([0x84, 0x0F], a=0xFF)  # ANDA #$0F
    c.step()
    assert c.a == 0x0F and c.v is False

    c = make([0x8A, 0xF0], a=0x0F)  # ORAA #$F0
    c.step()
    assert c.a == 0xFF and c.n is True

    c = make([0x88, 0xFF], a=0xAA)  # EORA #$FF
    c.step()
    assert c.a == 0x55


def test_bita_is_nondestructive():
    c = make([0x85, 0x80], a=0x80)  # BITA #$80
    c.step()
    assert c.a == 0x80 and c.n is True and c.z is False


def test_aba_sba_cba():
    c = make([0x1B], a=0x20, b=0x22)  # ABA
    c.step()
    assert c.a == 0x42

    c = make([0x10], a=0x50, b=0x10)  # SBA
    c.step()
    assert c.a == 0x40

    c = make([0x11], a=0x33, b=0x33)  # CBA
    c.step()
    assert c.z is True


# --------------------------------------------------------------------------- #
# Increment / decrement / clr / tst on memory and registers
# --------------------------------------------------------------------------- #
def test_inc_dec_memory():
    c = make([0x7C, 0x20, 0x00])  # INC $2000
    c.write8(0x2000, 0x7F)
    c.step()
    assert c.read8(0x2000) == 0x80 and c.v is True and c.n is True

    c = make([0x7A, 0x20, 0x00])  # DEC $2000
    c.write8(0x2000, 0x80)
    c.step()
    assert c.read8(0x2000) == 0x7F and c.v is True


def test_clr_sets_zero_flag():
    c = make([0x4F], a=0xFF)  # CLRA
    c.step()
    assert c.a == 0x00 and c.z is True and c.c is False and c.n is False


def test_tst():
    c = make([0x4D], a=0x80)  # TSTA
    c.step()
    assert c.n is True and c.z is False and c.c is False and c.v is False


def test_inx_dex_only_affect_z():
    c = make([0x08], x=0xFFFF, ccr=m68hc11._C)  # INX -> 0x0000, Z set
    c.step()
    assert c.x == 0x0000 and c.z is True and c.c is True  # C preserved

    c = make([0x09], x=0x0001)  # DEX -> 0, Z set
    c.step()
    assert c.x == 0x0000 and c.z is True


# --------------------------------------------------------------------------- #
# Shifts / rotates on memory
# --------------------------------------------------------------------------- #
def test_asl_memory():
    c = make([0x78, 0x20, 0x00])  # ASL $2000
    c.write8(0x2000, 0x81)
    c.step()
    assert c.read8(0x2000) == 0x02 and c.c is True


def test_rol_ror_through_carry():
    c = make([0x49], a=0x80, ccr=m68hc11._C)  # ROLA: 0x80<<1 | C=1 -> 0x01, C=1
    c.step()
    assert c.a == 0x01 and c.c is True

    c = make([0x46], a=0x01, ccr=m68hc11._C)  # RORA: C->bit7, 0x01>>1 -> 0x80, C=1
    c.step()
    assert c.a == 0x80 and c.c is True and c.n is True


def test_lsr_clears_n():
    c = make([0x44], a=0xFF)  # LSRA
    c.step()
    assert c.a == 0x7F and c.n is False and c.c is True


def test_asr_keeps_sign():
    c = make([0x47], a=0x80)  # ASRA: keeps sign -> 0xC0
    c.step()
    assert c.a == 0xC0 and c.n is True and c.c is False


# --------------------------------------------------------------------------- #
# Stack: push/pull round trips and ordering
# --------------------------------------------------------------------------- #
def test_push_pull_byte():
    c = make([0x36, 0x32], a=0xAB, sp=0x01FF)  # PSHA ; PULA
    c.step()
    assert c.sp == 0x01FE and c.read8(0x01FF) == 0xAB
    c.a = 0x00
    c.step()
    assert c.a == 0xAB and c.sp == 0x01FF


def test_push_pull_x_big_endian_layout():
    c = make([0x3C], x=0x1234, sp=0x01FF)  # PSHX
    c.step()
    # low byte pushed first -> high byte ends up at the lower address
    assert c.read8(0x01FE) == 0x12  # high
    assert c.read8(0x01FF) == 0x34  # low
    assert c.sp == 0x01FD


def test_pshx_pulx_roundtrip():
    c = make([0x3C, 0x38], x=0xDEAD, sp=0x01FF)  # PSHX ; PULX
    c.step()
    c.x = 0
    c.step()
    assert c.x == 0xDEAD and c.sp == 0x01FF


def test_pshy_puly_roundtrip():
    c = make([0x18, 0x3C, 0x18, 0x38], y=0xBEEF, sp=0x01FF)  # PSHY ; PULY
    c.step()
    c.y = 0
    c.step()
    assert c.y == 0xBEEF and c.sp == 0x01FF


# --------------------------------------------------------------------------- #
# Transfers and stack-pointer index ops
# --------------------------------------------------------------------------- #
def test_tab_tba():
    c = make([0x16], a=0x80)  # TAB
    c.step()
    assert c.b == 0x80 and c.n is True and c.v is False

    c = make([0x17], b=0x00)  # TBA
    c.step()
    assert c.a == 0x00 and c.z is True


def test_tap_tpa():
    c = make([0x06], a=0x0D)  # TAP: copy A to CCR (C and N bits set: 0x0D)
    c.step()
    assert c.c is True and c.n is True

    c = make([0x07], ccr=0xD1)  # TPA
    c.step()
    assert c.a == 0xD1


def test_tap_cannot_set_x_irq():
    # X (XIRQ) mask currently clear -> TAP cannot set it back
    c = make([0x06], a=0xFF, ccr=0x00)  # all flags currently 0
    c.step()
    assert c.x_irq is False  # bit 6 stayed clear despite A having it set


def test_tsx_txs():
    c = make([0x30], sp=0x01FF)  # TSX -> X = SP+1
    c.step()
    assert c.x == 0x0200

    c = make([0x35], x=0x0200)  # TXS -> SP = X-1
    c.step()
    assert c.sp == 0x01FF


def test_ins_des():
    c = make([0x31], sp=0x0100)  # INS
    c.step()
    assert c.sp == 0x0101
    c = make([0x34], sp=0x0100)  # DES
    c.step()
    assert c.sp == 0x00FF


# --------------------------------------------------------------------------- #
# IDIV / FDIV
# --------------------------------------------------------------------------- #
def test_idiv():
    c = make([0x02], d=1000, x=3)  # IDIV: 1000 / 3 = 333 r 1
    c.step()
    assert c.x == 333 and c.d == 1
    assert c.c is False and c.v is False and c.z is False


def test_idiv_by_zero():
    c = make([0x02], d=1000, x=0)
    c.step()
    assert c.x == 0xFFFF and c.c is True


def test_fdiv():
    # FDIV: (D<<16)/X. D=1, X=2 -> 0x10000/2 = 0x8000 r 0
    c = make([0x03], d=1, x=2)
    c.step()
    assert c.x == 0x8000 and c.d == 0
    assert c.v is False and c.c is False


def test_fdiv_overflow():
    # D >= X -> overflow flagged
    c = make([0x03], d=5, x=2)
    c.step()
    assert c.v is True


# --------------------------------------------------------------------------- #
# Bit manipulation
# --------------------------------------------------------------------------- #
def test_bset_bclr_direct():
    c = make([0x14, 0x40, 0x81])  # BSET $40,#$81
    c.write8(0x0040, 0x00)
    c.step()
    assert c.read8(0x0040) == 0x81 and c.n is True

    c = make([0x15, 0x40, 0x01])  # BCLR $40,#$01
    c.write8(0x0040, 0xFF)
    c.step()
    assert c.read8(0x0040) == 0xFE


def test_bset_indexed():
    c = make([0x1C, 0x02, 0x0F], x=0x0400)  # BSET $02,X #$0F
    c.write8(0x0402, 0xF0)
    c.step()
    assert c.read8(0x0402) == 0xFF


def test_brset_taken_and_not():
    # BRSET $40 #$0F -> branch if all of low nibble set
    c = make([0x12, 0x40, 0x0F, 0x10])  # BRSET $40,#$0F,+0x10
    c.write8(0x0040, 0x0F)
    c.step()
    assert c.pc == 0x0100 + 4 + 0x10  # taken

    c = make([0x12, 0x40, 0x0F, 0x10])
    c.write8(0x0040, 0x0E)  # bit0 clear -> not all set
    c.step()
    assert c.pc == 0x0104  # falls through


def test_brclr_taken():
    c = make([0x13, 0x40, 0xF0, 0x10])  # BRCLR $40,#$F0,+0x10
    c.write8(0x0040, 0x0F)  # high nibble clear -> taken
    c.step()
    assert c.pc == 0x0100 + 4 + 0x10


# --------------------------------------------------------------------------- #
# JMP / JSR / RTS / BSR control flow
# --------------------------------------------------------------------------- #
def test_jsr_rts_roundtrip():
    # main: JSR $2000 ; (return here)   sub @ $2000: RTS
    c = make([0xBD, 0x20, 0x00], sp=0x01FF)
    c.load(b"\x39", 0x2000)
    c.step()  # JSR
    assert c.pc == 0x2000 and c.sp == 0x01FD
    assert c.read16(0x01FE) == 0x0103  # return address stacked big-endian
    c.step()  # RTS
    assert c.pc == 0x0103 and c.sp == 0x01FF


def test_bsr_rts():
    c = make([0x8D, 0x10], sp=0x01FF)  # BSR +0x10 -> $0112
    c.load(b"\x39", 0x0112)
    c.step()
    assert c.pc == 0x0112
    c.step()
    assert c.pc == 0x0102


def test_jmp_indexed():
    c = make([0x6E, 0x05], x=0x3000)  # JMP $05,X
    c.step()
    assert c.pc == 0x3005


# --------------------------------------------------------------------------- #
# SWI / RTI context round trip
# --------------------------------------------------------------------------- #
def test_swi_rti_roundtrip():
    c = make([0x3F], a=0x11, b=0x22, x=0x3344, y=0x5566, sp=0x01FF)  # SWI
    c.write16(0xFFF6, 0x4000)  # SWI vector
    c.load(b"\x3b", 0x4000)  # RTI at handler
    c.set_regs(ccr=0x00)
    c.step()  # SWI
    assert c.pc == 0x4000 and c.i is True
    # mutate everything in the "handler"
    c.set_regs(a=0, b=0, x=0, y=0, ccr=0xFF)
    c.step()  # RTI restores
    assert c.a == 0x11 and c.b == 0x22 and c.x == 0x3344 and c.y == 0x5566
    assert c.pc == 0x0101 and c.sp == 0x01FF


# --------------------------------------------------------------------------- #
# WAI / STOP halt the run loop with the documented reason
# --------------------------------------------------------------------------- #
def test_wai_stop_reason():
    c = make([0x3E], sp=0x01FF)  # WAI
    reason = c.run(max_steps=10)
    assert reason == "wai"


def test_stop_disabled_when_s_set_acts_as_nop():
    c = make([0xCF, 0x01], ccr=m68hc11._S)  # STOP with S set -> NOP ; NOP
    st = c.step()
    assert st.cycles == 2 and c.stopped is False
    assert c.pc == 0x0101


def test_stop_halts_when_s_clear():
    c = make([0xCF], ccr=0x00)  # STOP with S clear
    reason = c.run(max_steps=10)
    assert reason == "stop"
