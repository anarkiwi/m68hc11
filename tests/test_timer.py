"""Spec section 6 -- optional on-chip main timer and interrupt delivery."""

import pytest

import m68hc11

# Register addresses in the default $1000 block.
TCNT = 0x100E
TOC2 = 0x1018
TCTL_TMSK1 = 0x1022
TFLG1 = 0x1023
TMSK2 = 0x1024
TFLG2 = 0x1025
OC2_BIT = 0x40
OC2_VEC = 0xFFE6
TOF_BIT = 0x80


def test_core_works_without_timer():
    # Until enable_timer() is called the timer registers are plain RAM.
    c = m68hc11.HC11()
    c.write8(TCNT, 0x55)
    assert c.read8(TCNT) == 0x55
    assert c._timer_on is False


def test_tcnt_free_runs_with_prescale_1():
    c = m68hc11.HC11()
    c.enable_timer()
    c.load(b"\x01" * 10, 0x0100)  # ten NOPs, 2 cycles each
    c.set_regs(pc=0x0100)
    c.run(until_cycles=20)
    assert c.tcnt == 20  # prescale 1 -> one tick per bus cycle
    assert c.read8(TCNT) == 0x00 and c.read8(TCNT + 1) == 20


def test_prescaler_divides():
    c = m68hc11.HC11()
    c.enable_timer()
    c.write8(TMSK2, 0x03)  # PR1:PR0 = 11 -> divide by 16
    c.load(b"\x01" * 20, 0x0100)
    c.set_regs(pc=0x0100)
    c.run(until_cycles=32)
    assert c.tcnt == 2  # 32 bus cycles / 16


def test_tcnt_read_is_live_through_hook():
    c = m68hc11.HC11()
    c.enable_timer()
    c.load(b"\x01\x01", 0x0100)
    c.set_regs(pc=0x0100)
    c.run(until_cycles=4)
    assert c.read16(TCNT) == 4


def test_output_compare_sets_flag():
    c = m68hc11.HC11()
    c.enable_timer()
    c.write16(TOC2, 0x0006)  # compare at count 6
    c.load(b"\x01" * 10, 0x0100)
    c.set_regs(pc=0x0100)
    c.run(until_cycles=20)
    assert c.tflg1 & OC2_BIT  # OC2F latched when TCNT swept past 6


def test_tflg_write_one_to_clear():
    c = m68hc11.HC11()
    c.enable_timer()
    c.tflg1 = 0xFF
    c.write8(TFLG1, OC2_BIT)  # writing a 1 clears that bit
    assert c.tflg1 & OC2_BIT == 0
    assert c.tflg1 == 0xFF & ~OC2_BIT


def test_output_compare_interrupt_delivered():
    c = m68hc11.HC11()
    c.enable_timer()
    c.set_regs(pc=0x0100, sp=0x01FF, ccr=0x00)  # I clear -> interrupts enabled
    c.write16(TOC2, 0x000A)
    c.write8(TCTL_TMSK1, OC2_BIT)  # enable OC2 interrupt
    c.write16(OC2_VEC, 0x3000)  # vector -> handler
    c.load(b"\x20\xfe", 0x0100)  # BRA * (spin so time passes)

    reason = c.run_until_irq(OC2_VEC, 1, max_cycles=1000)
    assert reason == "irq"
    assert c.pc == 0x3000  # jumped to the handler
    assert c.irq_counts[OC2_VEC] == 1
    assert c.i is True  # I set on interrupt entry
    assert c.sp == 0x01FF - 9  # full context stacked (9 bytes)
    assert c.tflg1 & OC2_BIT  # flag stays set until the ISR clears it


def test_interrupt_not_taken_when_masked():
    c = m68hc11.HC11()
    c.enable_timer()
    c.set_regs(pc=0x0100, sp=0x01FF, ccr=m68hc11._I)  # I set -> masked
    c.write16(TOC2, 0x0004)
    c.write8(TCTL_TMSK1, OC2_BIT)
    c.write16(OC2_VEC, 0x3000)
    c.load(b"\x01" * 20, 0x0100)
    c.run(until_cycles=30)
    assert c.tflg1 & OC2_BIT  # flag still latches...
    assert OC2_VEC not in c.irq_counts  # ...but no interrupt is taken


def test_wai_woken_by_timer_interrupt():
    c = m68hc11.HC11()
    c.enable_timer()
    c.set_regs(pc=0x0100, sp=0x01FF, ccr=0x00)
    c.write16(TOC2, 0x0005)
    c.write8(TCTL_TMSK1, OC2_BIT)
    c.write16(OC2_VEC, 0x3000)
    c.load(b"\x3e", 0x0100)  # WAI

    reason = c.run_until_irq(OC2_VEC, 1, max_cycles=1000)
    assert reason == "irq"
    assert c.irq_counts[OC2_VEC] == 1
    assert c.pc == 0x3000


def test_timer_overflow_flag():
    c = m68hc11.HC11()
    c.enable_timer()
    c.tcnt = 0xFFFE
    c.load(b"\x01" * 4, 0x0100)  # NOPs to push TCNT past the wrap
    c.set_regs(pc=0x0100)
    c.run(until_cycles=6)
    assert c.tflg2 & TOF_BIT  # TOF set on wrap through 0x0000
    assert c.tcnt < 0xFFFE


def test_run_until_irq_requires_timer():
    c = m68hc11.HC11()
    with pytest.raises(m68hc11.HC11Error):
        c.run_until_irq(OC2_VEC, 1)


def test_repeated_interrupts_counted():
    # Re-arm the compare in the handler and confirm multiple fires are counted.
    c = m68hc11.HC11()
    c.enable_timer()
    c.set_regs(pc=0x0100, sp=0x01FF, ccr=0x00)
    c.write16(TOC2, 0x0008)
    c.write8(TCTL_TMSK1, OC2_BIT)
    c.write16(OC2_VEC, 0x3000)
    # handler: re-arm TOC2 forward, clear OC2F, then RTI
    #   LDD $1018 ; ADDD #$0040 ; STD $1018 ; LDAA #$40 ; STAA $1023 ; RTI
    c.load(b"\xfc\x10\x18\xc3\x00\x40\xfd\x10\x18\x86\x40\xb7\x10\x23\x3b", 0x3000)
    c.load(b"\x20\xfe", 0x0100)  # BRA * in the foreground
    reason = c.run_until_irq(OC2_VEC, 3, max_cycles=20000)
    assert reason == "irq"
    assert c.irq_counts[OC2_VEC] >= 3
