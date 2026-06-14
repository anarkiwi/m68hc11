"""Decode-table integrity and the "never silently mis-decode" guarantee."""

import pytest

import m68hc11

# Page-1 opcodes that are documented as illegal on the 68HC11.
PAGE1_ILLEGAL = {
    0x41,
    0x42,
    0x45,
    0x4B,
    0x4E,
    0x51,
    0x52,
    0x55,
    0x5B,
    0x5E,
    0x61,
    0x62,
    0x65,
    0x6B,
    0x71,
    0x72,
    0x75,
    0x7B,
    0x87,
    0xC7,
}
PREFIXES = {0x18, 0x1A, 0xCD}


def test_every_page1_opcode_is_defined_or_illegal():
    for op in range(0x100):
        if op in PREFIXES:
            continue  # handled specially as a prefix
        defined = op in m68hc11.PAGE1
        illegal = op in PAGE1_ILLEGAL
        assert defined != illegal, f"opcode ${op:02X}: defined={defined} illegal={illegal}"


def test_page1_illegal_opcodes_raise():
    for op in PAGE1_ILLEGAL:
        c = m68hc11.HC11()
        c.load(bytes([op]), 0x0100)
        c.set_regs(pc=0x0100)
        with pytest.raises(m68hc11.IllegalOpcode) as exc:
            c.step()
        assert f"${op:02X}" in str(exc.value)
        assert "$0100" in str(exc.value)  # PC is reported


def test_prefix_with_bad_second_byte_raises():
    for prefix in PREFIXES:
        c = m68hc11.HC11()
        c.load(bytes([prefix, 0x00]), 0x0100)  # $00 is not valid on any prefix page
        c.set_regs(pc=0x0100)
        with pytest.raises(m68hc11.IllegalOpcode) as exc:
            c.step()
        assert f"${prefix:02X}00" in str(exc.value)


def test_run_reports_illegal_reason():
    c = m68hc11.HC11()
    c.load(b"\x01\x41", 0x0100)  # NOP ; illegal
    c.set_regs(pc=0x0100)
    reason = c.run(max_steps=10)
    assert reason == "illegal"
    assert c.pc == 0x0101  # stopped at the bad opcode


def test_all_table_entries_well_formed():
    valid_modes = {
        "inh",
        "imm8",
        "imm16",
        "dir",
        "ext",
        "idx",
        "rel",
        "bdir",
        "bidx",
        "brdir",
        "bridx",
    }
    for name, table in [
        ("PAGE1", m68hc11.PAGE1),
        ("PAGE2", m68hc11.PAGE2),
        ("PAGE3", m68hc11.PAGE3),
        ("PAGE4", m68hc11.PAGE4),
    ]:
        for op, (mnem, mode, handler, cyc) in table.items():
            assert 0 <= op <= 0xFF, f"{name} bad opcode {op}"
            assert isinstance(mnem, str) and mnem
            assert mode in valid_modes, f"{name} ${op:02X} bad mode {mode}"
            assert callable(handler)
            assert isinstance(cyc, int) and cyc > 0, f"{name} ${op:02X} bad cycles"


def test_every_defined_opcode_executes_without_decode_error():
    # Decode + execute each defined opcode from a zero-filled image (operands
    # read as $00).  None should raise IllegalOpcode.
    def run_table(table, prefix=None):
        for op in table:
            c = m68hc11.HC11()
            c.set_regs(pc=0x0100, sp=0x01FF)
            prog = bytes([prefix, op]) if prefix is not None else bytes([op])
            c.load(prog, 0x0100)
            c.write16(0xFFF6, 0x0100)  # SWI vector, harmless
            try:
                c.step()
            except m68hc11.IllegalOpcode as err:  # pragma: no cover
                raise AssertionError(
                    f"defined opcode mis-flagged illegal: {prefix} {op:02X}"
                ) from err

    run_table(m68hc11.PAGE1)
    run_table(m68hc11.PAGE2, 0x18)
    run_table(m68hc11.PAGE3, 0x1A)
    run_table(m68hc11.PAGE4, 0xCD)


def test_disassemble_without_executing():
    c = m68hc11.HC11()
    c.load(b"\xcc\x12\x34", 0x0100)  # LDD #$1234
    st = c.disassemble(0x0100)
    assert st.mnemonic == "LDD #$1234"
    assert c.pc == 0x0000  # disassemble must not move PC
    assert c.d == 0x0000  # nor execute


def test_trace_callback_fires_before_execution():
    seen = []
    c = m68hc11.HC11()
    c.load(b"\x86\x42\x01", 0x0100)  # LDAA #$42 ; NOP
    c.set_regs(pc=0x0100)
    c.set_trace(lambda step: seen.append((step.pc, step.mnemonic, c.a)))
    c.step()
    # at trace time for the first instruction, A has not been loaded yet
    assert seen[0] == (0x0100, "LDAA #$42", 0x00)
