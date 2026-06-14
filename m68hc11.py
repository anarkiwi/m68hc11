"""Instruction-level emulator for the Motorola 68HC11 microcontroller core.

A single self-contained module with no third-party dependencies.  The goal is
to *execute real ROM machine code* and read back exactly what it computes
(register values, memory writes, elapsed bus cycles) so that arithmetic/timing
done in firmware can be derived by running it rather than by reading a
disassembly.

The public API lives on :class:`HC11`.  See ``README.md`` for usage.

Design notes
------------
* Big-endian, single flat 64 KB address space.
* CCR layout (bit 7 -> bit 0): ``S X H I N Z V C``.
* The stack grows downward; ``SP`` points at the next *free* byte.  16-bit
  pushes store the low byte first (HC11 convention), so the value lands in
  memory big-endian with its high byte at the lower address.
* Every implemented opcode carries its documented bus-cycle count, accumulated
  in :attr:`HC11.cycles`.
* Any undefined / unimplemented opcode raises :class:`IllegalOpcode` -- the
  emulator never silently mis-decodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

__version__ = "0.1.0"
__all__ = ["HC11", "Step", "State", "IllegalOpcode", "HC11Error", "__version__"]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class HC11Error(Exception):
    """Base class for emulator errors."""


class IllegalOpcode(HC11Error):
    """Raised when an undefined / unimplemented opcode is fetched."""


# --------------------------------------------------------------------------- #
# CCR bit masks
# --------------------------------------------------------------------------- #
_S = 0x80  # Stop disable
_XF = 0x40  # XIRQ mask (called X in the manual; x_irq here to avoid clashing with index X)
_H = 0x20  # Half carry
_I = 0x10  # Interrupt mask
_N = 0x08  # Negative
_Z = 0x04  # Zero
_V = 0x02  # Overflow
_C = 0x01  # Carry


def _signed8(v: int) -> int:
    return v - 0x100 if v & 0x80 else v


# --------------------------------------------------------------------------- #
# Decoded operand passed to every handler
# --------------------------------------------------------------------------- #
class _Opnd:
    __slots__ = ("ea", "value", "mask", "target", "off")

    def __init__(self) -> None:
        self.ea: Optional[int] = None  # effective address (memory operand)
        self.value: Optional[int] = None  # immediate value (8 or 16 bit)
        self.mask: Optional[int] = None  # bit-op mask
        self.target: Optional[int] = None  # branch target
        self.off: Optional[int] = None  # raw indexed offset (for disassembly)


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """Returned by :meth:`HC11.step` and passed to the trace callback."""

    pc: int  # address of the instruction just decoded
    opcode: bytes  # full instruction bytes (including any prefix)
    mnemonic: str  # human-readable disassembly text
    cycles: int  # bus cycles this instruction costs
    ea: Optional[int]  # resolved effective address, if any


@dataclass
class State:
    """Snapshot returned by :meth:`HC11.call`."""

    a: int
    b: int
    d: int
    x: int
    y: int
    sp: int
    pc: int
    ccr: int
    s: bool
    x_irq: bool
    h: bool
    i: bool
    n: bool
    z: bool
    v: bool
    c: bool
    cycles: int  # bus cycles elapsed during the call


# --------------------------------------------------------------------------- #
# Operand source helpers (module level so the opcode tables are built once)
# --------------------------------------------------------------------------- #
def _src8(cpu: HC11, o: _Opnd) -> int:
    return o.value if o.value is not None else cpu.read8(o.ea)


def _src16(cpu: HC11, o: _Opnd) -> int:
    return o.value if o.value is not None else cpu.read16(o.ea)


# --------------------------------------------------------------------------- #
# Unary read-modify-write primitives.  Each sets flags on ``cpu`` and returns
# the 8-bit result.
# --------------------------------------------------------------------------- #
def _u_neg(cpu: HC11, v: int) -> int:
    res = (-v) & 0xFF
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = res == 0x80  # only 0x80 negates to itself
    cpu.c = res != 0  # borrow unless operand was zero
    return res


def _u_com(cpu: HC11, v: int) -> int:
    res = (~v) & 0xFF
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = False
    cpu.c = True  # COM always sets carry (6800 compatibility)
    return res


def _u_inc(cpu: HC11, v: int) -> int:
    res = (v + 1) & 0xFF
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = v == 0x7F
    return res  # carry unaffected


def _u_dec(cpu: HC11, v: int) -> int:
    res = (v - 1) & 0xFF
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = v == 0x80
    return res  # carry unaffected


def _u_asl(cpu: HC11, v: int) -> int:
    c = v & 0x80
    res = (v << 1) & 0xFF
    cpu.c = c
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = bool(cpu.n) ^ bool(c)
    return res


def _u_asr(cpu: HC11, v: int) -> int:
    c = v & 0x01
    res = ((v >> 1) | (v & 0x80)) & 0xFF  # keep sign bit
    cpu.c = c
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = bool(cpu.n) ^ bool(c)
    return res


def _u_lsr(cpu: HC11, v: int) -> int:
    c = v & 0x01
    res = (v >> 1) & 0xFF
    cpu.c = c
    cpu.n = False
    cpu.z = res == 0
    cpu.v = bool(c)  # N is always 0, so V = N ^ C = C
    return res


def _u_rol(cpu: HC11, v: int) -> int:
    old_c = 1 if cpu.c else 0
    c = v & 0x80
    res = ((v << 1) | old_c) & 0xFF
    cpu.c = c
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = bool(cpu.n) ^ bool(c)
    return res


def _u_ror(cpu: HC11, v: int) -> int:
    old_c = 0x80 if cpu.c else 0
    c = v & 0x01
    res = ((v >> 1) | old_c) & 0xFF
    cpu.c = c
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.v = bool(cpu.n) ^ bool(c)
    return res


# --------------------------------------------------------------------------- #
# Handler factories.  These build the small closures stored in the opcode
# tables.  Building them once at import keeps the per-instance state light.
# --------------------------------------------------------------------------- #
def _f_ld8(reg: str):
    def h(cpu: HC11, o: _Opnd) -> None:
        v = _src8(cpu, o)
        setattr(cpu, reg, v)
        cpu.n = v & 0x80
        cpu.z = v == 0
        cpu.v = False

    return h


def _f_st8(reg: str):
    def h(cpu: HC11, o: _Opnd) -> None:
        v = getattr(cpu, reg) & 0xFF
        cpu.write8(o.ea, v)
        cpu.n = v & 0x80
        cpu.z = v == 0
        cpu.v = False

    return h


def _f_ld16(reg: str):
    def h(cpu: HC11, o: _Opnd) -> None:
        v = _src16(cpu, o)
        setattr(cpu, reg, v)
        cpu.n = v & 0x8000
        cpu.z = v == 0
        cpu.v = False

    return h


def _f_st16(reg: str):
    def h(cpu: HC11, o: _Opnd) -> None:
        v = getattr(cpu, reg) & 0xFFFF
        cpu.write16(o.ea, v)
        cpu.n = v & 0x8000
        cpu.z = v == 0
        cpu.v = False

    return h


def _f_add(reg: str, with_carry: bool):
    def h(cpu: HC11, o: _Opnd) -> None:
        m = _src8(cpu, o)
        carry = (1 if cpu.c else 0) if with_carry else 0
        setattr(cpu, reg, cpu._add8(getattr(cpu, reg), m, carry))

    return h


def _f_sub(reg: str, with_borrow: bool, store: bool = True):
    def h(cpu: HC11, o: _Opnd) -> None:
        m = _src8(cpu, o)
        borrow = (1 if cpu.c else 0) if with_borrow else 0
        r = cpu._sub8(getattr(cpu, reg), m, borrow)
        if store:
            setattr(cpu, reg, r)

    return h


def _f_logic(reg: str, fn: Callable[[int, int], int]):
    def h(cpu: HC11, o: _Opnd) -> None:
        r = fn(getattr(cpu, reg), _src8(cpu, o)) & 0xFF
        setattr(cpu, reg, r)
        cpu.n = r & 0x80
        cpu.z = r == 0
        cpu.v = False

    return h


def _f_bit(reg: str):
    def h(cpu: HC11, o: _Opnd) -> None:
        r = (getattr(cpu, reg) & _src8(cpu, o)) & 0xFF
        cpu.n = r & 0x80
        cpu.z = r == 0
        cpu.v = False

    return h


def _f_cp16(reg: str):
    def h(cpu: HC11, o: _Opnd) -> None:
        cpu._sub16(getattr(cpu, reg), _src16(cpu, o))

    return h


def _f_unary(ufn, target: str):
    if target == "mem":

        def h(cpu: HC11, o: _Opnd) -> None:
            cpu.write8(o.ea, ufn(cpu, cpu.read8(o.ea)))

    else:

        def h(cpu: HC11, o: _Opnd) -> None:
            setattr(cpu, target, ufn(cpu, getattr(cpu, target)))

    return h


# --------------------------------------------------------------------------- #
# Standalone handlers
# --------------------------------------------------------------------------- #
def _op_addd(cpu, o):
    cpu.d = cpu._add16(cpu.d, _src16(cpu, o))


def _op_subd(cpu, o):
    cpu.d = cpu._sub16(cpu.d, _src16(cpu, o))


def _op_aba(cpu, o):
    cpu.a = cpu._add8(cpu.a, cpu.b, 0)


def _op_sba(cpu, o):
    cpu.a = cpu._sub8(cpu.a, cpu.b, 0)


def _op_cba(cpu, o):
    cpu._sub8(cpu.a, cpu.b, 0)


def _op_clra(cpu, o):
    cpu.a = 0
    cpu._clr_flags()


def _op_clrb(cpu, o):
    cpu.b = 0
    cpu._clr_flags()


def _op_clr_m(cpu, o):
    cpu.write8(o.ea, 0)
    cpu._clr_flags()


def _op_tsta(cpu, o):
    cpu._tst_flags(cpu.a)


def _op_tstb(cpu, o):
    cpu._tst_flags(cpu.b)


def _op_tst_m(cpu, o):
    cpu._tst_flags(cpu.read8(o.ea))


def _op_asld(cpu, o):
    d = cpu.d
    c = d & 0x8000
    res = (d << 1) & 0xFFFF
    cpu.d = res
    cpu.c = c
    cpu.n = res & 0x8000
    cpu.z = res == 0
    cpu.v = bool(cpu.n) ^ bool(c)


def _op_lsrd(cpu, o):
    d = cpu.d
    c = d & 0x0001
    res = d >> 1
    cpu.d = res
    cpu.c = c
    cpu.n = False
    cpu.z = res == 0
    cpu.v = bool(c)


def _op_mul(cpu, o):
    r = cpu.a * cpu.b
    cpu.d = r
    cpu.c = bool(r & 0x80)  # C = bit 7 of the 16-bit product


def _op_idiv(cpu, o):
    num = cpu.d
    den = cpu.x
    if den == 0:
        cpu.x = 0xFFFF
        cpu.c = True  # divide by zero
        cpu.v = False
        cpu.z = False
    else:
        q = num // den
        r = num % den
        cpu.x = q & 0xFFFF
        cpu.d = r & 0xFFFF
        cpu.c = False
        cpu.v = False
        cpu.z = cpu.x == 0


def _op_fdiv(cpu, o):
    num = cpu.d
    den = cpu.x
    if den == 0:
        cpu.x = 0xFFFF
        cpu.c = True
        cpu.v = True
        cpu.z = False
    else:
        cpu.v = num >= den  # quotient would overflow 16 bits
        cpu.c = False
        q = (num << 16) // den
        r = (num << 16) % den
        cpu.x = q & 0xFFFF
        cpu.d = r & 0xFFFF
        cpu.z = cpu.x == 0


def _op_daa(cpu, o):
    a = cpu.a
    lo = a & 0x0F
    hi = a >> 4
    corr = 0
    if cpu.h or lo > 9:
        corr |= 0x06
    set_c = cpu.c or hi > 9 or (hi == 9 and lo > 9)
    if set_c:
        corr |= 0x60
    res = (a + corr) & 0xFF
    cpu.a = res
    cpu.n = res & 0x80
    cpu.z = res == 0
    cpu.c = set_c
    # V is documented as undefined after DAA; left unchanged for determinism.


def _op_nop(cpu, o):
    pass


def _op_test(cpu, o):
    # Factory test-mode instruction; behaves as a no-op outside test mode.
    pass


def _op_stop(cpu, o):
    if cpu.s:
        return  # S set -> STOP disabled, acts as a 2-cycle NOP
    cpu.stopped = True


def _op_wai(cpu, o):
    cpu._push_context()
    cpu.waiting = True


def _op_swi(cpu, o):
    cpu._push_context()
    cpu.i = True
    cpu.pc = cpu.read16(0xFFF6)


def _op_rti(cpu, o):
    cpu.ccr = cpu._pull8()
    cpu.b = cpu._pull8()
    cpu.a = cpu._pull8()
    cpu.x = cpu._pull16()
    cpu.y = cpu._pull16()
    cpu.pc = cpu._pull16()


# control flow
def _op_jmp(cpu, o):
    cpu.pc = o.ea


def _op_jsr(cpu, o):
    cpu._push16(cpu.pc)  # PC already points past the instruction
    cpu.pc = o.ea


def _op_bsr(cpu, o):
    cpu._push16(cpu.pc)
    cpu.pc = o.target


def _op_rts(cpu, o):
    cpu.pc = cpu._pull16()


# branches
def _op_bra(cpu, o):
    cpu.pc = o.target


def _op_brn(cpu, o):
    pass


def _op_bhi(cpu, o):
    if not cpu.c and not cpu.z:
        cpu.pc = o.target


def _op_bls(cpu, o):
    if cpu.c or cpu.z:
        cpu.pc = o.target


def _op_bcc(cpu, o):
    if not cpu.c:
        cpu.pc = o.target


def _op_bcs(cpu, o):
    if cpu.c:
        cpu.pc = o.target


def _op_bne(cpu, o):
    if not cpu.z:
        cpu.pc = o.target


def _op_beq(cpu, o):
    if cpu.z:
        cpu.pc = o.target


def _op_bvc(cpu, o):
    if not cpu.v:
        cpu.pc = o.target


def _op_bvs(cpu, o):
    if cpu.v:
        cpu.pc = o.target


def _op_bpl(cpu, o):
    if not cpu.n:
        cpu.pc = o.target


def _op_bmi(cpu, o):
    if cpu.n:
        cpu.pc = o.target


def _op_bge(cpu, o):
    if bool(cpu.n) == bool(cpu.v):
        cpu.pc = o.target


def _op_blt(cpu, o):
    if bool(cpu.n) != bool(cpu.v):
        cpu.pc = o.target


def _op_bgt(cpu, o):
    if not cpu.z and (bool(cpu.n) == bool(cpu.v)):
        cpu.pc = o.target


def _op_ble(cpu, o):
    if cpu.z or (bool(cpu.n) != bool(cpu.v)):
        cpu.pc = o.target


# condition-code ops
def _op_clc(cpu, o):
    cpu.c = False


def _op_sec(cpu, o):
    cpu.c = True


def _op_cli(cpu, o):
    cpu.i = False


def _op_sei(cpu, o):
    cpu.i = True


def _op_clv(cpu, o):
    cpu.v = False


def _op_sev(cpu, o):
    cpu.v = True


# transfers / index ops
def _op_tab(cpu, o):
    cpu.b = cpu.a
    cpu.n = cpu.a & 0x80
    cpu.z = cpu.a == 0
    cpu.v = False


def _op_tba(cpu, o):
    cpu.a = cpu.b
    cpu.n = cpu.b & 0x80
    cpu.z = cpu.b == 0
    cpu.v = False


def _op_tap(cpu, o):
    new = cpu.a & 0xFF
    # The X (XIRQ) mask can be cleared but never set by software.
    if not (cpu.ccr & _XF):
        new &= ~_XF & 0xFF
    cpu.ccr = new


def _op_tpa(cpu, o):
    cpu.a = cpu.ccr


def _op_tsx(cpu, o):
    cpu.x = (cpu.sp + 1) & 0xFFFF


def _op_tsy(cpu, o):
    cpu.y = (cpu.sp + 1) & 0xFFFF


def _op_txs(cpu, o):
    cpu.sp = (cpu.x - 1) & 0xFFFF


def _op_tys(cpu, o):
    cpu.sp = (cpu.y - 1) & 0xFFFF


def _op_ins(cpu, o):
    cpu.sp = (cpu.sp + 1) & 0xFFFF


def _op_des(cpu, o):
    cpu.sp = (cpu.sp - 1) & 0xFFFF


def _op_inx(cpu, o):
    cpu.x = (cpu.x + 1) & 0xFFFF
    cpu.z = cpu.x == 0


def _op_dex(cpu, o):
    cpu.x = (cpu.x - 1) & 0xFFFF
    cpu.z = cpu.x == 0


def _op_iny(cpu, o):
    cpu.y = (cpu.y + 1) & 0xFFFF
    cpu.z = cpu.y == 0


def _op_dey(cpu, o):
    cpu.y = (cpu.y - 1) & 0xFFFF
    cpu.z = cpu.y == 0


def _op_abx(cpu, o):
    cpu.x = (cpu.x + cpu.b) & 0xFFFF


def _op_aby(cpu, o):
    cpu.y = (cpu.y + cpu.b) & 0xFFFF


def _op_xgdx(cpu, o):
    d = cpu.d
    cpu.d = cpu.x
    cpu.x = d


def _op_xgdy(cpu, o):
    d = cpu.d
    cpu.d = cpu.y
    cpu.y = d


# push / pull
def _op_psha(cpu, o):
    cpu._push8(cpu.a)


def _op_pshb(cpu, o):
    cpu._push8(cpu.b)


def _op_pshx(cpu, o):
    cpu._push16(cpu.x)


def _op_pshy(cpu, o):
    cpu._push16(cpu.y)


def _op_pula(cpu, o):
    cpu.a = cpu._pull8()


def _op_pulb(cpu, o):
    cpu.b = cpu._pull8()


def _op_pulx(cpu, o):
    cpu.x = cpu._pull16()


def _op_puly(cpu, o):
    cpu.y = cpu._pull16()


# bit manipulation
def _op_bset(cpu, o):
    r = (cpu.read8(o.ea) | o.mask) & 0xFF
    cpu.write8(o.ea, r)
    cpu.n = r & 0x80
    cpu.z = r == 0
    cpu.v = False


def _op_bclr(cpu, o):
    r = cpu.read8(o.ea) & (~o.mask & 0xFF)
    cpu.write8(o.ea, r)
    cpu.n = r & 0x80
    cpu.z = r == 0
    cpu.v = False


def _op_brset(cpu, o):
    m = cpu.read8(o.ea)
    if (m & o.mask) == o.mask:  # all selected bits set
        cpu.pc = o.target


def _op_brclr(cpu, o):
    m = cpu.read8(o.ea)
    if (m & o.mask) == 0:  # all selected bits clear
        cpu.pc = o.target


# --------------------------------------------------------------------------- #
# Build the named handlers from the factories
# --------------------------------------------------------------------------- #
_ldaa = _f_ld8("a")
_ldab = _f_ld8("b")
_staa = _f_st8("a")
_stab = _f_st8("b")
_ldd = _f_ld16("d")
_std = _f_st16("d")
_ldx = _f_ld16("x")
_stx = _f_st16("x")
_ldy = _f_ld16("y")
_sty = _f_st16("y")
_lds = _f_ld16("sp")
_sts = _f_st16("sp")

_adda = _f_add("a", False)
_addb = _f_add("b", False)
_adca = _f_add("a", True)
_adcb = _f_add("b", True)
_suba = _f_sub("a", False)
_subb = _f_sub("b", False)
_sbca = _f_sub("a", True)
_sbcb = _f_sub("b", True)
_cmpa = _f_sub("a", False, store=False)
_cmpb = _f_sub("b", False, store=False)

_anda = _f_logic("a", lambda p, q: p & q)
_andb = _f_logic("b", lambda p, q: p & q)
_oraa = _f_logic("a", lambda p, q: p | q)
_orab = _f_logic("b", lambda p, q: p | q)
_eora = _f_logic("a", lambda p, q: p ^ q)
_eorb = _f_logic("b", lambda p, q: p ^ q)
_bita = _f_bit("a")
_bitb = _f_bit("b")

_cpd = _f_cp16("d")
_cpx = _f_cp16("x")
_cpy = _f_cp16("y")

_nega = _f_unary(_u_neg, "a")
_negb = _f_unary(_u_neg, "b")
_neg_m = _f_unary(_u_neg, "mem")
_coma = _f_unary(_u_com, "a")
_comb = _f_unary(_u_com, "b")
_com_m = _f_unary(_u_com, "mem")
_inca = _f_unary(_u_inc, "a")
_incb = _f_unary(_u_inc, "b")
_inc_m = _f_unary(_u_inc, "mem")
_deca = _f_unary(_u_dec, "a")
_decb = _f_unary(_u_dec, "b")
_dec_m = _f_unary(_u_dec, "mem")
_asla = _f_unary(_u_asl, "a")
_aslb = _f_unary(_u_asl, "b")
_asl_m = _f_unary(_u_asl, "mem")
_asra = _f_unary(_u_asr, "a")
_asrb = _f_unary(_u_asr, "b")
_asr_m = _f_unary(_u_asr, "mem")
_lsra = _f_unary(_u_lsr, "a")
_lsrb = _f_unary(_u_lsr, "b")
_lsr_m = _f_unary(_u_lsr, "mem")
_rola = _f_unary(_u_rol, "a")
_rolb = _f_unary(_u_rol, "b")
_rol_m = _f_unary(_u_rol, "mem")
_rora = _f_unary(_u_ror, "a")
_rorb = _f_unary(_u_ror, "b")
_ror_m = _f_unary(_u_ror, "mem")


# --------------------------------------------------------------------------- #
# Opcode tables.  Each entry: (mnemonic, mode, handler, cycles).
#
# Modes: inh imm8 imm16 dir ext idx rel bdir bidx brdir bridx
# The index register used by idx/bidx/bridx is chosen by the page:
#   page 1 -> X, page 2 ($18) -> Y, page 3 ($1A) -> X, page 4 ($CD) -> Y.
# --------------------------------------------------------------------------- #
Entry = Tuple[str, str, Callable, int]

PAGE1: dict[int, Entry] = {
    0x00: ("TEST", "inh", _op_test, 1),
    0x01: ("NOP", "inh", _op_nop, 2),
    0x02: ("IDIV", "inh", _op_idiv, 41),
    0x03: ("FDIV", "inh", _op_fdiv, 41),
    0x04: ("LSRD", "inh", _op_lsrd, 3),
    0x05: ("ASLD", "inh", _op_asld, 3),
    0x06: ("TAP", "inh", _op_tap, 2),
    0x07: ("TPA", "inh", _op_tpa, 2),
    0x08: ("INX", "inh", _op_inx, 3),
    0x09: ("DEX", "inh", _op_dex, 3),
    0x0A: ("CLV", "inh", _op_clv, 2),
    0x0B: ("SEV", "inh", _op_sev, 2),
    0x0C: ("CLC", "inh", _op_clc, 2),
    0x0D: ("SEC", "inh", _op_sec, 2),
    0x0E: ("CLI", "inh", _op_cli, 2),
    0x0F: ("SEI", "inh", _op_sei, 2),
    0x10: ("SBA", "inh", _op_sba, 2),
    0x11: ("CBA", "inh", _op_cba, 2),
    0x12: ("BRSET", "brdir", _op_brset, 6),
    0x13: ("BRCLR", "brdir", _op_brclr, 6),
    0x14: ("BSET", "bdir", _op_bset, 6),
    0x15: ("BCLR", "bdir", _op_bclr, 6),
    0x16: ("TAB", "inh", _op_tab, 2),
    0x17: ("TBA", "inh", _op_tba, 2),
    # 0x18 prefix (page 2)
    0x19: ("DAA", "inh", _op_daa, 2),
    # 0x1A prefix (page 3)
    0x1B: ("ABA", "inh", _op_aba, 2),
    0x1C: ("BSET", "bidx", _op_bset, 7),
    0x1D: ("BCLR", "bidx", _op_bclr, 7),
    0x1E: ("BRSET", "bridx", _op_brset, 7),
    0x1F: ("BRCLR", "bridx", _op_brclr, 7),
    0x20: ("BRA", "rel", _op_bra, 3),
    0x21: ("BRN", "rel", _op_brn, 3),
    0x22: ("BHI", "rel", _op_bhi, 3),
    0x23: ("BLS", "rel", _op_bls, 3),
    0x24: ("BCC", "rel", _op_bcc, 3),
    0x25: ("BCS", "rel", _op_bcs, 3),
    0x26: ("BNE", "rel", _op_bne, 3),
    0x27: ("BEQ", "rel", _op_beq, 3),
    0x28: ("BVC", "rel", _op_bvc, 3),
    0x29: ("BVS", "rel", _op_bvs, 3),
    0x2A: ("BPL", "rel", _op_bpl, 3),
    0x2B: ("BMI", "rel", _op_bmi, 3),
    0x2C: ("BGE", "rel", _op_bge, 3),
    0x2D: ("BLT", "rel", _op_blt, 3),
    0x2E: ("BGT", "rel", _op_bgt, 3),
    0x2F: ("BLE", "rel", _op_ble, 3),
    0x30: ("TSX", "inh", _op_tsx, 3),
    0x31: ("INS", "inh", _op_ins, 3),
    0x32: ("PULA", "inh", _op_pula, 4),
    0x33: ("PULB", "inh", _op_pulb, 4),
    0x34: ("DES", "inh", _op_des, 3),
    0x35: ("TXS", "inh", _op_txs, 3),
    0x36: ("PSHA", "inh", _op_psha, 3),
    0x37: ("PSHB", "inh", _op_pshb, 3),
    0x38: ("PULX", "inh", _op_pulx, 5),
    0x39: ("RTS", "inh", _op_rts, 5),
    0x3A: ("ABX", "inh", _op_abx, 3),
    0x3B: ("RTI", "inh", _op_rti, 12),
    0x3C: ("PSHX", "inh", _op_pshx, 4),
    0x3D: ("MUL", "inh", _op_mul, 10),
    0x3E: ("WAI", "inh", _op_wai, 14),
    0x3F: ("SWI", "inh", _op_swi, 14),
    0x40: ("NEGA", "inh", _nega, 2),
    0x43: ("COMA", "inh", _coma, 2),
    0x44: ("LSRA", "inh", _lsra, 2),
    0x46: ("RORA", "inh", _rora, 2),
    0x47: ("ASRA", "inh", _asra, 2),
    0x48: ("ASLA", "inh", _asla, 2),
    0x49: ("ROLA", "inh", _rola, 2),
    0x4A: ("DECA", "inh", _deca, 2),
    0x4C: ("INCA", "inh", _inca, 2),
    0x4D: ("TSTA", "inh", _op_tsta, 2),
    0x4F: ("CLRA", "inh", _op_clra, 2),
    0x50: ("NEGB", "inh", _negb, 2),
    0x53: ("COMB", "inh", _comb, 2),
    0x54: ("LSRB", "inh", _lsrb, 2),
    0x56: ("RORB", "inh", _rorb, 2),
    0x57: ("ASRB", "inh", _asrb, 2),
    0x58: ("ASLB", "inh", _aslb, 2),
    0x59: ("ROLB", "inh", _rolb, 2),
    0x5A: ("DECB", "inh", _decb, 2),
    0x5C: ("INCB", "inh", _incb, 2),
    0x5D: ("TSTB", "inh", _op_tstb, 2),
    0x5F: ("CLRB", "inh", _op_clrb, 2),
    0x60: ("NEG", "idx", _neg_m, 6),
    0x63: ("COM", "idx", _com_m, 6),
    0x64: ("LSR", "idx", _lsr_m, 6),
    0x66: ("ROR", "idx", _ror_m, 6),
    0x67: ("ASR", "idx", _asr_m, 6),
    0x68: ("ASL", "idx", _asl_m, 6),
    0x69: ("ROL", "idx", _rol_m, 6),
    0x6A: ("DEC", "idx", _dec_m, 6),
    0x6C: ("INC", "idx", _inc_m, 6),
    0x6D: ("TST", "idx", _op_tst_m, 6),
    0x6E: ("JMP", "idx", _op_jmp, 3),
    0x6F: ("CLR", "idx", _op_clr_m, 6),
    0x70: ("NEG", "ext", _neg_m, 6),
    0x73: ("COM", "ext", _com_m, 6),
    0x74: ("LSR", "ext", _lsr_m, 6),
    0x76: ("ROR", "ext", _ror_m, 6),
    0x77: ("ASR", "ext", _asr_m, 6),
    0x78: ("ASL", "ext", _asl_m, 6),
    0x79: ("ROL", "ext", _rol_m, 6),
    0x7A: ("DEC", "ext", _dec_m, 6),
    0x7C: ("INC", "ext", _inc_m, 6),
    0x7D: ("TST", "ext", _op_tst_m, 6),
    0x7E: ("JMP", "ext", _op_jmp, 3),
    0x7F: ("CLR", "ext", _op_clr_m, 6),
    0x80: ("SUBA", "imm8", _suba, 2),
    0x81: ("CMPA", "imm8", _cmpa, 2),
    0x82: ("SBCA", "imm8", _sbca, 2),
    0x83: ("SUBD", "imm16", _op_subd, 4),
    0x84: ("ANDA", "imm8", _anda, 2),
    0x85: ("BITA", "imm8", _bita, 2),
    0x86: ("LDAA", "imm8", _ldaa, 2),
    0x88: ("EORA", "imm8", _eora, 2),
    0x89: ("ADCA", "imm8", _adca, 2),
    0x8A: ("ORAA", "imm8", _oraa, 2),
    0x8B: ("ADDA", "imm8", _adda, 2),
    0x8C: ("CPX", "imm16", _cpx, 4),
    0x8D: ("BSR", "rel", _op_bsr, 6),
    0x8E: ("LDS", "imm16", _lds, 3),
    0x8F: ("XGDX", "inh", _op_xgdx, 3),
    0x90: ("SUBA", "dir", _suba, 3),
    0x91: ("CMPA", "dir", _cmpa, 3),
    0x92: ("SBCA", "dir", _sbca, 3),
    0x93: ("SUBD", "dir", _op_subd, 5),
    0x94: ("ANDA", "dir", _anda, 3),
    0x95: ("BITA", "dir", _bita, 3),
    0x96: ("LDAA", "dir", _ldaa, 3),
    0x97: ("STAA", "dir", _staa, 3),
    0x98: ("EORA", "dir", _eora, 3),
    0x99: ("ADCA", "dir", _adca, 3),
    0x9A: ("ORAA", "dir", _oraa, 3),
    0x9B: ("ADDA", "dir", _adda, 3),
    0x9C: ("CPX", "dir", _cpx, 5),
    0x9D: ("JSR", "dir", _op_jsr, 5),
    0x9E: ("LDS", "dir", _lds, 4),
    0x9F: ("STS", "dir", _sts, 4),
    0xA0: ("SUBA", "idx", _suba, 4),
    0xA1: ("CMPA", "idx", _cmpa, 4),
    0xA2: ("SBCA", "idx", _sbca, 4),
    0xA3: ("SUBD", "idx", _op_subd, 6),
    0xA4: ("ANDA", "idx", _anda, 4),
    0xA5: ("BITA", "idx", _bita, 4),
    0xA6: ("LDAA", "idx", _ldaa, 4),
    0xA7: ("STAA", "idx", _staa, 4),
    0xA8: ("EORA", "idx", _eora, 4),
    0xA9: ("ADCA", "idx", _adca, 4),
    0xAA: ("ORAA", "idx", _oraa, 4),
    0xAB: ("ADDA", "idx", _adda, 4),
    0xAC: ("CPX", "idx", _cpx, 6),
    0xAD: ("JSR", "idx", _op_jsr, 6),
    0xAE: ("LDS", "idx", _lds, 5),
    0xAF: ("STS", "idx", _sts, 5),
    0xB0: ("SUBA", "ext", _suba, 4),
    0xB1: ("CMPA", "ext", _cmpa, 4),
    0xB2: ("SBCA", "ext", _sbca, 4),
    0xB3: ("SUBD", "ext", _op_subd, 6),
    0xB4: ("ANDA", "ext", _anda, 4),
    0xB5: ("BITA", "ext", _bita, 4),
    0xB6: ("LDAA", "ext", _ldaa, 4),
    0xB7: ("STAA", "ext", _staa, 4),
    0xB8: ("EORA", "ext", _eora, 4),
    0xB9: ("ADCA", "ext", _adca, 4),
    0xBA: ("ORAA", "ext", _oraa, 4),
    0xBB: ("ADDA", "ext", _adda, 4),
    0xBC: ("CPX", "ext", _cpx, 6),
    0xBD: ("JSR", "ext", _op_jsr, 6),
    0xBE: ("LDS", "ext", _lds, 5),
    0xBF: ("STS", "ext", _sts, 5),
    0xC0: ("SUBB", "imm8", _subb, 2),
    0xC1: ("CMPB", "imm8", _cmpb, 2),
    0xC2: ("SBCB", "imm8", _sbcb, 2),
    0xC3: ("ADDD", "imm16", _op_addd, 4),
    0xC4: ("ANDB", "imm8", _andb, 2),
    0xC5: ("BITB", "imm8", _bitb, 2),
    0xC6: ("LDAB", "imm8", _ldab, 2),
    0xC8: ("EORB", "imm8", _eorb, 2),
    0xC9: ("ADCB", "imm8", _adcb, 2),
    0xCA: ("ORAB", "imm8", _orab, 2),
    0xCB: ("ADDB", "imm8", _addb, 2),
    0xCC: ("LDD", "imm16", _ldd, 3),
    # 0xCD prefix (page 4)
    0xCE: ("LDX", "imm16", _ldx, 3),
    0xCF: ("STOP", "inh", _op_stop, 2),
    0xD0: ("SUBB", "dir", _subb, 3),
    0xD1: ("CMPB", "dir", _cmpb, 3),
    0xD2: ("SBCB", "dir", _sbcb, 3),
    0xD3: ("ADDD", "dir", _op_addd, 5),
    0xD4: ("ANDB", "dir", _andb, 3),
    0xD5: ("BITB", "dir", _bitb, 3),
    0xD6: ("LDAB", "dir", _ldab, 3),
    0xD7: ("STAB", "dir", _stab, 3),
    0xD8: ("EORB", "dir", _eorb, 3),
    0xD9: ("ADCB", "dir", _adcb, 3),
    0xDA: ("ORAB", "dir", _orab, 3),
    0xDB: ("ADDB", "dir", _addb, 3),
    0xDC: ("LDD", "dir", _ldd, 4),
    0xDD: ("STD", "dir", _std, 4),
    0xDE: ("LDX", "dir", _ldx, 4),
    0xDF: ("STX", "dir", _stx, 4),
    0xE0: ("SUBB", "idx", _subb, 4),
    0xE1: ("CMPB", "idx", _cmpb, 4),
    0xE2: ("SBCB", "idx", _sbcb, 4),
    0xE3: ("ADDD", "idx", _op_addd, 6),
    0xE4: ("ANDB", "idx", _andb, 4),
    0xE5: ("BITB", "idx", _bitb, 4),
    0xE6: ("LDAB", "idx", _ldab, 4),
    0xE7: ("STAB", "idx", _stab, 4),
    0xE8: ("EORB", "idx", _eorb, 4),
    0xE9: ("ADCB", "idx", _adcb, 4),
    0xEA: ("ORAB", "idx", _orab, 4),
    0xEB: ("ADDB", "idx", _addb, 4),
    0xEC: ("LDD", "idx", _ldd, 5),
    0xED: ("STD", "idx", _std, 5),
    0xEE: ("LDX", "idx", _ldx, 5),
    0xEF: ("STX", "idx", _stx, 5),
    0xF0: ("SUBB", "ext", _subb, 4),
    0xF1: ("CMPB", "ext", _cmpb, 4),
    0xF2: ("SBCB", "ext", _sbcb, 4),
    0xF3: ("ADDD", "ext", _op_addd, 6),
    0xF4: ("ANDB", "ext", _andb, 4),
    0xF5: ("BITB", "ext", _bitb, 4),
    0xF6: ("LDAB", "ext", _ldab, 4),
    0xF7: ("STAB", "ext", _stab, 4),
    0xF8: ("EORB", "ext", _eorb, 4),
    0xF9: ("ADCB", "ext", _adcb, 4),
    0xFA: ("ORAB", "ext", _orab, 4),
    0xFB: ("ADDB", "ext", _addb, 4),
    0xFC: ("LDD", "ext", _ldd, 5),
    0xFD: ("STD", "ext", _std, 5),
    0xFE: ("LDX", "ext", _ldx, 5),
    0xFF: ("STX", "ext", _stx, 5),
}

# Page 2 ($18): Y-register operations and Y-indexed addressing.
PAGE2: dict[int, Entry] = {
    0x08: ("INY", "inh", _op_iny, 4),
    0x09: ("DEY", "inh", _op_dey, 4),
    0x1C: ("BSET", "bidx", _op_bset, 8),
    0x1D: ("BCLR", "bidx", _op_bclr, 8),
    0x1E: ("BRSET", "bridx", _op_brset, 8),
    0x1F: ("BRCLR", "bridx", _op_brclr, 8),
    0x30: ("TSY", "inh", _op_tsy, 4),
    0x35: ("TYS", "inh", _op_tys, 4),
    0x38: ("PULY", "inh", _op_puly, 6),
    0x3A: ("ABY", "inh", _op_aby, 4),
    0x3C: ("PSHY", "inh", _op_pshy, 5),
    0x60: ("NEG", "idx", _neg_m, 7),
    0x63: ("COM", "idx", _com_m, 7),
    0x64: ("LSR", "idx", _lsr_m, 7),
    0x66: ("ROR", "idx", _ror_m, 7),
    0x67: ("ASR", "idx", _asr_m, 7),
    0x68: ("ASL", "idx", _asl_m, 7),
    0x69: ("ROL", "idx", _rol_m, 7),
    0x6A: ("DEC", "idx", _dec_m, 7),
    0x6C: ("INC", "idx", _inc_m, 7),
    0x6D: ("TST", "idx", _op_tst_m, 7),
    0x6E: ("JMP", "idx", _op_jmp, 4),
    0x6F: ("CLR", "idx", _op_clr_m, 7),
    0x8C: ("CPY", "imm16", _cpy, 5),
    0x8F: ("XGDY", "inh", _op_xgdy, 4),
    0x9C: ("CPY", "dir", _cpy, 6),
    0xA0: ("SUBA", "idx", _suba, 5),
    0xA1: ("CMPA", "idx", _cmpa, 5),
    0xA2: ("SBCA", "idx", _sbca, 5),
    0xA3: ("SUBD", "idx", _op_subd, 7),
    0xA4: ("ANDA", "idx", _anda, 5),
    0xA5: ("BITA", "idx", _bita, 5),
    0xA6: ("LDAA", "idx", _ldaa, 5),
    0xA7: ("STAA", "idx", _staa, 5),
    0xA8: ("EORA", "idx", _eora, 5),
    0xA9: ("ADCA", "idx", _adca, 5),
    0xAA: ("ORAA", "idx", _oraa, 5),
    0xAB: ("ADDA", "idx", _adda, 5),
    0xAC: ("CPY", "idx", _cpy, 7),
    0xAD: ("JSR", "idx", _op_jsr, 7),
    0xAE: ("LDS", "idx", _lds, 6),
    0xAF: ("STS", "idx", _sts, 6),
    0xBC: ("CPY", "ext", _cpy, 7),
    0xCE: ("LDY", "imm16", _ldy, 4),
    0xDE: ("LDY", "dir", _ldy, 5),
    0xDF: ("STY", "dir", _sty, 5),
    0xE0: ("SUBB", "idx", _subb, 5),
    0xE1: ("CMPB", "idx", _cmpb, 5),
    0xE2: ("SBCB", "idx", _sbcb, 5),
    0xE3: ("ADDD", "idx", _op_addd, 7),
    0xE4: ("ANDB", "idx", _andb, 5),
    0xE5: ("BITB", "idx", _bitb, 5),
    0xE6: ("LDAB", "idx", _ldab, 5),
    0xE7: ("STAB", "idx", _stab, 5),
    0xE8: ("EORB", "idx", _eorb, 5),
    0xE9: ("ADCB", "idx", _adcb, 5),
    0xEA: ("ORAB", "idx", _orab, 5),
    0xEB: ("ADDB", "idx", _addb, 5),
    0xEC: ("LDD", "idx", _ldd, 6),
    0xED: ("STD", "idx", _std, 6),
    0xEE: ("LDY", "idx", _ldy, 6),
    0xEF: ("STY", "idx", _sty, 6),
    0xFE: ("LDY", "ext", _ldy, 5),
    0xFF: ("STY", "ext", _sty, 5),
}

# Page 3 ($1A): CPD, plus the X-indexed forms of CPY/LDY/STY.  Indexes with X.
PAGE3: dict[int, Entry] = {
    0x83: ("CPD", "imm16", _cpd, 5),
    0x93: ("CPD", "dir", _cpd, 6),
    0xA3: ("CPD", "idx", _cpd, 7),
    0xAC: ("CPY", "idx", _cpy, 7),
    0xB3: ("CPD", "ext", _cpd, 7),
    0xEE: ("LDY", "idx", _ldy, 6),
    0xEF: ("STY", "idx", _sty, 6),
}

# Page 4 ($CD): the Y-indexed forms of CPD/CPX/LDX/STX.  Indexes with Y.
PAGE4: dict[int, Entry] = {
    0xA3: ("CPD", "idx", _cpd, 7),
    0xAC: ("CPX", "idx", _cpx, 7),
    0xEE: ("LDX", "idx", _ldx, 6),
    0xEF: ("STX", "idx", _stx, 6),
}


# --------------------------------------------------------------------------- #
# Timer / interrupt support (optional layer, spec section 6)
# --------------------------------------------------------------------------- #
# Register offsets from the on-chip register base (default $1000).
_TCNT = 0x0E
_TOC = {1: 0x16, 2: 0x18, 3: 0x1A, 4: 0x1C, 5: 0x1E}
_TCTL1, _TCTL2 = 0x20, 0x21
_TMSK1, _TFLG1 = 0x22, 0x23
_TMSK2, _TFLG2 = 0x24, 0x25
_PACTL = 0x26

# TFLG1 / TMSK1 output-compare bit positions (OC1..OC5).
_OCF = {1: 0x80, 2: 0x40, 3: 0x20, 4: 0x10, 5: 0x08}
# Output-compare interrupt vectors.
_OC_VEC = {1: 0xFFE8, 2: 0xFFE6, 3: 0xFFE4, 4: 0xFFE2, 5: 0xFFE0}
_TOF = 0x80  # timer overflow flag/enable bit in TFLG2 / TMSK2
_TOF_VEC = 0xFFDE
_PRESCALE = (1, 4, 8, 16)


class HC11:
    """A Motorola 68HC11 CPU core with a flat 64 KB memory."""

    # ------------------------------------------------------------------ #
    # Construction / reset
    # ------------------------------------------------------------------ #
    def __init__(self, *, ram_fill: int = 0x00) -> None:
        self._fill = ram_fill & 0xFF
        self.mem = bytearray([self._fill]) * 0x10000

        # registers
        self.a = 0
        self.b = 0
        self.x = 0
        self.y = 0
        self.sp = 0x01FF  # sensible default so call() can stack immediately
        self.pc = 0
        self.ccr = _S | _XF | _I  # reset value: S, X, I set

        # execution bookkeeping
        self.cycles = 0
        self.stopped = False
        self.waiting = False

        # I/O hooks
        self._read_hooks: List[Tuple[int, int, Callable[[int], int]]] = []
        self._write_hooks: List[Tuple[int, int, Callable[[int, int], None]]] = []

        # trace + decode scratch
        self._trace: Optional[Callable[[Step], None]] = None
        self._fetched: List[int] = []

        # timer (created lazily by enable_timer)
        self._timer_on = False
        self.io_base = 0x1000
        self.tcnt = 0
        self.toc = {n: 0 for n in range(1, 6)}
        self.tmsk1 = self.tflg1 = 0
        self.tmsk2 = self.tflg2 = 0
        self.tctl1 = self.tctl2 = self.pactl = 0
        self._presc_accum = 0
        self.irq_counts: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Condition-code flag properties
    # ------------------------------------------------------------------ #
    def _flag(mask):  # noqa: N805 (descriptor factory)
        def getter(self):
            return bool(self.ccr & mask)

        def setter(self, val):
            if val:
                self.ccr |= mask
            else:
                self.ccr &= ~mask & 0xFF

        return property(getter, setter)

    s = _flag(_S)
    x_irq = _flag(_XF)
    h = _flag(_H)
    i = _flag(_I)
    n = _flag(_N)
    z = _flag(_Z)
    v = _flag(_V)
    c = _flag(_C)
    del _flag

    @property
    def d(self) -> int:
        return ((self.a << 8) | self.b) & 0xFFFF

    @d.setter
    def d(self, value: int) -> None:
        value &= 0xFFFF
        self.a = value >> 8
        self.b = value & 0xFF

    # ------------------------------------------------------------------ #
    # Memory
    # ------------------------------------------------------------------ #
    def load(self, data: bytes, base: int) -> None:
        """Copy a raw binary blob to ``base`` (bypasses I/O hooks)."""
        base &= 0xFFFF
        for i, byte in enumerate(data):
            self.mem[(base + i) & 0xFFFF] = byte & 0xFF

    def read8(self, addr: int) -> int:
        addr &= 0xFFFF
        for start, end, fn in self._read_hooks:
            if start <= addr <= end:
                return fn(addr) & 0xFF
        return self.mem[addr]

    def write8(self, addr: int, val: int) -> None:
        addr &= 0xFFFF
        val &= 0xFF
        for start, end, fn in self._write_hooks:
            if start <= addr <= end:
                fn(addr, val)
                return
        self.mem[addr] = val

    def read16(self, addr: int) -> int:
        return (self.read8(addr) << 8) | self.read8(addr + 1)

    def write16(self, addr: int, value: int) -> None:
        value &= 0xFFFF
        self.write8(addr, value >> 8)
        self.write8(addr + 1, value & 0xFF)

    def read(self, addr: int, n: int = 1) -> bytes:
        return bytes(self.read8(addr + i) for i in range(n))

    def write(self, addr: int, data: Union[bytes, int]) -> None:
        if isinstance(data, int):
            self.write8(addr, data)
        else:
            for i, byte in enumerate(data):
                self.write8(addr + i, byte)

    def on_read(self, start: int, end: int, fn: Callable[[int], int]) -> None:
        """Register a read hook for ``[start, end]`` (inclusive)."""
        self._read_hooks.append((start & 0xFFFF, end & 0xFFFF, fn))

    def on_write(self, start: int, end: int, fn: Callable[[int, int], None]) -> None:
        """Register a write hook for ``[start, end]`` (inclusive)."""
        self._write_hooks.append((start & 0xFFFF, end & 0xFFFF, fn))

    # ------------------------------------------------------------------ #
    # Register helpers
    # ------------------------------------------------------------------ #
    def set_regs(
        self, *, a=None, b=None, d=None, x=None, y=None, sp=None, pc=None, ccr=None
    ) -> None:
        if d is not None:
            self.d = d
        if a is not None:
            self.a = a & 0xFF
        if b is not None:
            self.b = b & 0xFF
        if x is not None:
            self.x = x & 0xFFFF
        if y is not None:
            self.y = y & 0xFFFF
        if sp is not None:
            self.sp = sp & 0xFFFF
        if pc is not None:
            self.pc = pc & 0xFFFF
        if ccr is not None:
            self.ccr = ccr & 0xFF

    def reset_cycles(self) -> None:
        self.cycles = 0

    # ------------------------------------------------------------------ #
    # Internal ALU helpers (set flags, return result)
    # ------------------------------------------------------------------ #
    def _add8(self, x: int, m: int, carry: int) -> int:
        r = x + m + carry
        res = r & 0xFF
        self.h = ((x & 0xF) + (m & 0xF) + carry) > 0xF
        self.n = res & 0x80
        self.z = res == 0
        self.v = bool((x ^ res) & (m ^ res) & 0x80)
        self.c = r > 0xFF
        return res

    def _sub8(self, x: int, m: int, borrow: int) -> int:
        r = x - m - borrow
        res = r & 0xFF
        self.n = res & 0x80
        self.z = res == 0
        self.v = bool((x ^ m) & (x ^ res) & 0x80)
        self.c = r < 0
        return res

    def _add16(self, x: int, m: int) -> int:
        r = x + m
        res = r & 0xFFFF
        self.n = res & 0x8000
        self.z = res == 0
        self.v = bool((x ^ res) & (m ^ res) & 0x8000)
        self.c = r > 0xFFFF
        return res

    def _sub16(self, x: int, m: int) -> int:
        r = x - m
        res = r & 0xFFFF
        self.n = res & 0x8000
        self.z = res == 0
        self.v = bool((x ^ m) & (x ^ res) & 0x8000)
        self.c = r < 0
        return res

    def _clr_flags(self) -> None:
        self.n = False
        self.z = True
        self.v = False
        self.c = False

    def _tst_flags(self, v: int) -> None:
        self.n = v & 0x80
        self.z = v == 0
        self.v = False
        self.c = False

    # ------------------------------------------------------------------ #
    # Stack
    # ------------------------------------------------------------------ #
    def _push8(self, v: int) -> None:
        self.write8(self.sp, v & 0xFF)
        self.sp = (self.sp - 1) & 0xFFFF

    def _pull8(self) -> int:
        self.sp = (self.sp + 1) & 0xFFFF
        return self.read8(self.sp)

    def _push16(self, v: int) -> None:
        v &= 0xFFFF
        self._push8(v & 0xFF)  # low byte first
        self._push8(v >> 8)

    def _pull16(self) -> int:
        hi = self._pull8()  # high byte first
        lo = self._pull8()
        return (hi << 8) | lo

    def _push_context(self) -> None:
        """Stack the full interrupt context (PC, Y, X, A, B, CCR)."""
        self._push16(self.pc)
        self._push16(self.y)
        self._push16(self.x)
        self._push8(self.a)
        self._push8(self.b)
        self._push8(self.ccr)

    # ------------------------------------------------------------------ #
    # Instruction fetch
    # ------------------------------------------------------------------ #
    def _fetch8(self) -> int:
        v = self.read8(self.pc)
        self._fetched.append(v)
        self.pc = (self.pc + 1) & 0xFFFF
        return v

    def _fetch16(self) -> int:
        hi = self._fetch8()
        lo = self._fetch8()
        return (hi << 8) | lo

    # ------------------------------------------------------------------ #
    # Execute one instruction
    # ------------------------------------------------------------------ #
    def step(self) -> Step:
        """Decode and execute a single instruction, returning a :class:`Step`."""
        start_pc = self.pc
        self._fetched = []
        op = self._fetch8()

        if op == 0x18:
            page, table = 2, PAGE2
            op2 = self._fetch8()
            entry = table.get(op2)
        elif op == 0x1A:
            page, table = 3, PAGE3
            op2 = self._fetch8()
            entry = table.get(op2)
        elif op == 0xCD:
            page, table = 4, PAGE4
            op2 = self._fetch8()
            entry = table.get(op2)
        else:
            page, table, op2 = 1, PAGE1, op
            entry = table.get(op)

        if entry is None:
            if page == 1:
                msg = f"illegal opcode ${op:02X} at ${start_pc:04X}"
            else:
                msg = f"illegal opcode ${op:02X}{op2:02X} at ${start_pc:04X}"
            self.pc = start_pc  # leave PC on the offending instruction
            raise IllegalOpcode(msg)

        mnem, mode, handler, cyc = entry
        idxchar = "Y" if page in (2, 4) else "X"
        idxval = self.y if page in (2, 4) else self.x

        o = _Opnd()
        text = self._decode(mnem, mode, idxchar, idxval, o)

        step_obj = Step(start_pc, bytes(self._fetched), text, cyc, o.ea)
        if self._trace is not None:
            self._trace(step_obj)

        handler(self, o)
        self.cycles += cyc
        return step_obj

    def _decode(self, mnem: str, mode: str, idxchar: str, idxval: int, o: _Opnd) -> str:
        """Read operand bytes per ``mode``, fill ``o``, return disassembly text."""
        if mode == "inh":
            return mnem
        if mode == "imm8":
            v = self._fetch8()
            o.value = v
            return f"{mnem} #${v:02X}"
        if mode == "imm16":
            v = self._fetch16()
            o.value = v
            return f"{mnem} #${v:04X}"
        if mode == "dir":
            a = self._fetch8()
            o.ea = a
            return f"{mnem} ${a:02X}"
        if mode == "ext":
            a = self._fetch16()
            o.ea = a
            return f"{mnem} ${a:04X}"
        if mode == "idx":
            off = self._fetch8()
            o.off = off
            o.ea = (idxval + off) & 0xFFFF
            return f"{mnem} ${off:02X},{idxchar}"
        if mode == "rel":
            rel = self._fetch8()
            tgt = (self.pc + _signed8(rel)) & 0xFFFF
            o.target = o.ea = tgt
            return f"{mnem} ${tgt:04X}"
        if mode == "bdir":
            a = self._fetch8()
            mask = self._fetch8()
            o.ea = a
            o.mask = mask
            return f"{mnem} ${a:02X},#${mask:02X}"
        if mode == "bidx":
            off = self._fetch8()
            mask = self._fetch8()
            o.off = off
            o.ea = (idxval + off) & 0xFFFF
            o.mask = mask
            return f"{mnem} ${off:02X},{idxchar},#${mask:02X}"
        if mode == "brdir":
            a = self._fetch8()
            mask = self._fetch8()
            rel = self._fetch8()
            o.ea = a
            o.mask = mask
            tgt = (self.pc + _signed8(rel)) & 0xFFFF
            o.target = tgt
            return f"{mnem} ${a:02X},#${mask:02X},${tgt:04X}"
        if mode == "bridx":
            off = self._fetch8()
            mask = self._fetch8()
            rel = self._fetch8()
            o.off = off
            o.ea = (idxval + off) & 0xFFFF
            o.mask = mask
            tgt = (self.pc + _signed8(rel)) & 0xFFFF
            o.target = tgt
            return f"{mnem} ${off:02X},{idxchar},#${mask:02X},${tgt:04X}"
        raise HC11Error(f"unknown addressing mode {mode!r}")  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        max_steps: Optional[int] = None,
        until_pc: Optional[int] = None,
        until_cycles: Optional[int] = None,
    ) -> str:
        """Execute until a stop condition is hit; return the stop reason.

        Reasons: ``"max_steps"``, ``"until_pc"``, ``"until_cycles"``,
        ``"wai"``, ``"stop"``, ``"illegal"``.
        """
        self.stopped = False
        self.waiting = False
        steps = 0
        while True:
            if until_pc is not None and self.pc == until_pc:
                return "until_pc"
            if until_cycles is not None and self.cycles >= until_cycles:
                return "until_cycles"
            if max_steps is not None and steps >= max_steps:
                return "max_steps"
            try:
                step = self.step()
            except IllegalOpcode:
                return "illegal"
            steps += 1
            if self._timer_on:
                self._timer_advance(step.cycles)
                self._service_irq()
            if self.stopped:
                return "stop"
            if self.waiting:
                if self._timer_on and self._run_wai(until_cycles):
                    continue  # an interrupt woke us; keep running
                return "wai"

    def call(
        self,
        addr: int,
        *,
        a=None,
        b=None,
        d=None,
        x=None,
        y=None,
        max_steps: int = 1_000_000,
        sentinel: int = 0xFFFE,
    ) -> State:
        """Call ``addr`` like a subroutine and return the post-state.

        Pushes ``sentinel`` as the return address, sets the requested argument
        registers and ``PC=addr``, then single-steps until the routine's final
        ``RTS`` pops the sentinel (``PC == sentinel``) or ``max_steps`` elapses.
        """
        self.set_regs(a=a, b=b, d=d, x=x, y=y)
        self._push16(sentinel)
        self.pc = addr & 0xFFFF
        start = self.cycles
        self.stopped = False
        self.waiting = False
        for _ in range(max_steps):
            if self.pc == sentinel:
                break
            self.step()
            if self._timer_on:
                self._timer_advance_for_call()
            if self.stopped or self.waiting:
                break
        return self._snapshot(self.cycles - start)

    def _snapshot(self, elapsed: int) -> State:
        return State(
            a=self.a,
            b=self.b,
            d=self.d,
            x=self.x,
            y=self.y,
            sp=self.sp,
            pc=self.pc,
            ccr=self.ccr,
            s=self.s,
            x_irq=self.x_irq,
            h=self.h,
            i=self.i,
            n=self.n,
            z=self.z,
            v=self.v,
            c=self.c,
            cycles=elapsed,
        )

    # ------------------------------------------------------------------ #
    # Tracing
    # ------------------------------------------------------------------ #
    def set_trace(self, fn: Optional[Callable[[Step], None]]) -> None:
        """Install (or clear with ``None``) a callback run before each instruction."""
        self._trace = fn

    def disassemble(self, addr: int) -> Step:
        """Decode the instruction at ``addr`` without executing it."""
        saved_pc, saved_fetched = self.pc, self._fetched
        try:
            self.pc = addr & 0xFFFF
            self._fetched = []
            start_pc = self.pc
            op = self._fetch8()
            if op == 0x18:
                page, table = 2, PAGE2
                entry = table.get(self._fetch8())
            elif op == 0x1A:
                page, table = 3, PAGE3
                entry = table.get(self._fetch8())
            elif op == 0xCD:
                page, table = 4, PAGE4
                entry = table.get(self._fetch8())
            else:
                page, table = 1, PAGE1
                entry = table.get(op)
            if entry is None:
                return Step(start_pc, bytes(self._fetched), "???", 0, None)
            mnem, mode, _handler, cyc = entry
            idxchar = "Y" if page in (2, 4) else "X"
            idxval = self.y if page in (2, 4) else self.x
            o = _Opnd()
            text = self._decode(mnem, mode, idxchar, idxval, o)
            return Step(start_pc, bytes(self._fetched), text, cyc, o.ea)
        finally:
            self.pc, self._fetched = saved_pc, saved_fetched

    # ================================================================== #
    # Timer / interrupt layer (optional, spec section 6)
    # ================================================================== #
    def enable_timer(self, base: int = 0x1000) -> None:
        """Enable the on-chip main timer and interrupt delivery.

        Installs I/O hooks for the timer registers in the ``base`` block
        (default ``$1000``).  Layered on top of the core; the core works
        without ever calling this.
        """
        self.io_base = base & 0xFFFF
        self._timer_on = True
        lo = self.io_base
        hi = self.io_base + 0x3F
        self.on_read(lo, hi, self._timer_read)
        self.on_write(lo, hi, self._timer_write)

    def _timer_read(self, addr: int) -> int:
        off = (addr - self.io_base) & 0xFFFF
        if off == _TCNT:
            return (self.tcnt >> 8) & 0xFF
        if off == _TCNT + 1:
            return self.tcnt & 0xFF
        for n, base_off in _TOC.items():
            if off == base_off:
                return (self.toc[n] >> 8) & 0xFF
            if off == base_off + 1:
                return self.toc[n] & 0xFF
        if off == _TMSK1:
            return self.tmsk1
        if off == _TFLG1:
            return self.tflg1
        if off == _TMSK2:
            return self.tmsk2
        if off == _TFLG2:
            return self.tflg2
        if off == _TCTL1:
            return self.tctl1
        if off == _TCTL2:
            return self.tctl2
        if off == _PACTL:
            return self.pactl
        return self.mem[addr]

    def _timer_write(self, addr: int, val: int) -> None:
        off = (addr - self.io_base) & 0xFFFF
        if off in (_TCNT, _TCNT + 1):
            return  # TCNT is read-only outside test mode
        for n, base_off in _TOC.items():
            if off == base_off:
                self.toc[n] = (self.toc[n] & 0x00FF) | (val << 8)
                return
            if off == base_off + 1:
                self.toc[n] = (self.toc[n] & 0xFF00) | val
                return
        if off == _TMSK1:
            self.tmsk1 = val
            return
        if off == _TFLG1:
            self.tflg1 &= ~val & 0xFF  # write 1 to clear
            return
        if off == _TMSK2:
            self.tmsk2 = val
            return
        if off == _TFLG2:
            self.tflg2 &= ~val & 0xFF
            return
        if off == _TCTL1:
            self.tctl1 = val
            return
        if off == _TCTL2:
            self.tctl2 = val
            return
        if off == _PACTL:
            self.pactl = val
            return
        self.mem[addr] = val

    def _timer_advance(self, cycles: int) -> None:
        """Advance TCNT by ``cycles`` bus cycles, raising compare/overflow flags."""
        self._presc_accum += cycles
        presc = _PRESCALE[self.tmsk2 & 0x03]
        ticks = self._presc_accum // presc
        if ticks == 0:
            return
        self._presc_accum -= ticks * presc
        old = self.tcnt
        if ticks >= 0x10000:
            ticks = 0x10000  # a full wrap touches every value once
        new = (old + ticks) & 0xFFFF
        if old + ticks > 0xFFFF:
            self.tflg2 |= _TOF
        for n, tv in self.toc.items():
            dist = (tv - old) & 0xFFFF
            if 1 <= dist <= ticks:
                self.tflg1 |= _OCF[n]
        self.tcnt = new

    def _timer_advance_for_call(self) -> None:
        # In call(), service interrupts between steps using the last step cost.
        # The cost is already folded into self.cycles; advance + service here.
        self._service_irq()

    def _pending_irq(self) -> Optional[int]:
        """Return the vector of the highest-priority pending+enabled interrupt."""
        if self.i:  # interrupts masked
            return None
        for n in range(1, 6):  # OC1 has highest priority
            if (self.tflg1 & _OCF[n]) and (self.tmsk1 & _OCF[n]):
                return _OC_VEC[n]
        if (self.tflg2 & _TOF) and (self.tmsk2 & _TOF):
            return _TOF_VEC
        return None

    def _service_irq(self) -> bool:
        vec = self._pending_irq()
        if vec is None:
            return False
        self._push_context()
        self.i = True
        self.waiting = False
        self.pc = self.read16(vec)
        self.irq_counts[vec] = self.irq_counts.get(vec, 0) + 1
        return True

    def _run_wai(self, until_cycles: Optional[int]) -> bool:
        """While halted in WAI, advance the timer until an interrupt fires.

        Returns True if an interrupt woke the CPU, False if it stays asleep.
        """
        guard = 0
        while True:
            if until_cycles is not None and self.cycles >= until_cycles:
                return False
            self.cycles += 1
            self._timer_advance(1)
            if self._service_irq():
                return True
            guard += 1
            if guard > 0x200000:  # safety: no interrupt source will ever fire
                return False

    def run_until_irq(self, vector: int, count: int = 1, max_cycles: int = 10_000_000) -> str:
        """Run until ``vector`` has been taken ``count`` times (or cycles run out).

        Returns ``"irq"`` on success or ``"max_cycles"`` if the budget is hit.
        """
        if not self._timer_on:
            raise HC11Error("timer not enabled; call enable_timer() first")
        target = self.irq_counts.get(vector, 0) + count
        end = self.cycles + max_cycles
        while self.cycles < end:
            if self.waiting:
                if not self._run_wai(end):
                    return "max_cycles"
            else:
                step = self.step()
                self._timer_advance(step.cycles)
                self._service_irq()
                if self.stopped:
                    return "stop"
            if self.irq_counts.get(vector, 0) >= target:
                return "irq"
        return "max_cycles"
