# m68hc11

An instruction-level emulator for the **Motorola 68HC11** 8-bit microcontroller
core, in a single self-contained Python module with **no third-party runtime
dependencies**.

It exists to *execute real ROM machine code and read back exactly what it
computes* — register values, memory writes, and elapsed bus cycles — so that
arithmetic and timing done in firmware can be derived by running it rather than
by squinting at a disassembly.

- Full 68HC11 instruction set: all addressing modes, all four opcode pages
  (un-prefixed plus the `$18` / `$1A` / `$CD` prefix pages), bit-manipulation
  (`BSET`/`BCLR`/`BRSET`/`BRCLR`), and the fiddly ALU ops (`MUL`, `IDIV`,
  `FDIV`, `DAA`, the shifts/rotates, `ABA`/`SBA`/`CBA`, `ADDD`/`SUBD`, …) with
  **correct condition-code flag effects**.
- A documented **per-instruction cycle count** accumulated into a running total,
  so elapsed time follows from a known bus-clock frequency.
- A flat 64 KB memory with **I/O hooks** for memory-mapped peripheral registers
  / open bus.
- A `call()` harness that runs a subroutine like a function and hands back the
  full post-state.
- An optional, layered **main-timer + interrupt** model (output compares, timer
  overflow, vectored interrupt delivery).
- **Illegal opcodes raise** — the emulator never silently mis-decodes.

Determinism: pure functions only, no wall-clock, no RNG, no global state. The
same inputs always produce the same outputs.

## Install

```sh
pip install m68hc11
```

The import name is `m68hc11`:

```python
import m68hc11
cpu = m68hc11.HC11()
```

Requires Python 3.9+.

## Quick start

```python
import m68hc11

cpu = m68hc11.HC11()

# LDAA #$07 ; LDAB #$06 ; MUL ; RTS   ->  D = 7*6 = 42
cpu.load(bytes([0x86, 0x07, 0xC6, 0x06, 0x3D, 0x39]), 0x2000)

state = cpu.call(0x2000)
print(state.d)        # 42
print(state.cycles)   # 2 + 2 + 10 + 5 = 19 bus cycles
```

## Hitachi HD6301 / HD6303 mode

`m68hc11.HD6303` is a drop-in CPU variant for the Hitachi **HD6301 / HD6303**
(the CMOS 6801/6803 core Yamaha and others used throughout the 1980s). Same API
as `HC11` — only the instruction decode changes:

- **Adds** the six Hitachi instructions: `XGDX` (`$18`), `SLP` (`$1A`), and the
  `AIM` / `OIM` / `EIM` / `TIM` immediate-with-memory bit operations. On this
  core `$18` and `$1A` are ordinary opcodes, **not** the 68HC11 prefix bytes.
- **Removes** the 68HC11-only opcodes (the `$00` TEST opcode, `IDIV` / `FDIV`,
  the `BSET` / `BCLR` / `BRSET` / `BRCLR` bit ops, `STOP`, and the Y register
  with its `$18` / `$1A` / `$CD` prefix pages). Those bytes raise
  `IllegalOpcode` rather than silently mis-decoding as their HC11 meaning.

```python
import m68hc11

cpu = m68hc11.HD6303()

# AIM #$F0,$40 ; LDX #$0040 ; RTS   -> clears the low nibble of [$40], X=$0040
cpu.load(bytes([0x71, 0xF0, 0x40, 0xCE, 0x00, 0x40, 0x39]), 0x0200)
cpu.write(0x0040, 0x3C)
state = cpu.call(0x0200)
print(hex(cpu.read(0x0040)[0]))   # 0x30
print(hex(state.x))               # 0x40
```

Decode and functional execution (registers, memory, condition codes) are
accurate. Cycle counts for the shared 6801 base opcodes are inherited from the
68HC11 table and are *not* guaranteed cycle-exact for the 6303; the six
Hitachi-specific opcodes carry their documented HD6303 cycle counts.

## The `call()` harness

`call()` pushes a sentinel return address, sets `PC` and any argument registers,
then single-steps until the routine's final `RTS` pops the sentinel (or until
`max_steps`). It returns a `State` snapshot of every register, flag, and the
cycles elapsed *during the call*.

```python
cpu = m68hc11.HC11()

# Double the byte at $00C0 in place and also return it in A.
#   LDAA $C0 ; ASLA ; STAA $C0 ; RTS
cpu.load(bytes([0x96, 0xC0, 0x48, 0x97, 0xC0, 0x39]), 0x2000)
cpu.write(0x00C0, 0x09)

st = cpu.call(0x2000)
assert cpu.read(0x00C0) == b"\x12"
assert st.a == 0x12
assert st.cycles == 13
```

Pass inputs via keyword: `call(addr, a=…, b=…, d=…, x=…, y=…)`. Set the stack
pointer first (it defaults to `$01FF`) if your routine pushes.

## API

```python
class HC11:
    def __init__(self, *, ram_fill: int = 0x00) -> None: ...

    # memory
    def load(self, data: bytes, base: int) -> None      # raw blob copy (no hooks)
    def read(self, addr: int, n: int = 1) -> bytes
    def write(self, addr: int, data: bytes | int) -> None
    def read16(self, addr: int) -> int                  # big-endian
    def write16(self, addr: int, value: int) -> None
    def read8(self, addr: int) -> int                   # hook-aware
    def write8(self, addr: int, val: int) -> None       # hook-aware

    # registers (plain attributes)
    a; b; x; y; sp; pc; ccr
    d                                                   # property (a<<8)|b, settable
    s; x_irq; h; i; n; z; v; c                          # flag bool properties
    def set_regs(self, *, a=None, b=None, d=None, x=None,
                 y=None, sp=None, pc=None, ccr=None) -> None

    # I/O hooks (inclusive ranges)
    def on_read(self, start: int, end: int, fn) -> None   # fn(addr) -> int
    def on_write(self, start: int, end: int, fn) -> None  # fn(addr, value)

    # execution
    cycles: int                                         # cumulative bus cycles
    def reset_cycles(self) -> None
    def step(self) -> Step                               # execute ONE instruction
    def run(self, *, max_steps=None, until_pc=None,
                  until_cycles=None) -> str              # returns stop reason
    def call(self, addr, *, a=None, b=None, d=None, x=None, y=None,
             max_steps=1_000_000, sentinel=0xFFFE) -> State

    # tracing / debug
    def set_trace(self, fn) -> None                     # fn(Step) before each instruction
    def disassemble(self, addr: int) -> Step            # decode without executing

    # optional timer + interrupts (spec section 6)
    def enable_timer(self, base: int = 0x1000) -> None
    def run_until_irq(self, vector, count=1, max_cycles=10_000_000) -> str
```

`Step` (returned by `step`, passed to the trace fn): `pc`, `opcode` bytes, a
`mnemonic` disassembly string, `cycles` (this instruction), and the resolved
`ea` (effective address) if any.

`State` (returned by `call`): all registers + individual flags + `cycles`
elapsed during the call.

`run()` stop reasons: `"max_steps"`, `"until_pc"`, `"until_cycles"`, `"wai"`,
`"stop"`, `"illegal"`.

### I/O hooks

Plain reads/writes hit the backing array unless an address is covered by a hook,
in which case the hook is called instead — letting peripheral registers be
stubbed or modelled without touching the core:

```python
regs = {0x1004: 0x37}
cpu.on_read(0x1000, 0x103F, lambda addr: regs.get(addr, 0xFF))
cpu.on_write(0x1000, 0x103F, lambda addr, val: regs.__setitem__(addr, val))
```

### Tracing

```python
cpu.set_trace(lambda step: print(f"{step.pc:04X}: {step.mnemonic}"))
cpu.run(max_steps=20)
```

### Timer + interrupts (optional)

Layered on top of the core — the core works without ever calling
`enable_timer()`. When enabled, a 16-bit free-running counter `TCNT` advances by
the prescale selected in `TMSK2`, output compares latch flags in `TFLG1`, and
enabled+unmasked sources deliver vectored interrupts (full HC11 context stacked).
The on-chip register block base defaults to `$1000` and is parameterizable.

```python
cpu = m68hc11.HC11()
cpu.enable_timer()                 # registers in the $1000 block
cpu.set_regs(pc=0x0100, sp=0x01FF, ccr=0x00)  # I clear -> interrupts enabled
cpu.write16(0x1018, 0x000A)        # TOC2 compare value
cpu.write8(0x1022, 0x40)           # TMSK1: enable the OC2 interrupt
cpu.write16(0xFFE6, 0x3000)        # OC2 interrupt vector -> handler
cpu.load(bytes([0x20, 0xFE]), 0x0100)  # BRA * (spin)

reason = cpu.run_until_irq(0xFFE6, count=1)
assert reason == "irq" and cpu.pc == 0x3000
```

## Correctness & validation

The test suite (`tests/`) encodes the spec's acceptance criteria:

1. **Flag / arithmetic self-test** — hand-checked sequences for the tricky cases
   (`MUL` C-bit, `ASLD`/`LSRD`, `DAA`, `ADDD`/`SUBD` overflow & carry,
   `NEG`/`COM` carry, signed branches, `ABX`/`XGDX`).
2. **Cross-validation routines** — small hand-assembled routines whose final
   registers, CCR, and *exact* cycle counts are computed by hand from the
   M68HC11 reference manual. These are good candidates to diff against
   `sim/m68hc11` (GNU binutils-gdb) or MAME.
3. **The `call()` harness** demonstrably running self-contained subroutines that
   read inputs from registers/memory and return results in `D`/`X`/memory with
   cycles reported.

Run them with:

```sh
pip install -e ".[test]"
pytest
```

### Notes on a couple of documented edge cases

- `DAA` leaves the `V` flag unchanged (the reference manual documents `V` as
  undefined after `DAA`).
- `IDIV`/`FDIV` leave `N` and `H` unaffected; divide-by-zero sets `C` and forces
  the quotient (`X`) to `$FFFF`.
- `TAP` can clear the `X` (XIRQ) mask but, per the hardware, cannot set it.

## Reference material

- **M68HC11 Reference Manual** (`M68HC11RM`) — instruction set, cycle-by-cycle
  operation, exact CCR effects.
- **M68HC11 Programming Reference Guide** (`M68HC11PG`) — compact opcode map for
  the page-1 / `$18` / `$1A` / `$CD` pages, lengths, and cycle counts.
- A device data sheet (e.g. **MC68HC11A8** / **MC68HC11F1**) for the `$1000`
  register block and the `$FFC0–$FFFF` vector table.

## Releasing (maintainers)

Tagged releases publish to PyPI automatically via **trusted publishing** (OIDC,
no API tokens). See [`.github/workflows/publish.yml`](.github/workflows/publish.yml).
To wire it up the first time, on PyPI add a *pending publisher* (or, after the
first manual upload, a trusted publisher) for project **`m68hc11`** with:

| Field            | Value             |
| ---------------- | ----------------- |
| Owner            | `anarkiwi`        |
| Repository       | `m68hc11`         |
| Workflow name    | `publish.yml`     |
| Environment name | `pypi`            |

Then publishing a GitHub Release builds the sdist + wheel and uploads them with
no secrets stored in the repo.

## License

Apache-2.0. See [LICENSE](LICENSE).
