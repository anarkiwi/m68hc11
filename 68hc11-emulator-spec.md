# Motorola 68HC11 emulator — build specification

Build an instruction-level emulator for the **Motorola 68HC11** 8-bit
microcontroller core. The goal is to *execute real ROM machine code* and read
back exactly what it computes (register values, memory writes, elapsed cycles),
so that arithmetic/timing done in firmware can be derived by running it rather
than by reading a disassembly.

Deliver a **single self-contained Python 3 module** (`hc11.py`), no third-party
dependencies, with the API in §4. Correctness and a clean function-call harness
matter more than speed.

---

## 1. Target CPU

- Motorola **68HC11** core: 8-bit, **big-endian**, single 64 KB address space
  (`$0000–$FFFF`), von Neumann.
- Registers: **A**, **B** (8-bit) combinable as **D = A:B** (16-bit, A = high);
  **X**, **Y** (16-bit index); **SP** (16-bit stack pointer); **PC** (16-bit);
  **CCR** (8-bit condition codes: `S X H I N Z V C`, bit7→bit0).
- Stack grows **downward**; `PSH` writes then decrements SP, `PUL` increments
  then reads (standard 68HC11 stack discipline). Subroutine calls push the
  **return address** (PC) low-byte-first per the HC11 convention — implement
  exactly per the reference manual so `JSR`/`BSR`/`RTS`/`RTI` round-trip.

Only the CPU **core + instruction set** is mandatory. On-chip peripherals are
optional (see §6) and must be accessible via I/O hooks (§4) when not modelled.

---

## 2. Instruction set (mandatory, complete)

Implement the **full 68HC11 instruction set**, all addressing modes, with
**correct condition-code flag effects for every instruction**:

- **Addressing modes:** inherent, immediate (8- and 16-bit), direct (zero page),
  extended (16-bit absolute), indexed-X, indexed-Y, relative (8-bit signed
  branches).
- **Opcode pages:** page 1 (un-prefixed) **and** the prefixed pages reached via
  `$18` (Y-indexed / Y variants of LDX→LDY etc.), `$1A` (e.g. `CPD`, Y forms),
  and `$CD` (X/Y mixed forms). Decode prefix → real opcode → mode correctly,
  including the correct instruction length and cycle count per prefixed form.
- **Bit-manipulation:** `BSET`, `BCLR` (direct and indexed) and `BRSET`,
  `BRCLR` (direct and indexed, with their trailing relative branch byte).
- **Arithmetic/logic with exact flags**, paying special attention to the
  fiddly ones:
  - `MUL` (16-bit unsigned product in D; **C = bit 7 of the result**),
  - `IDIV`, `FDIV` (quotient→X, remainder→D; flag rules),
  - `DAA`,
  - all shifts/rotates `ASL/ASR/LSR/ROL/ROR/ASLD/LSRD` (C from shifted-out bit; V = N⊕C),
  - `ABA`, `SBA`, `CBA`, `NEG`, `COM` (C set), `CPX/CPY/CPD`,
  - `ADDD/SUBD` 16-bit flags,
  - `ABX`, `ABY` (X/Y += B, unsigned, **no flags**),
  - `XGDX`, `XGDY`, `TAP`, `TPA`, `TSX/TSY`, `TXS/TYS`, `INS`, `DES`.
- `NOP`, `TEST`, `STOP`, `WAI`, `SWI`, `RTI` present (even if `STOP/WAI` just
  halt/raise in core-only mode).
- **Unimplemented or illegal opcode → raise an explicit exception** (PC + byte
  in the message). Never silently skip or guess. This is important: silent
  mis-decode is the exact failure mode this emulator exists to avoid.

A correct per-opcode **cycle count** must be attached to each instruction (it
varies by addressing mode and prefix) and accumulated — see §5. Cycle counts are
in the reference manual (§7).

---

## 3. Memory model

- A flat 64 KB `bytearray`. Default fill configurable (`0x00` or `0xFF`).
- `load(data, base)` copies a raw binary blob to `base`.
- Plain reads/writes hit the array **unless** an address is covered by an I/O
  hook (§4), in which case the hook is called instead. This lets memory-mapped
  peripheral registers be stubbed or modelled without touching the core.
- No region is implicitly read-only; the caller decides what is "ROM" by simply
  not writing to it. (Self-modifying / RAM-copied code must work, so do not
  enforce write protection.)

---

## 4. API (this is the part I will actually use — keep it precise)

```python
class HC11:
    def __init__(self, *, ram_fill: int = 0x00) -> None: ...

    # --- memory ---
    def load(self, data: bytes, base: int) -> None
    def read(self, addr: int, n: int = 1) -> bytes
    def write(self, addr: int, data: bytes | int) -> None
    def read16(self, addr: int) -> int          # big-endian
    def write16(self, addr: int, value: int) -> None

    # --- registers (plain attributes) ---
    a: int; b: int; x: int; y: int; sp: int; pc: int; ccr: int
    @property
    def d(self) -> int                          # (a<<8)|b, settable
    # individual flags as bool properties: h, i, n, z, v, c (and s, x_irq)

    def set_regs(self, *, a=None, b=None, d=None, x=None, y=None,
                 sp=None, pc=None, ccr=None) -> None

    # --- I/O hooks for memory-mapped registers / open bus ---
    # called for any access whose address is in [start, end]; return an int for
    # reads, ignore return for writes. Lets peripherals be stubbed/modelled.
    def on_read(self, start: int, end: int, fn) -> None     # fn(addr) -> int
    def on_write(self, start: int, end: int, fn) -> None     # fn(addr, value)

    # --- execution ---
    cycles: int        # cumulative bus (E-clock) cycles since reset/clear
    def step(self) -> Step          # execute ONE instruction
    def run(self, *, max_steps: int | None = None,
                  until_pc: int | None = None,
                  until_cycles: int | None = None) -> str   # returns stop reason

    # --- the harness I care about most: run a subroutine as a function ---
    # Push a sentinel return address, set PC=addr and the given args, then run
    # until the routine's final RTS pops the sentinel (PC == sentinel), or until
    # max_steps. Returns the full post-state. Use this to call a leaf/!leaf
    # routine and read its result out of D / X / memory.
    def call(self, addr: int, *, a=None, b=None, d=None, x=None, y=None,
             max_steps: int = 1_000_000, sentinel: int = 0xFFFE) -> State

    # --- tracing / debug ---
    def set_trace(self, fn) -> None   # fn(Step) called before each instruction
```

- **`Step`** (returned by `step` / passed to the trace fn): `pc`, `opcode`
  bytes, a textual **mnemonic+operand disassembly** string, `cycles`
  (this instruction), and the resolved effective address if any.
- **`State`** (returned by `call`): a snapshot of all registers + flags +
  `cycles` elapsed during the call.
- `run` stop reasons: `"max_steps"`, `"until_pc"`, `"until_cycles"`, `"wai"`,
  `"stop"`, `"illegal"`.
- The trace disassembly text should be good enough to eyeball control flow; it
  does **not** need to match any particular assembler's syntax. (Building the
  decode/disassembly table straight from the opcode map in §7 is fine.)

Determinism: pure functions only — **no wall-clock, no RNG, no global state.**
The same inputs must always produce the same outputs (so runs are reproducible).

---

## 5. Cycle counting (required)

The emulator must track **cumulative bus cycles** (`cycles`) by adding each
instruction's documented cycle count as it executes. This is what makes timing
derivable: with a known bus-clock frequency, elapsed cycles ⇒ elapsed time.
Cycle counts per opcode/addressing mode are tabulated in the reference manual
(§7); transcribe them. Get the prefixed-page (`$18/$1A/$CD`) extra cycle right.

`step()` returns the cycle cost of the instruction it ran; `cycles` is the
running total; provide a way to reset it.

---

## 6. On-chip timer + interrupts (optional, second priority)

Strongly useful but not required for a first version. If implemented, model the
**68HC11 main timer** and interrupt delivery so interrupt-driven firmware can be
run and its timing measured:

- 16-bit **free-running counter `TCNT`** that increments every *N* bus cycles,
  where *N* is the prescale (1/4/8/16) selected by `TMSK2[PR1:PR0]`.
- **Output-compare** registers `TOC1–TOC5` with compare-match logic; on match,
  set the corresponding flag in **`TFLG1`**, and if enabled in **`TMSK1`** and
  interrupts are unmasked (`CCR.I = 0`), **take the interrupt**: push the CPU
  context (CCR, then B, A, X, Y per HC11 order), set `I`, and load `PC` from the
  vector.
- **Vector table at `$FFC0–$FFFF`** (16-bit big-endian pointers). Make the
  on-chip register **base address parameterizable** (it differs by package /
  `INIT` register on real parts; default the common `$1000` block) so the same
  core serves any 68HC11 variant.
- Provide a way to run "for K bus cycles" (or "until N interrupts of vector V
  have fired") and observe state — this is what lets interrupt-rate / timer-reload
  behaviour be measured directly.

Keep this layered **on top of** the core so the core (§2–§5) is usable alone.

---

## 7. Where to get the material / docs

Authoritative Motorola/Freescale/NXP documents (search by the codes — they are
freely available as PDFs):

- **M68HC11 Reference Manual** — `M68HC11RM/D` (a.k.a. *M68HC11RM Rev …*). The
  definitive source: full instruction set, every addressing mode, **cycle-by-cycle
  operation**, and exact **condition-code (CCR) effects** per instruction. This
  is the primary reference for §2 and §5.
- **M68HC11 Programming Reference Guide** — `M68HC11PG/D` / `M68HC11PRG`. Compact
  opcode map (page 1 + `$18/$1A/$CD` prefix pages), instruction lengths, and
  cycle counts — ideal for building the decode + cycle tables.
- **A device data sheet** for the on-chip register map (§6), e.g.
  **`MC68HC11A8`** or **`MC68HC11F1`** Technical Data — gives the `$1000`-block
  register addresses (`TCNT $100E`, `TOC1–5`, `TMSK1/2`, `TFLG1/2`, etc.) and the
  reset vector layout `$FFC0–$FFFF`.

Existing open-source 68HC11 cores to **study and to cross-validate against**
(run the same code on both and diff register/flag/cycle results):

- **GNU binutils-gdb** `sim/m68hc11` (the GDB instruction-set simulator for
  `m68hc11-elf` / `m6811`). A complete, well-tested reference implementation.
- **MAME** CPU core for the 6800 family including HC11 (`src/devices/cpu/m6800/`,
  the `m68hc11` device) — another careful reference.
- Smaller standalone simulators (e.g. **THRSim11**, **2511em**) for spot checks.

---

## 8. Validation / acceptance

Before trusting it, the emulator must pass:

1. A **flag/arith self-test**: a handful of hand-checked sequences exercising the
   tricky cases (`MUL` C-bit, `ASLD`/`LSRD`, `DAA`, `ADDD`/`SUBD` overflow/carry,
   `NEG`/`COM` carry, signed branches, `ABX`/`XGDX`).
2. **Cross-validation**: assemble (or hand-assemble) a few small routines, run
   them on `sim/m68hc11` (or MAME) and on this emulator, and assert identical
   final registers, CCR, and cycle counts.
3. The **`call()` harness** demonstrably runs a self-contained subroutine that
   reads an input from a register and/or a memory location, and returns its
   result in `D`/`X`/memory, with `cycles` reported — using only this API.

Acceptance = (1)–(3) pass and any unimplemented opcode raises rather than
mis-decodes.
