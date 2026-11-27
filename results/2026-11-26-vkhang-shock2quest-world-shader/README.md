# vkhang: what shock2quest's world shader costs the GPU

`hang/main.c` (target `make vkhang`, not in `all`) draws shock2quest's world as the game does: the game's 56-byte
world vertex, its descriptor layout and `flip_y` push constant, a 4096x2048 lightmap atlas and a mipmapped diffuse,
and the game's own world modules, copied from its console bundle (`hang/shaders/game_*.spv`) or cut down from its
dumped GLSL through shock2quest's `console_link_files` test, which rebuilds the unmodified source byte for byte into
the bundle's modules. Each phase draws one full-screen wall for 120 frames and waits up to 30 s for each frame's
fence. Built at `2901d33` (with the working tree's uncommitted changes) against xsdk `ade2c81`, and run on the
user's console on 2026-11-26; `vkhang.log` is the last run.

## The game's GPU stalls weren't reproduced: the shader is just slow

The first build, on the harness's frame loop, failed the game's shader on 54 walls with `vkWaitForFences returned 2`.
That was the harness's 1 s fence wait: with a 30 s wait, every frame of 54 full-screen walls finished, each in
3.65 s, 54 times the 68 ms of one wall. No phase stopped the GPU.

## Time a full-screen layer takes, by fragment shader

The fence's time is quantized to about 2 ms. Registers and instruction slots are from each container's
`SQ_PROGRAM_CNTL` and size (`xenos-toolchain/xdis/xdis.py`).

| Fragment shader | ms a layer | Registers | Slots |
|---|---|---|---|
| flat colour | 2.0 | 1 | 3 |
| six-pass loop counted by a constant (unrolled) | 4.0 | 5 | 53 |
| six-pass loop counted by a uniform (255-pass hardware loop) | 10.0 | 11 | 37 |
| the game's world shader | 68.7 | 25 | 390 |
| ... less the bump | 66.0 | 25 | 331 |
| ... less the spotlight loop | 4.0 | 11 | 110 |
| ... less both | 4.0 | 11 | 110 |
| ... with one spotlight pass | 28.0 | 25 | 389 |
| ... one pass, no highlight | 28.0 | 25 | 381 |
| ... one pass, no cone | 24.0 | 23 | 329 |
| ... one pass, no `cos` in the cone | 28.0 | 25 | 378 |
| ... one pass, a plain distance falloff | 26.1 | 25 | 371 |
| ... spotlight loop unrolled to 6 | 86.0 | 37 | 1086 |

- **The spotlight loop is the cost:** 64 of the 68.7 ms. A light's body, inlined with every branch evaluated as
  xost evaluates an if on data, is about 280 slots, and each pass runs it whether the light is lit or not (the
  phases' lights all have intensity 0).
- **The hardware loop does leave early:** the bare six-pass loop on a uniform count costs 6 ms over its unrolled
  form, not 255 passes' worth, and the game's loop at 6 passes costs about what 6 bodies do.
- **Unrolling is worse:** 86 ms, 37 registers.
- **No single part of a light dominates:** the highlight, the `cos`, the falloff and the cone each save 0-4 ms.

## What shock2quest changed

Its lightmap material loops to a `spotlightCount` uniform, the slots in use, instead of `SHOCK_COUNT(6)`; with no
light lit the world shader costs what it does without the loop.
