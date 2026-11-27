# vkreplay: shock2quest's own frame, drawn again, hangs the GPU: the spotlight loop

shock2quest (Xbox 360 branch) hangs the GPU a frame or a few into earth.mis, RBBM_STATUS 0xEEA51110 with the ring read
to its end. vkhang's synthetic walls never did. vkreplay (`hang/replay.c`, `make XSDK=../xsdk vkreplay`) draws the
frame the game itself issued: the game writes it with `dump_draws.txt` (frame number) beside `shock2.xex`, into
`draws.bin` (shock2quest:engine/src/gfx/draw_dump.rs), and vkreplay issues those draws again, with the game's
pipeline state, vertices, uniform blocks and console shader bundle (`build/replay_shaders/`, the game's modules plus
the variants below). Textures are gradients at the game's sizes and levels. `draws.py` prints a dump.

Frame 1 of earth.mis: one pass (B8G8R8A8 + D24S8, 1280x720), two clears, 58 draws (54 world, 3 object, 1 HUD).
World draws: blend SRC_ALPHA/ONE_MINUS_SRC_ALPHA, depth test and write LESS, no culling, stride 56, 5 attributes.

## Runs (each phase 180 frames, stopping at the first that hangs)

| Phase | Result |
|---|---|
| the dump as it is (r1) | hang on frame 0, RBBM_STATUS EEA51110, the game's |
| no blending | hang |
| single draws 0, 42, 15, 12, 2 (r3) | 180 frames each, 2 ms |
| draw 1 alone (705 vertices, 6490 px) | hang |
| draw 1, no depth test or write | hang |
| draw 1 in eighths: vertices 0-87 / 88-175 (r5) | pass / hang |
| draw 1, vertices 87-176 in threes (r7) | every three triangles pass: it takes enough pixels, not one triangle |
| whole frame less its zero-area triangles | hang |
| draw 1, fragment shader `plain` (no bump, discard, shine or fog) (r8) | hang |
| draw 1, `flat`, `tex`, `noloop` (r9) | pass |
| draw 1 and all 54 world draws, `fullnoloop` (the game's shader less only the spotlight loop) (r10) | pass, 26 ms a frame for the 54 |
| draw 1, the game's shader with spotlightCount 6 (r11) | hang |
| draw 1 and all 54, `one` / `oneif` (light 0 unrolled, constant index; spotlightCount 0 and 1) (r12) | pass; 44 ms (count 0) and 60 ms (count 1) a frame for the 54 |

## What hangs

The world shader's spotlight loop, `for (int i = 0; i < int(spotlightCount); i++) spotlights += calculateSpotlight(i,
...)`, whatever its count (0 and 6 both hang), on enough pixels. The same body called once with a constant index
doesn't. The loop indexes the light arrays (`spotlightPos[i]` and five more) with its counter: constants addressed
relative to the loop register, which vkhang's six-pass loop (the same count, a body with no arrays) never did. That
is the next thing to take to xost: a minimal shader looping over a uniform array by its counter, on many pixels.
