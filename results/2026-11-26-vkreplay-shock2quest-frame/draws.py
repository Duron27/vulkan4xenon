#!/usr/bin/env python3
"""draws.py <draws.bin>: summarise the game's draw dump (engine/src/gfx/draw_dump.rs)."""
import struct, sys

data = open(sys.argv[1], 'rb').read()
at = 0


def u(n=1):
    global at
    v = struct.unpack_from('<%dI' % n, data, at)
    at += 4 * n
    return v if n > 1 else v[0]


def f(n):
    global at
    v = struct.unpack_from('<%df' % n, data, at)
    at += 4 * n
    return v


def blob():
    global at
    n = u()
    b = data[at:at + n]
    at += (n + 3) & ~3
    return b


draw = 0
while at < len(data):
    tag = data[at:at + 4].decode()
    at += 4
    if tag == 'FRAM':
        print('FRAM', u(5))
    elif tag == 'CAPB':
        print('CAPB', u(3))
    elif tag == 'CAPE':
        print('CAPE')
    elif tag == 'CLER':
        c, d = u(2)
        print('CLER color', c, 'depth', d, f(4), [x - (1 << 32) if x >= 1 << 31 else x for x in u(4)])
    elif tag == 'PROG':
        pid = u()
        vs = u(); at += 4 * vs
        fs = u(); at += 4 * fs
        nb = u()
        binds = [u(3) for _ in range(nb)]
        print('PROG', pid, 'vs', vs, 'fs', fs, 'words; bindings', binds)
    elif tag == 'DRAW':
        program, topo, count, stride = u(4)
        na = u()
        attrs = [u(4) for _ in range(na)]
        cull = u(2)
        dt, dw, dc, bl = u(4)
        bf = u(4)
        mask, db = u(2)
        bias = f(2)
        fmts = u(3)
        vp = f(6)
        sc = [x - (1 << 32) if x >= 1 << 31 else x for x in u(4)]
        verts = blob(); idx = blob(); b0 = blob(); b1 = blob()
        ns = u()
        samp = [u(7) for _ in range(ns)]
        pos = []
        if verts and not idx:
            for i in range(min(count, len(verts) // stride)):
                pos.append(struct.unpack_from('>3f', verts, i * stride))
        lo = [min(p[k] for p in pos) for k in range(3)] if pos else None
        hi = [max(p[k] for p in pos) for k in range(3)] if pos else None
        print(f'DRAW {draw}: prog {program} topo {topo} count {count} stride {stride} attrs {attrs} cull {cull} '
              f'depth t{dt} w{dw} op{dc} blend {bl} {bf} mask {mask:x} bias {db} {bias} fmts {fmts} vp {vp} sc {sc} '
              f'vbytes {len(verts)} ibytes {len(idx)} blocks {len(b0)}/{len(b1)} samplers {[(s[0], s[1], s[2], s[3], s[4], s[5], s[6]) for s in samp]} '
              f'pos {lo}..{hi}')
        draw += 1
    else:
        print('bad tag', repr(tag), 'at', at - 4)
        break
