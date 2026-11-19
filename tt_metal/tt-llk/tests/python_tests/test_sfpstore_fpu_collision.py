# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Reproducer (Blackhole): an SFPU store and a Matrix Unit (FPU) Dst write that land in the same Dst half in the same cycle,
# on two rows that share a half-bank, lose the FPU write. Only a store scheduled by SFPLOADMACRO can do this; a plain
# SFPSTORE in the same position never does. DEST_ACCESS_CFG remap_addrs / swizzle_32b are not required: they only change
# which row bit picks the half-bank (32-bit Dst: bit 3 with both off, bit 4 remap, bit 2 xor bit 4 swizzle, bit 2 xor
# bit 5 both; 16-bit: bit 4, or bit 5 with remap).
# Main case: test_interleaved[...cfg0...loadmacro...S64...] loses one ELWADD's 8 rows; the sfpstore arm loses nothing.
# The harness reseeds every test with one fixed seed; set PROBE_FRESH_SEED=1 for fresh inputs per test (a dropped write
# can otherwise read back stale data that equals the expected value). See sources/sfpstore_fpu_collision_test.cpp.

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from ttexalens.tt_exalens_lib import read_words_from_device
from helpers.test_config import TestConfig
from helpers.llk_params import BlocksCalculationAlgorithm, DestAccumulation, DestSync, Tilize
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_INDEX,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)


@pytest.mark.parametrize("trial", range(3))
@pytest.mark.parametrize("reps", [4])
@pytest.mark.parametrize("hammer", [0, 1, 2, 3, 4], ids=["quiet", "sfpstore", "poison", "loadmacro", "macropoison"])
@pytest.mark.parametrize("remap", [0, 1], ids=["linear", "remap"])
def test_sfpstore_fpu_collision(remap, hammer, reps, trial):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole probe")
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [256, 256]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard
    )
    mode = remap | ((hammer >= 1) << 1) | ((hammer in (2, 4)) << 2) | ((hammer in (3, 4)) << 3) | (reps << 4)
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=None,
        test_config_kwargs={
            "test_name": "sources/sfpstore_fpu_collision_test.cpp",
            "formats": formats,
            "templates": [generate_input_dim(input_dimensions, input_dimensions), TILIZE(Tilize.No)],
            "runtimes": [
                DEST_INDEX(mode),
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(4),
                NUM_BLOCKS(num_blocks),
                NUM_TILES_IN_BLOCK(num_tiles_in_block),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A, formats.input_format, src_B, formats.input_format, formats.output_format,
                tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A, num_faces=4,
            ),
            "dest_acc": dest_acc,
            "unpack_to_dest": False,
        },
    )
    res = configuration.run().result
    got = torch.tensor(res, dtype=torch.float32).flatten()
    want = src_A.to(torch.bfloat16).to(torch.float32).flatten()[: got.numel()]
    bad = (got != want).nonzero().flatten()
    per_tile = {}
    for i in bad.tolist():
        per_tile[i // 1024] = per_tile.get(i // 1024, 0) + 1
    print(
        f"\nCOLLISION PROBE remap={remap} hammer={hammer} reps={reps} blocks={num_blocks}x{num_tiles_in_block}: "
        f"bad={bad.numel()}/{got.numel()} tiles={sorted(per_tile.items())[:16]} zeros_among_bad={(got[bad] == 0).sum().item()}"
    )
    m = read_words_from_device(addr=0x110000, word_count=num_blocks, location=TestConfig.TENSIX_LOCATION)
    p = read_words_from_device(addr=0x111000, word_count=3 * num_blocks, location=TestConfig.TENSIX_LOCATION)
    t0 = p[0]
    overl = []
    for b in range(num_blocks - 1):
        h0, h1 = p[3 * b] - t0, p[3 * b + 1] - t0
        m0, w1 = m[b + 1] - t0, p[3 * (b + 1)] - t0  # MATH starts block b+1 .. PACK sees it done
        overl.append((b, m0, w1, h0, h1, max(0, min(h1, w1) - max(h0, m0))))
    print("  block: math_b+1_start math_b+1_done_seen hammer_start hammer_end overlap(cycles)")
    for o in overl[:4]:
        print("  ", o)
    print(f"  total overlap cycles={sum(o[5] for o in overl)} over {len(overl)} blocks")
    if hammer in (2, 4):
        assert bad.numel() > 0, "poison stores never reached the packed rows"
        return
    assert bad.numel() == 0, f"{bad.numel()} wrong datums"


@pytest.mark.parametrize("nops", range(0, 12))
@pytest.mark.parametrize("row", [72, 64, 0, 56], ids=["S72", "S64", "S0", "S56"])
@pytest.mark.parametrize("plain", [0, 1, 2], ids=["loadmacro", "sfpstore", "macrocycles"])
def test_same_half_collision(plain, row, nops):
    _same_half(plain, row, nops, 0)


@pytest.mark.parametrize("d", range(1, 8))
@pytest.mark.parametrize("knop", range(0, 4))
@pytest.mark.parametrize("row", [64, 72], ids=["S64", "S72"])
@pytest.mark.parametrize("plain", [0, 1], ids=["loadmacro", "sfpstore"])
@pytest.mark.parametrize("remap", [0, 1], ids=["cfg0", "cfg3"])
def test_interleaved(remap, plain, row, knop, d):
    _same_half(plain, row, knop | (d << 2), (1 << 26) | remap)


def _same_half(plain, row, nops, extra, fp32=True, tag=None, golden_b=False, u2d=False):
    """MATH: SFPU write to row S of its own half, released k NOPs after the tile's FPU (ELWADD) writes."""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole probe")
    formats = InputOutputFormat(DataFormat.Float32 if u2d else DataFormat.Float16_b, DataFormat.Float32 if fp32 else DataFormat.Float16_b)
    dest_acc = DestAccumulation.Yes if fp32 else DestAccumulation.No
    input_dimensions = [256, 256]
    import os  # fresh inputs per variant: the plugin reseeds identically, so stale Dst could mask a dropped write
    if os.environ.get("PROBE_FRESH_SEED"):
        _seed = int.from_bytes(os.urandom(4), "little")
        torch.manual_seed(_seed)
        print(f"\nPROBE_SEED {_seed}")
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    if plain == 2 and nops > 7:
        pytest.skip("delay is 3 bits")
    mode = 0x100 | ((plain == 1) << 9) | (nops << 10) | (row << 15) | ((plain == 2) << 25) | extra
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=None,
        test_config_kwargs={
            "test_name": "sources/sfpstore_fpu_collision_test.cpp",
            "formats": formats,
            "templates": [generate_input_dim(input_dimensions, input_dimensions), TILIZE(Tilize.No)],
            "runtimes": [
                DEST_INDEX(mode),
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(4),
                NUM_BLOCKS(tile_cnt_A),
                NUM_TILES_IN_BLOCK(1),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A, formats.input_format, src_B, formats.input_format, formats.output_format,
                tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A, num_faces=4,
            ),
            "dest_acc": dest_acc,
            "unpack_to_dest": u2d,
        },
    )
    res = configuration.run().result
    got = torch.tensor(res, dtype=torch.float32 if fp32 else torch.bfloat16).to(torch.float32).flatten()
    want = (src_B if golden_b else src_A)
    want = (want.to(torch.float32) if u2d else want.to(torch.bfloat16).to(torch.float32)).flatten()[: got.numel()]
    bad = (got != want).nonzero().flatten()
    tiles = sorted({i // 1024 for i in bad.tolist()})
    pos = sorted({i % 1024 for i in bad.tolist()})
    print(
        f"\nSAME-HALF {'MATRIX ' + tag + ' ' if tag else ('INTERLEAVED ' if extra else '')}cfg={3 if extra & 1 else 0} plain={plain} S={row} nops={nops} (knop={nops & 3} d={nops >> 2}): bad={bad.numel()}/{got.numel()} in {len(tiles)}/{tile_cnt_A} tiles"
        f" positions={pos[:8]}{'...' if len(pos) > 8 else ''} zeros={(got[bad] == 0).sum().item()}"
    )
    assert bad.numel() == 0, f"{bad.numel()} wrong datums in {len(tiles)} tiles"


CFG_BITS = {"cfg0": 0, "remap": 1, "swizzle": 1 << 28, "cfg3": 1 | (1 << 28)}


@pytest.mark.parametrize("d", range(1, 8))
@pytest.mark.parametrize("knop", range(0, 4))
@pytest.mark.parametrize("row", [64, 72, 80, 88], ids=["S64", "S72", "S80", "S88"])
@pytest.mark.parametrize("plain", [0, 1], ids=["loadmacro", "sfpstore"])
@pytest.mark.parametrize("cfg", list(CFG_BITS))
@pytest.mark.parametrize("writer", ["elwadd32", "mova2d16"])
def test_matrix(writer, cfg, plain, row, knop, d):
    """Interleaved probe over FPU writer (ELWADD into 32-bit Dst, MOVA2D into 16-bit Dst) and each DEST_ACCESS_CFG."""
    extra = (1 << 26) | CFG_BITS[cfg] | ((writer == "mova2d16") << 27)
    _same_half(plain, row, knop | (d << 2), extra, fp32=(writer == "elwadd32"), tag=f"{writer} {cfg}")


@pytest.mark.parametrize("d", range(1, 8))
@pytest.mark.parametrize("knop", range(0, 4))
@pytest.mark.parametrize("row", [64, 80], ids=["S64", "S80"])
@pytest.mark.parametrize("plain", [0, 1], ids=["loadmacro", "sfpstore"])
@pytest.mark.parametrize("cfg", list(CFG_BITS))
def test_movb2d(cfg, plain, row, knop, d):
    """Interleaved probe with MOVB2D (16-bit Dst, 4 rows per write) as the FPU writer."""
    extra = (1 << 26) | CFG_BITS[cfg] | (1 << 29)
    _same_half(plain, row, knop | (d << 2), extra, fp32=False, tag=f"movb2d16 {cfg}", golden_b=True)


@pytest.mark.parametrize("stallwait", [0, 1], ids=["nowait", "waitsfpu"])
@pytest.mark.parametrize("d", [1, 7])
def test_waitsfpu_pending_store(stallwait, d):
    """Macro store on row 0 (loaded before the tile, stored after ELWADD d): stale row 0 = the store fired after the
    FPU write. With a STALLWAIT(MATH, WAIT_SFPU) right after the SFPLOADMACRO: a hang means WAIT_SFPU waits for the
    pending store; stale row 0 means it does not."""
    _same_half(0, 0, 0 | (d << 2), (1 << 26) | (stallwait << 30), tag=f"waitsfpu={stallwait}")


@pytest.mark.parametrize("nops", range(0, 8))
@pytest.mark.parametrize("row", [64, 72, 0, 8], ids=["S64", "S72", "S0", "S8"])
@pytest.mark.parametrize("plain", [0, 1], ids=["loadmacro", "sfpstore"])
def test_unpack_to_dest(plain, row, nops):
    """Float32 unpack-to-dest tile, then (right after the LLK datacopy returns) an identity SFPU write of row S in the
    same half. Rows 0/8 are inside the tile: a wrong value there also shows a read-before-unpack race."""
    if plain == 0 and nops > 7:
        pytest.skip("delay is 3 bits")
    extra = 1 << 7
    _same_half(plain, row, nops, extra, fp32=True, tag="u2d", u2d=True)


@pytest.mark.parametrize("trial", range(3))
@pytest.mark.parametrize("zrep", [0, 64, 255], ids=["noextra", "zr256", "zr1020"])
@pytest.mark.parametrize("nops", [0, 1, 2, 4, 8, 16, 32, 63])
def test_zeroacc_flag_vs_sfpu_store(nops, zrep, trial):
    """SFPU-only fill of each fresh half with 10.0 while PACK ZEROACC-clears the other half. Output must be all 10.0."""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole probe")
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
    input_dimensions = [256, 256]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format, input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format, input_dimensions_B=input_dimensions,
    )
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=None,
        test_config_kwargs={
            "test_name": "sources/sfpstore_fpu_collision_test.cpp",
            "formats": formats,
            "templates": [generate_input_dim(input_dimensions, input_dimensions), TILIZE(Tilize.No)],
            "runtimes": [
                DEST_INDEX((1 << 6) | (nops << 10) | (zrep << 16)),
                TILE_COUNT(tile_cnt_A), NUM_FACES(4), NUM_BLOCKS(tile_cnt_A), NUM_TILES_IN_BLOCK(1), LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A, formats.input_format, src_B, formats.input_format, formats.output_format,
                tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A, num_faces=4,
            ),
            "dest_acc": DestAccumulation.Yes,
            "unpack_to_dest": False,
        },
    )
    got = torch.tensor(configuration.run().result, dtype=torch.float32).flatten()
    zeros = (got == 0).nonzero().flatten()
    other = ((got != 0) & (got != 10.0)).sum().item()
    tiles = sorted({i // 1024 for i in zeros.tolist()})
    print(f"\nZEROFLAG nops={nops}: zeros={zeros.numel()}/{got.numel()} in {len(tiles)} tiles {tiles[:10]} "
          f"pos={sorted({i % 1024 for i in zeros.tolist()})[:8]} other_wrong={other}")
    n = tile_cnt_A
    mt = read_words_from_device(addr=0x112000, word_count=2 * n, location=TestConfig.TENSIX_LOCATION)
    zt = read_words_from_device(addr=0x113000, word_count=n, location=TestConfig.TENSIX_LOCATION)
    # PACK's ZEROACC for block b (issued just after zt[b]) vs MATH's store window for block b+1
    pk = read_words_from_device(addr=0x111000, word_count=3 * n, location=TestConfig.TENSIX_LOCATION)
    # PACK's ZEROACC stretch for block b runs from after its pack (pk[3b+1]) to zt[b]; MATH's fill of block b+1 is mt
    inside = sum(1 for b in range(n - 1) if max(mt[2 * (b + 1)], pk[3 * b + 1]) < min(mt[2 * (b + 1) + 1], zt[b]))
    if trial == 0:
        for b in range(1, 5):
            print(f"  RAW b={b}: math_fill[{mt[2*b]-mt[0]}..{mt[2*b+1]-mt[0]}]  pack_after_pack(b-1)={pk[3*(b-1)+1]-mt[0]}  zeroacc(b-1)~{zt[b-1]-mt[0]}  zeroacc(b)~{zt[b]-mt[0]}")
    print(f"  zrep={zrep}: ZEROACC stretch overlaps MATH's SFPU store window in {inside}/{n - 1} blocks; "
          f"example b=5: store window {mt[12]}..{mt[13]}, ZEROACC at ~{zt[5]}")
    assert zeros.numel() == 0 and other == 0


@pytest.mark.parametrize("trial", range(20))
@pytest.mark.parametrize("nops", [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 63])
def test_zeroacc_shipped_timing(nops, trial):
    """Shipped LLK timing (one ZEROACC per release, nothing forced): does an SFPU-only fill ever lose values?"""
    test_zeroacc_flag_vs_sfpu_store(nops, 0, trial)


@pytest.mark.parametrize("trial", range(5))
@pytest.mark.parametrize("zrep", [0, 64, 255], ids=["noextra", "zr256", "zr1020"])
@pytest.mark.parametrize("fp32", [1, 0], ids=["elwadd32", "mova2d16"])
def test_zeroacc_vs_fpu_datacopy(fp32, zrep, trial):
    """MATH: normal LLK datacopy (FPU writes) of 4 tiles per block into its half; PACK: after packing, ZEROACC CLR_HALF on
    its own half repeated zrep*4 times. A 0 in the output = an FPU write's zero-flag clear lost to the other half's ZEROACC."""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole probe")
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32 if fp32 else DataFormat.Float16_b)
    dest_acc = DestAccumulation.Yes if fp32 else DestAccumulation.No
    input_dimensions = [256, 256]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format, input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format, input_dimensions_B=input_dimensions,
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard
    )
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=None,
        test_config_kwargs={
            "test_name": "sources/sfpstore_fpu_collision_test.cpp",
            "formats": formats,
            "templates": [generate_input_dim(input_dimensions, input_dimensions), TILIZE(Tilize.No)],
            "runtimes": [
                DEST_INDEX((1 << 5) | (zrep << 16)),
                TILE_COUNT(tile_cnt_A), NUM_FACES(4), NUM_BLOCKS(num_blocks), NUM_TILES_IN_BLOCK(num_tiles_in_block), LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A, formats.input_format, src_B, formats.input_format, formats.output_format,
                tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A, num_faces=4,
            ),
            "dest_acc": dest_acc,
            "unpack_to_dest": False,
        },
    )
    got = torch.tensor(configuration.run().result, dtype=torch.float32 if fp32 else torch.bfloat16).to(torch.float32).flatten()
    want = src_A.to(torch.bfloat16).to(torch.float32).flatten()[: got.numel()]
    bad = (got != want).nonzero().flatten()
    zeros = (got[bad] == 0).sum().item()
    print(f"\nZA-FPU {'elwadd32' if fp32 else 'mova2d16'} zrep={zrep} blocks={num_blocks}x{num_tiles_in_block}: bad={bad.numel()} zeros={zeros} tiles={sorted({i // 1024 for i in bad.tolist()})[:10]}")
    assert bad.numel() == 0


@pytest.mark.parametrize("trial", range(20))
@pytest.mark.parametrize("nops", [0, 1, 2, 4, 8, 16])
def test_zeroacc_single_store_shipped(nops, trial):
    """Shipped timing (one ZEROACC per release). MATH SFPU-writes only even columns, one store per 4-row group, into its
    fresh half; odd columns are never written and stay undefined. Even-column values must be 10.0 (a 0 = lost clear)."""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole probe")
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
    input_dimensions = [256, 256]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format, input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format, input_dimensions_B=input_dimensions,
    )
    configuration = create_test_or_perf_config(
        is_perf=False, run_types=None,
        test_config_kwargs={
            "test_name": "sources/sfpstore_fpu_collision_test.cpp", "formats": formats,
            "templates": [generate_input_dim(input_dimensions, input_dimensions), TILIZE(Tilize.No)],
            "runtimes": [DEST_INDEX((1 << 6) | (nops << 10) | (1 << 24)), TILE_COUNT(tile_cnt_A), NUM_FACES(4),
                         NUM_BLOCKS(tile_cnt_A), NUM_TILES_IN_BLOCK(1), LOOP_FACTOR(1)],
            "variant_stimuli": StimuliConfig(src_A, formats.input_format, src_B, formats.input_format, formats.output_format,
                tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A, num_faces=4),
            "dest_acc": DestAccumulation.Yes, "unpack_to_dest": False,
        },
    )
    got = torch.tensor(configuration.run().result, dtype=torch.float32).reshape(-1, 1024)
    tens = (got == 10.0).sum(dim=1)
    per_tile = sorted(set(tens.tolist()))
    print(f"\nSINGLE nops={nops}: 10.0 per tile values={per_tile}  tiles_short={(tens < tens.max()).sum().item()} max={tens.max().item()}")
    assert (tens == tens.max()).all()
