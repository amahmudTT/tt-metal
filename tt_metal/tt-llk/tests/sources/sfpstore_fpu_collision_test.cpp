// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Reproducer for lost Dst writes on Blackhole when an SFPU store and an FPU write hit the same Dst half-bank in the same
// cycle (see python_tests/test_sfpstore_fpu_collision.py for the rule and the main case). Several modes, selected by
// params.DST_INDEX bits as documented per block below. The first mode:
// MATH: datacopy Float16_b tiles into a 32-bit Dst (FPU writes), SyncHalf double buffering.
// PACK: after MATH hands over a half, PACK rewrites every row of that half with its own value
// (SFPLOAD + SFPSTORE, 32-bit) for a while, then packs it. Meanwhile MATH writes the next block into
// the other half. The rewrites cannot change any data, so a wrong output row means a lost FPU write.
// params.DST_INDEX selects the mode: bit 0 = remap_addrs+swizzle_32b on, bit 1 = PACK hammers, bit 2 = poison
// (the hammer stores a constant instead of each row's value: a check that the stores reach the packed rows),
// bit 3 = the hammer uses SFPLOADMACRO (macro 0: load, then store the same row one cycle later) instead of
// SFPLOAD + SFPSTORE, bits 4+ = hammer repetitions. Timestamps (wall clock, low word) go to L1: MATH 0x110000, PACK 0x111000.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const std::uint32_t num_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    const bool unpack_ab = (params.DST_INDEX & (1u << 29)) != 0; // MOVB2D writer: data comes in through SrcB
    if ((params.DST_INDEX & (1u << 6)) != 0)
    {
        return; // zero-flag test: MATH uses no source data
    }
    const bool u2d       = (params.DST_INDEX & (1u << 7)) != 0; // the writer is the unpacker, straight into Dst
    if (u2d)
    {
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
            0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);
        for (std::uint32_t i = 0; i < num_tiles; ++i)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
                L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
        }
        return;
    }
    if (unpack_ab)
    {
        _llk_unpack_AB_init_<BroadcastType::NONE>(ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4));
        for (std::uint32_t i = 0; i < num_tiles; ++i)
        {
            _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
        }
        return;
    }
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t loop = 0; loop < params.LOOP_FACTOR; ++loop)
    {
        for (std::uint32_t i = 0; i < num_tiles; ++i)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "llk_math_common.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const bool remap            = (params.DST_INDEX & 1) != 0;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // bit 0 = remap_addrs, bit 28 = swizzle_32b (each on its own; the old cfg3 arm passes both)
    const bool swizzle = (params.DST_INDEX & (1u << 28)) != 0;
    _llk_math_reconfig_remap_(false);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_remap_addrs_RMW>(remap ? 1 : 0);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_swizzle_32b_RMW>(swizzle ? 1 : 0);
    volatile std::uint32_t* ts = reinterpret_cast<volatile std::uint32_t*>(0x110000);
    // Same-half mode (bit 8): MATH itself puts an SFPU write to row S of its own half next to the tile's last FPU
    // write. bit 9 = plain SFPLOAD/SFPSTORE instead of SFPLOADMACRO, bits 10..14 = NOPs before the trigger, bits
    // 15..24 = S. The macro's store waits for the next SFPU instruction (the SFPNOP trigger); the plain store is the
    // trigger itself. Both rewrite S with its own value.
    const bool same_half   = (params.DST_INDEX & 0x100) != 0;
    const bool plain       = (params.DST_INDEX & 0x200) != 0;
    const std::uint32_t nk = (params.DST_INDEX >> 10) & 31;
    const std::uint32_t S  = (params.DST_INDEX >> 15) & 0x3ff;
    // bit 25: the macro's store counts cycles (delay = nk & 7) instead of waiting for the next SFPU instruction.
    const bool by_cycles = (params.DST_INDEX & (1u << 25)) != 0;
    if (same_half && !plain)
    {
        const std::uint32_t store_bits = by_cycles ? (((nk & 7) << 3) | 3) : ((1 << 3) | 3); // SFPSTORE, delay
        TTI_SFPLOADI(p_sfpu::LREG0, 10, 0);
        TT_SFPLOADI(p_sfpu::LREG0, 8, store_bits << 8);
        TTI_SFPCONFIG(0, 4 + 0, 0);
        if (by_cycles)
        {
            TTI_SFPCONFIG(0x010, 8, 1); // UsesLoadMod0ForStore for macro 0; every delay counts cycles
        }
        else
        {
            TTI_SFPCONFIG(0x810, 8, 1); // UsesLoadMod0ForStore for macro 0; the store sub-unit waits for elapsed instructions
        }
    }
    // bit 26: interleaved. The tile's eight ELWADDs are issued by hand; after each one come nk & 3 NOPs and an SFPU
    // instruction. Macro: store delay d = (nk >> 2) & 7 elapsed SFPU instructions, all eight slots SFPNOP. Plain: the
    // SFPSTORE takes slot d, the other slots are SFPNOP.
    // bit 6: zero-flag test. No FPU write: MATH fills tile 0 of its fresh (ZEROACC-cleared) half with 10.0 using
    // SFPSTOREs only, nk (bits 10..15) NOPs between stores, while PACK packs the other half and clears it with
    // ZEROACC CLR_HALF. A 0 in the output = that store's zero-flag clear was lost.
    if ((params.DST_INDEX & (1u << 6)) != 0)
    {
        const std::uint32_t zn = (params.DST_INDEX >> 10) & 63;
        for (int block = 0; block < params.NUM_BLOCKS; ++block)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            TTI_SFPLOADI(p_sfpu::LREG1, 0 /*fp16_b*/, 0x4120); // 10.0
            reinterpret_cast<volatile std::uint32_t*>(0x112000)[2 * block] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            // bit 24: one store per 4-row zero-flag group (addresses step 4: even columns only), else two (step 2)
            const std::uint32_t astep = (params.DST_INDEX & (1u << 24)) ? 4 : 2;
            for (std::uint32_t a = 0; a < 64; a += astep)
            {
                TT_SFPSTORE(p_sfpu::LREG1, 3, ADDR_MOD_7, a);
                for (std::uint32_t k = 0; k < zn; ++k)
                {
                    TTI_NOP;
                }
            }
            reinterpret_cast<volatile std::uint32_t*>(0x112000)[2 * block + 1] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        return;
    }
    // bit 7: unpack-to-dest writes the tile; right after the LLK datacopy returns, MATH rewrites row S of the same half
    // with its own value: plain SFPLOAD + SFPSTORE (bit 9) or SFPLOADMACRO with a cycle delay (nk & 7) after nk NOPs.
    if ((params.DST_INDEX & (1u << 7)) != 0)
    {
        const bool u_plain     = (params.DST_INDEX & 0x200) != 0;
        const std::uint32_t un = (params.DST_INDEX >> 10) & 31;
        const std::uint32_t uS = (params.DST_INDEX >> 15) & 0x3ff;
        if (!u_plain)
        {
            const std::uint32_t store_bits = ((un & 7) << 3) | 3;
            TTI_SFPLOADI(p_sfpu::LREG0, 10, 0);
            TT_SFPLOADI(p_sfpu::LREG0, 8, store_bits << 8);
            TTI_SFPCONFIG(0, 4 + 0, 0);
            TTI_SFPCONFIG(0x010, 8, 1);
        }
        for (int block = 0; block < params.NUM_BLOCKS; ++block)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, true>(
                0, formats.math, formats.math, ckernel::DEFAULT_TENSOR_SHAPE);
            for (std::uint32_t k = 0; k < (u_plain ? un : 0); ++k)
            {
                TTI_NOP;
            }
            if (u_plain)
            {
                TT_SFPLOAD(p_sfpu::LREG1, 3, ADDR_MOD_7, uS);
                TT_SFPSTORE(p_sfpu::LREG1, 3, ADDR_MOD_7, uS);
            }
            else
            {
                TT_SFPLOADMACRO((0 << 2) | p_sfpu::LREG1, 3, ADDR_MOD_7, uS);
            }
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        return;
    }
    const bool interleaved = (params.DST_INDEX & (1u << 26)) != 0;
    // bit 27: the FPU writer is MOVA2D (16-bit Dst) instead of ELWADD. SFPU accesses use the Dst width's mode.
    const bool mov_writer          = (params.DST_INDEX & (1u << 27)) != 0;
    // bit 29: the FPU writer is MOVB2D, 4 rows per instruction (16 per tile; SrcB from the AB unpack).
    const bool movb_writer         = (params.DST_INDEX & (1u << 29)) != 0;
    const std::uint32_t per_face   = movb_writer ? 4 : 2;
    const std::uint32_t step       = movb_writer ? 4 : 8;
    if (movb_writer)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 4},
            .dest = {.incr = 4},
        }
            .set(ADDR_MOD_4);
    }
    constexpr std::uint32_t SFPMOD = is_fp32_dest_acc_en ? 3 : 2;
    if (interleaved)
    {
        const std::uint32_t knop = nk & 3;
        const std::uint32_t d    = (nk >> 2) & 7;
        if (!plain)
        {
            const std::uint32_t store_bits = (d << 3) | 3;
            TTI_SFPLOADI(p_sfpu::LREG0, 10, 0);
            TT_SFPLOADI(p_sfpu::LREG0, 8, store_bits << 8);
            TTI_SFPCONFIG(0, 4 + 0, 0);
            TTI_SFPCONFIG(0x810, 8, 1);
        }
        for (std::uint32_t loop = 0; loop < params.LOOP_FACTOR; ++loop)
        {
            for (int block = 0; block < params.NUM_BLOCKS; ++block)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                if (plain)
                {
                    TT_SFPLOAD(p_sfpu::LREG1, SFPMOD, ADDR_MOD_7, S);
                }
                else
                {
                    TT_SFPLOADMACRO((0 << 2) | p_sfpu::LREG1, SFPMOD, ADDR_MOD_7, S);
                }
                if (params.DST_INDEX & (1u << 30))
                {
                    // bit 30: does a STALLWAIT on WAIT_SFPU wait for the macro's still-pending store? If it does, this
                    // deadlocks (the store needs a later SFPU instruction, which is behind the stalled ELWADDs).
                    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
                }
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                std::uint32_t slot = 0;
                for (std::uint32_t f = 0; f < 4; ++f)
                {
                    for (std::uint32_t i = 0; i < per_face; ++i)
                    {
                        if (movb_writer)
                        {
                            TTI_MOVB2D(0, 0, ADDR_MOD_4, p_movb2d::MOV_4_ROWS, 0);
                        }
                        else if (mov_writer)
                        {
                            TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
                        }
                        else
                        {
                            TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
                        }
                        for (std::uint32_t k = 0; k < knop; ++k)
                        {
                            TTI_NOP;
                        }
                        ++slot;
                        if (plain && slot == d)
                        {
                            TT_SFPSTORE(p_sfpu::LREG1, SFPMOD, ADDR_MOD_7, S - step * slot); // the Dst counter has moved one write's rows per FPU instruction
                        }
                        else
                        {
                            TTI_SFPNOP;
                        }
                    }
                    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
                }
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        return;
    }
    if (same_half)
    {
        for (std::uint32_t loop = 0; loop < params.LOOP_FACTOR; ++loop)
        {
            for (int block = 0; block < params.NUM_BLOCKS; ++block)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                if (plain)
                {
                    TT_SFPLOAD(p_sfpu::LREG1, 3, ADDR_MOD_7, S);
                }
                else
                {
                    TT_SFPLOADMACRO((0 << 2) | p_sfpu::LREG1, 3, ADDR_MOD_7, S);
                }
                _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                    0, formats.math, formats.math, ckernel::DEFAULT_TENSOR_SHAPE);
                for (std::uint32_t k = 0; k < (by_cycles ? 0 : nk); ++k)
                {
                    TTI_NOP;
                }
                if (plain)
                {
                    TT_SFPSTORE(p_sfpu::LREG1, 3, ADDR_MOD_7, S);
                }
                else
                {
                    TTI_SFPNOP;
                }
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        return;
    }
    for (std::uint32_t loop = 0; loop < params.LOOP_FACTOR; ++loop)
    {
        for (int block = 0; block < params.NUM_BLOCKS; ++block)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            ts[block] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            for (std::uint32_t t = 0; t < params.NUM_TILES_IN_BLOCK; ++t)
            {
                _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                    t, formats.math, formats.math, ckernel::DEFAULT_TENSOR_SHAPE);
            }
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

// One SFPLOAD + SFPSTORE pair per even Dst address, 32-bit (mode 3), ADDR_MOD_7 (no counter change).
template <bool POISON, bool MACRO, std::uint32_t ADDR, std::uint32_t COUNT>
inline __attribute__((always_inline)) void rewrite_rows()
{
    if constexpr (COUNT > 0)
    {
        if constexpr (MACRO)
        {
            TTI_SFPLOADMACRO((0 << 2) | p_sfpu::LREG0, 3, ADDR_MOD_7, ADDR);
        }
        else
        {
            if constexpr (!POISON)
            {
                TTI_SFPLOAD(p_sfpu::LREG0, 3, ADDR_MOD_7, ADDR);
            }
            TTI_SFPSTORE(p_sfpu::LREG0, 3, ADDR_MOD_7, ADDR);
        }
        rewrite_rows<POISON, MACRO, ADDR + 2, COUNT - 1>();
    }
}

// The half PACK holds: addresses BASE .. BASE + 255 (four 32-bit tiles of 64 rows).
template <bool POISON, bool MACRO, std::uint32_t BASE>
inline __attribute__((noinline)) void rewrite_half(std::uint32_t reps)
{
    if constexpr (POISON)
    {
        TTI_SFPLOADI(p_sfpu::LREG0, 0 /*fp16_b*/, 0x4120); // 10.0
    }
    for (std::uint32_t r = 0; r < reps; ++r)
    {
        rewrite_rows<POISON, MACRO, BASE + 0, 32>();
        rewrite_rows<POISON, MACRO, BASE + 64, 32>();
        rewrite_rows<POISON, MACRO, BASE + 128, 32>();
        rewrite_rows<POISON, MACRO, BASE + 192, 32>();
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const bool hammer           = (params.DST_INDEX & 2) != 0;
    const bool poison           = (params.DST_INDEX & 4) != 0;
    const bool macro            = (params.DST_INDEX & 8) != 0;
    const std::uint32_t reps    = params.DST_INDEX >> 4;
    volatile std::uint32_t* ts  = reinterpret_cast<volatile std::uint32_t*>(0x111000);
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    if (macro)
    {
        // Macro 0: only the store sub-unit, SFPSTORE of the loaded LReg to the loaded address, delay 0, the load's Mod0.
        const std::uint32_t store_bits = poison ? (0x40 | (0 << 3) | 3) : ((0 << 3) | 3); // poison: store LReg[16] instead
        TTI_SFPLOADI(p_sfpu::LREG0, 10 /*lower 16 bits*/, 0);
        TT_SFPLOADI(p_sfpu::LREG0, 8 /*upper 16 bits*/, store_bits << 8);
        TTI_SFPCONFIG(0, 4 + 0, 0);
        TTI_SFPCONFIG(0x010, 8, 1); // Misc: StoreMod0 0, UsesLoadMod0ForStore for macro 0, every delay counts cycles
    }
    for (std::uint32_t loop = 0; loop < params.LOOP_FACTOR; ++loop)
    {
        for (int block = 0; block < params.NUM_BLOCKS; ++block)
        {
            _llk_packer_wait_for_math_done_();
            ts[3 * block + 0] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            if (hammer)
            {
                if (dest_offset_id == 0)
                {
                    (poison && !macro) ? rewrite_half<true, false, 0>(reps) : macro ? rewrite_half<false, true, 0>(reps) : rewrite_half<false, false, 0>(reps);
                }
                else
                {
                    (poison && !macro) ? rewrite_half<true, false, DEST_REGISTER_HALF_SIZE>(reps)
                    : macro  ? rewrite_half<false, true, DEST_REGISTER_HALF_SIZE>(reps)
                             : rewrite_half<false, false, DEST_REGISTER_HALF_SIZE>(reps);
                }
                TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
            }
            ts[3 * block + 1] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            for (std::uint32_t t = 0; t < params.NUM_TILES_IN_BLOCK; ++t)
            {
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(
                    t, L1_ADDRESS(params.buffer_Res[block * params.NUM_TILES_IN_BLOCK + t]));
            }
            if ((params.DST_INDEX & ((1u << 6) | (1u << 5))) != 0) // bit 5: same ZEROACC repeat with the normal FPU datacopy on MATH
            {
                // zero-flag test: after the pack, clear this (already packed) half again and again, so ZEROACC HALF
                // keeps landing while MATH's SFPU stores fill the other half. bits 16..23 = repetitions / 4.
                TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
                const std::uint32_t zr = ((params.DST_INDEX >> 16) & 0xff) * 4;
                for (std::uint32_t z = 0; z < zr; ++z)
                {
                    TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
                }
            }
            reinterpret_cast<volatile std::uint32_t*>(0x113000)[block] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            ts[3 * block + 2] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        }
    }
}

#endif
