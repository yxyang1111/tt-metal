// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// CUSTOM_MM — Wormhole B0 version
//
// WH port: Replaces BH's CFGSHIFTMASK + SCRATCH_SEC* MOP-based address
// auto-increment with RISC-V managed software loops. Functionally equivalent
// to the BH version. SrcB address management via hardware counters is preserved.
//
// Constraints (same as BH):
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32]
// rt_dim: 1
// ct_dim: any integer from 1 to 16
// kt_dim: even number from 2 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported

inline void _llk_unpack_AB_custom_mm_mop_config_(const std::uint32_t ct_dim, const bool post1) {
    // WH: No MOP / CFGSHIFTMASK configuration needed — using software loops
    (void)ct_dim;
    (void)post1;
}

template <bool transpose = false>
inline void _llk_unpack_AB_custom_mm_init_(
    const std::uint32_t unpB_face_r_dim, const std::uint32_t unpA_dst_format, const std::uint32_t ct_dim = 1) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    _llk_unpack_AB_custom_mm_mop_config_(ct_dim, false);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

// WH software loop: processes kt_dim k-rows, each with ct_dim tiles.
// First tile per k-row: full unpack (SrcA + SrcB).
// Remaining tiles per k-row: SrcA only (SrcB reused).
//
// Address management uses pipeline instructions (TT_SETDMAREG + TTI_REG2FLOP
// + TTI_ADDDMAREG) instead of direct RISC-V config register writes, which
// avoids a race condition where the RISC-V loop runs ahead and overwrites the
// config register before the unpack hardware has consumed the previous value.
inline void _llk_unpack_AB_custom_mm_run_(
    volatile uint* cfg,
    std::uint32_t address_a,
    const std::uint32_t address_b,
    const std::uint32_t block_increment,
    const std::uint32_t inner_increment,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {

    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;

    // Load block_increment into GPR TMP1 (used by TTI_ADDDMAREG in inner loop)
    TT_SETDMAREG(0, LOWER_HALFWORD(block_increment), 0, LO_16(p_gpr_unpack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(block_increment), 0, HI_16(p_gpr_unpack::TMP1));

    semaphore_post(semaphore::UNPACK_SYNC);

    // Ensure SrcB base address cfg write has been flushed before any unpack
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    std::uint32_t current_addr_a = address_a;
    for (std::uint32_t k = 0; k < kt_dim; k++) {
        // Load current SrcA address into GPR TMP0 and write to config via pipeline
        TT_SETDMAREG(0, LOWER_HALFWORD(current_addr_a), 0, LO_16(p_gpr_unpack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(current_addr_a), 0, HI_16(p_gpr_unpack::TMP0));
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
        TTI_NOP;
        TTI_NOP;

        // First tile: full unpack (SrcA + SrcB)
        TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);

        if (ct_dim > 1) {
            for (std::uint32_t c = 1; c < ct_dim; c++) {
                // Advance SrcA address in pipeline: TMP0 += block_increment
                TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP1);
                TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                TTI_NOP;
                TTI_NOP;

                TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);
            }
        }
        // Advance RISC-V side address for next k-row's TT_SETDMAREG
        current_addr_a += (ct_dim > 1) ? ((ct_dim - 1) * block_increment + inner_increment) : inner_increment;
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    wait_for_next_context(1);
    reset_config_context();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

template <bool read_transposed = false, bool clear_src = true>
inline void _llk_unpack_AB_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    volatile uint* cfg = get_cfg_pointer();

    const std::uint32_t block_increment = read_transposed ? kt_dim * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * kt_dim) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    wait_for_next_context(1);
    reset_config_context();

    if constexpr (clear_src) {
        TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC_RESET_ALL_BANKS);
    }

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, block_increment, inner_increment, kt_dim, ct_dim);
}
