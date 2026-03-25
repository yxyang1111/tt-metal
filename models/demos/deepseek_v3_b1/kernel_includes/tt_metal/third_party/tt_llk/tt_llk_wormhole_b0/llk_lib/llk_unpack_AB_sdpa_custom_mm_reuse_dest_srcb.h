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

// SDPA_CUSTOM_MM_REUSE_DEST_SRCB — Wormhole B0 version
//
// Custom matmul that reuses SrcB from dest and only unpacks SrcA.
// Output height and width should be single tile with tile shape [1, 32].
//
// WH port: Replaces BH's CFGSHIFTMASK + SCRATCH_SEC* MOP-based address
// auto-increment with RISC-V managed software loops. Functionally equivalent
// to the BH version but without hardware MOP address stepping.
//
// Constraints (same as BH):
// - ct_dim = 1, rt_dim = 1 (single output tile)
// - nt_dim: 1 to 16
// - kt_dim: even number from 2 to 256 (inclusive)
// - kernel_broadcast_a = 0, kernel_broadcast_b = 0
inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_mop_config_(const std::uint32_t nt_dim) {
    // WH: No MOP / CFGSHIFTMASK configuration needed — using software loops
    (void)nt_dim;
}

__attribute__((always_inline)) inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(
    const std::uint32_t nt_dim = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces = 4) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_mop_config_(nt_dim);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

// WH software loop replaces BH's collapsed MOP + CFGSHIFTMASK.
// Uses pipeline instructions (TT_SETDMAREG + TTI_REG2FLOP + TTI_ADDDMAREG)
// for address management to avoid RISC-V / unpack-pipeline race condition.
inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
    const std::uint32_t base_address_a,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_size_a,
    const std::uint32_t kt_dim = 1,
    const std::uint32_t nt_dim = 1,
    const std::uint32_t in1_k_stride = 1) {
    volatile uint* cfg = get_cfg_pointer();

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t block_increment = tile_size_a;
    const std::uint32_t row_stride = in1_k_stride * tile_size_a;

    wait_for_next_context(1);
    reset_config_context();

    // Load block_increment into GPR TMP1 for inner-loop TTI_ADDDMAREG
    TT_SETDMAREG(0, LOWER_HALFWORD(block_increment), 0, LO_16(p_gpr_unpack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(block_increment), 0, HI_16(p_gpr_unpack::TMP1));

    semaphore_post(semaphore::UNPACK_SYNC);

    for (std::uint32_t k = 0; k < kt_dim; k++) {
        const std::uint32_t row_base = address_a + k * row_stride;

        // Load row base address into GPR TMP0 and write to config via pipeline
        TT_SETDMAREG(0, LOWER_HALFWORD(row_base), 0, LO_16(p_gpr_unpack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(row_base), 0, HI_16(p_gpr_unpack::TMP0));
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
        TTI_NOP;
        TTI_NOP;

        TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);

        for (std::uint32_t n = 1; n < nt_dim; n++) {
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP1);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
            TTI_NOP;
            TTI_NOP;

            TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);
        }
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    wait_for_next_context(1);
    reset_config_context();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}
