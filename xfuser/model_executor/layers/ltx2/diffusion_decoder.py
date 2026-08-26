"""Tile-parallel diffusion VAE decode for LTX-2.5 across 2/4/8 GPUs.

The stock ``LTX2VideoDiffusionDecoderModel.tiled_decode`` runs a triple-nested loop over
``(temporal × height × width)`` tiles, each independently calling ``forward_stage_4``
+ ``denoise`` (8 neighbourhood-attention blocks each).

``xFuserLTX2VideoDiffusionDecoderWrapper`` distributes those tiles across the SP group
via round-robin ownership, then does one small all_reduce for shape metadata followed by
per-tile broadcasts to reassemble the full video on every rank.  Communication is a
single collective per tile.

Tile geometry for the default 1024×1536×121 config: 12 tiles, adequate for 2/4/8 GPUs.
"""
from __future__ import annotations

import math

import torch
import torch.distributed

from diffusers import LTX2VideoDiffusionDecoderModel
from diffusers.models.autoencoders.ltx2_diffusion_decoder import _tile_intervals
from diffusers.utils.torch_utils import randn_tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)


class xFuserLTX2VideoDiffusionDecoderWrapper(LTX2VideoDiffusionDecoderModel):
    """``LTX2VideoDiffusionDecoderModel`` with tile-parallel decode across the SP group.
    Drop-in replacement: use ``from_pretrained`` on this class exactly as on the base
    class.  The runner activates tile-parallel decode by:
    1. Setting ``decoder._parallel_decode = True`` (done when ``--use_parallel_vae`` is on).
    2. Calling ``decoder.enable_tiling()`` so that ``decode()`` dispatches to ``tiled_decode``.
    When ``_parallel_decode`` is False, or ``sp_world_size == 1``, or the tile count is
    less than 2, ``tiled_decode`` delegates to the stock implementation unchanged.
    """

    # Set to True by the runner when --use_parallel_vae is on.
    _parallel_decode: bool = False

    def tiled_decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Decode with tiles distributed across SP ranks.
        Each rank owns a round-robin subset of tiles.  Owned tiles are computed locally;
        all ranks gather all tiles via a shape-metadata all_reduce + per-tile broadcast,
        then run the stock blend/assembly loop to produce the full video.
        Noise determinism:
        - *Shipping 1-step x0 path*: each tile is seeded from ``base_seed XOR f(t,h,w)``,
          making noise rank-invariant without requiring non-owners to know tile shapes.
          Output differs from single-GPU stock (different per-tile seeds) but is an
          equally valid sample.
        - *Multi-step path*: the full noise canvas is drawn identically on all ranks from
          a shared generator seeded by ``base_seed``, then each owner slices its region.
        """
        sp_world_size = get_sequence_parallel_world_size()
        if not self._parallel_decode or sp_world_size <= 1:
            return super().tiled_decode(z, generator=generator,
                                        num_inference_steps=num_inference_steps)

        sp_group = get_sp_group()
        sp_rank = get_sequence_parallel_rank()

        # Tile geometry
        decoder = self.decoder
        num_inference_steps = num_inference_steps or decoder.default_num_inference_steps
        batch_size = z.shape[0]
        patch_size = decoder.patch_size

        upsample_stride = decoder.upsamples[-1].stride
        scale_t = upsample_stride[0]
        scale_h = upsample_stride[1] * patch_size
        scale_w = upsample_stride[2] * patch_size

        tile_t  = self.tile_sample_min_num_frames // scale_t
        stride_t = self.tile_sample_stride_num_frames // scale_t
        tile_h  = self.tile_sample_min_height // scale_h
        stride_h = self.tile_sample_stride_height // scale_h
        tile_w  = self.tile_sample_min_width // scale_w
        stride_w = self.tile_sample_stride_width // scale_w

        min_sizes = [
            max(k4, -(-k5 // s))
            for k4, k5, s in zip(
                self.config.decoder_stage_kernels[-1],
                self.config.decoder_stage5_kernel,
                upsample_stride,
            )
        ]

        # Cheap deterministic stages run on the full volume on every rank.
        # This avoids communicating the intermediate features tensor and keeps the
        # tile-grid computation identical across ranks.
        features = decoder.forward_stages_1_to_3(z)
        ghost_frames = decoder.trailing_pad_latent_frames * math.prod(
            up.stride[0] for up in decoder.upsamples[:-1]
        )
        num_frames = features.shape[1] - ghost_frames
        height, width = features.shape[2], features.shape[3]

        temporal_tiles = _tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])
        height_tiles   = _tile_intervals(height,      tile_h, stride_h, min_sizes[1])
        width_tiles    = _tile_intervals(width,        tile_w, stride_w, min_sizes[2])

        blend_frames = (tile_t - stride_t) * scale_t
        blend_height = (tile_h - stride_h) * scale_h
        blend_width  = (tile_w - stride_w) * scale_w

        n_tiles = len(temporal_tiles) * len(height_tiles) * len(width_tiles)

        # Fall back to stock when there are too few tiles to distribute.
        if n_tiles < 2:
            return super().tiled_decode(z, generator=generator,
                                        num_inference_steps=num_inference_steps)

        # Noise setup
        single_step_x0 = (num_inference_steps == 1 and decoder.model_output_type == "x0")

        # Draw one integer from the shared generator to derive a base seed.
        # Generator state is guaranteed identical across SP ranks at decode entry, so
        # all ranks draw the same value, advancing the generator identically.
        if generator is not None:
            base_seed_t = torch.randint(0, 2**31, (1,), generator=generator, device=z.device)
            base_seed = int(base_seed_t.item())
        else:
            # generator=None: use a fixed seed (in practice the runner always provides one)
            base_seed = 0

        x_t_full: torch.Tensor | None = None
        if not single_step_x0:
            pixel_frames = num_frames * scale_t - (1 if scale_t == 2 else 0)
            shared_gen = torch.Generator(device=z.device).manual_seed(base_seed)
            x_t_full = randn_tensor(
                (batch_size, decoder.out_channels,
                 pixel_frames, height * scale_h, width * scale_w),
                generator=shared_gen,
                device=z.device,
                dtype=z.dtype,
            )

        # Compute owned tiles
        # shape_table[i] = (T_px, H_px, W_px) for tile i; zero for non-owned tiles.
        # After all_reduce(SUM) every rank knows all tile pixel shapes.
        shape_table = torch.zeros(n_tiles, 3, dtype=torch.int64, device=z.device)
        tile_outputs: dict[int, torch.Tensor] = {}

        tile_idx = 0
        for t_idx, (t0, t1) in enumerate(temporal_tiles):
            is_origin   = (t0 == 0)
            is_trailing = (t1 == num_frames)
            # Trailing temporal tile carries ghost frames into forward_stage_4.
            feature_t1 = features.shape[1] if is_trailing else t1

            for h_idx, (h0, h1) in enumerate(height_tiles):
                for w_idx, (w0, w1) in enumerate(width_tiles):
                    owner = tile_idx % sp_world_size

                    if owner == sp_rank:
                        context = decoder.forward_stage_4(
                            features[:, t0:feature_t1, h0:h1, w0:w1],
                            drop_leading_frame=is_origin,
                            crop_trailing_ghost=is_trailing,
                        )
                        tile_pixel_shape = (
                            batch_size,
                            decoder.out_channels,
                            context.shape[1],
                            context.shape[2] * patch_size,
                            context.shape[3] * patch_size,
                        )

                        if single_step_x0:
                            # Per-tile deterministic seeding: owner-only draw, no
                            # cross-rank RNG synchronisation needed.
                            # Small integer hash keeps seeds well-separated.
                            tile_seed = (base_seed ^ (t_idx * 0x3_D6F1
                                                      + h_idx * 0x1_E3
                                                      + w_idx * 0x7)) & 0x7FFF_FFFF
                            tile_gen = torch.Generator(device=z.device).manual_seed(tile_seed)
                            x_t = randn_tensor(tile_pixel_shape, generator=tile_gen,
                                               device=z.device, dtype=z.dtype)
                        else:
                            # Slice from the shared noise canvas (drawn identically on all
                            # ranks above).
                            pixel_t0 = (t0 * scale_t
                                        - (1 if not is_origin and scale_t == 2 else 0))
                            x_t = x_t_full[
                                :, :,
                                pixel_t0 : pixel_t0 + tile_pixel_shape[2],
                                h0 * scale_h : h0 * scale_h + tile_pixel_shape[3],
                                w0 * scale_w : w0 * scale_w + tile_pixel_shape[4],
                            ]

                        out = decoder.denoise(context, x_t, num_inference_steps)
                        tile_outputs[tile_idx] = out
                        shape_table[tile_idx, 0] = out.shape[2]  # T_px
                        shape_table[tile_idx, 1] = out.shape[3]  # H_px
                        shape_table[tile_idx, 2] = out.shape[4]  # W_px

                    tile_idx += 1

        # Gather: shape metadata + per-tile broadcast
        # all_reduce(SUM): safe because each element is non-zero on exactly one rank.
        torch.distributed.all_reduce(
            shape_table,
            op=torch.distributed.ReduceOp.SUM,
            group=sp_group.device_group,
        )

        gathered: dict[int, torch.Tensor] = {}
        for i in range(n_tiles):
            owner = i % sp_world_size
            T_px = int(shape_table[i, 0])
            H_px = int(shape_table[i, 1])
            W_px = int(shape_table[i, 2])
            if sp_rank == owner:
                buf = tile_outputs[i]
            else:
                buf = torch.empty(
                    batch_size, decoder.out_channels, T_px, H_px, W_px,
                    dtype=z.dtype, device=z.device,
                )
            # src is the SP-group-local rank (0 … sp_world_size-1).
            sp_group.broadcast(buf, src=owner)
            gathered[i] = buf

        # Assembly — identical to stock tiled_decode
        frame_groups: list[torch.Tensor] = []
        tile_idx = 0
        for _t_idx, (_t0, _t1) in enumerate(temporal_tiles):
            rows: list[list[torch.Tensor]] = []
            for _h_idx, _ in enumerate(height_tiles):
                row: list[torch.Tensor] = []
                for _w_idx, _ in enumerate(width_tiles):
                    row.append(gathered[tile_idx])
                    tile_idx += 1
                rows.append(row)

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_width)
                    keep_height = stride_h * scale_h if i < len(rows) - 1 else tile.shape[3]
                    keep_width  = stride_w * scale_w if j < len(row)  - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_height, :keep_width])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

        result = []
        for k, group in enumerate(frame_groups):
            if k > 0:
                group = self.blend_t(frame_groups[k - 1], group, blend_frames)
            if k < len(frame_groups) - 1:
                keep_frames = stride_t * scale_t - (1 if k == 0 and scale_t == 2 else 0)
                group = group[:, :, :keep_frames]
            result.append(group)
        return torch.cat(result, dim=2)