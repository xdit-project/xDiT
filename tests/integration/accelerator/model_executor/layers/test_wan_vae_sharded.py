"""Row-sharded Wan VAE decode (xfuser.model_executor.layers.wan_vae_sharded) against diffusers' own decode."""

import pytest

pytestmark = [pytest.mark.multi_gpu]


def _small_vae():
    import torch
    from diffusers import AutoencoderKLWan

    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim=32, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, temperal_downsample=[False, True, True]
    )
    for p in vae.parameters():  # non-trivial norm scales and biases
        if p.dim() <= 1 or p.shape[1:] == (1, 1, 1):
            p.data.normal_(1.0 if p.dim() > 1 else 0.0, 0.1)
    return vae.to(device="cuda", dtype=torch.bfloat16).eval()


def _worker(rank, world_size, init_method, halos):
    import torch
    import torch.distributed as dist
    from diffusers.models.autoencoders import autoencoder_kl_wan as wanmod

    from xfuser.model_executor.layers.wan_vae_sharded import decode as wsd

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    try:
        g = torch.Generator(device="cuda").manual_seed(1)
        z = torch.randn(1, 16, 3, 16, 24, device="cuda", dtype=torch.bfloat16, generator=g)
        with torch.inference_mode():
            ref = _small_vae().decode(z).sample.float()
        outs = {}
        for halo in halos:
            vae = _small_vae()
            assert wsd.install(vae, None, halo=halo) == halo
            with torch.inference_mode():
                for _ in range(2):  # a second decode reuses the history buffers and plans
                    out = vae.decode(z).sample.float()
            assert out.shape == ref.shape
            rel = float((out - ref).norm() / ref.norm())
            assert rel < 2e-2, f"{halo}: sharded decode differs from the stock decode by {rel:.3e}"
            outs[halo] = out
        if len(outs) == 2:
            a, b = outs.values()
            assert torch.equal(a, b), "the halo transports must give bitwise equal videos"
        vae = _small_vae()
        wsd.install(vae, None, halo=halos[-1])
        with torch.inference_mode():
            # a batch decodes sample by sample, exactly as the samples alone
            z2 = torch.cat([z, 0.5 * z])
            both = vae.decode(z2).sample
            assert torch.equal(both[0:1], vae.decode(z).sample)
            assert torch.equal(both[1:2], vae.decode(0.5 * z).sample)
            # bands of one or two latent rows (no interior rows to overlap the exchange with)
            zs = z[:, :, :, : 2 * world_size - 1]
            small_ref = _small_vae().decode(zs).sample.float()
            small = vae.decode(zs).sample.float()
            rel = float((small - small_ref).norm() / small_ref.norm())
            assert rel < 2e-2, f"two-row bands differ from the stock decode by {rel:.3e}"
            # fewer latent rows than ranks: every rank runs diffusers' own decode
            zt = z[:, :, :, : world_size - 1]
            assert torch.equal(vae.decode(zt).sample, _small_vae().decode(zt).sample)
        # bound per instance: a fresh VAE still decodes with diffusers' own modules
        assert wanmod.WanResidualBlock.forward is not None
        assert "forward" not in vars(_small_vae().decoder.up_blocks[0].resnets[0])
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_decode_over_nccl_matches_stock(accelerator_ranks, world_size):
    accelerator_ranks(_worker, world_size=world_size, timeout=600, args=(("nccl",),))


@pytest.mark.rocm
@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_decode_over_sdma_matches_nccl_bitwise(accelerator_ranks, world_size):
    accelerator_ranks(_worker, world_size=world_size, timeout=600, args=(("nccl", "sdma"),))
