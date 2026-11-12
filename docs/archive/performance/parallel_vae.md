# Historical parallel VAE result

Combining patch parallelism and chunked processing can reduce the memory needed for high-resolution decoding. The original report described decoding images up to 10240px, but did not record the complete hardware and software configuration here. Treat that resolution as a historical result; the usable resolution depends on the VAE, image geometry, device memory, and parallel configuration.
