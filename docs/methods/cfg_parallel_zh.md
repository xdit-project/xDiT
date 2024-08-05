# Classifier-Free Guidance (CFG) Parallel 

Classifier-Free Guidance通过提供更广泛的条件控制、减少训练负担、增强生成内容的质量和细节，以及提高模型的实用性和适应性，成为了扩散模型领域的一个重要进展技术。

对于一个输入prompt，使用CFG需要同时进行unconditional guide和text guide的生成 ，相当于输入DiT blocks的input latents batch_size = 2。CFG Parallel分离两个latents分别进行计算，在每个Diffusion Step forward完成后、Scheduler执行前Allgather一次latent space结果。它通信量远小于Pipefusion和Sequence Parallel。因此，使用CFG一定要使用CFG Parallel。