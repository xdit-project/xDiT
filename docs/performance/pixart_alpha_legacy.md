# Pixart-Alpha Legacy Version Performance

Here are the benchmark results for Pixart-Alpha using the 20-step DPM solver as the scheduler across various image resolutions. 
To replicate these findings, please refer to the script at [legacy/scripts/benchmark.sh](../../legacy/scripts/benchmark.sh).

1. The Latency on 4xA100-80GB (PCIe)

<div align="center">
    <img src="../../assets/latency-A100-PCIe.png" alt="A100 PCIe latency">
</div>

2. The Latency on 8xL20-48GB (PCIe)

<div align="center">
    <img src="../../assets/latency-L20.png" alt="L20 latency">
</div>

3. The Latency on 8xA100-80GB (NVLink)

<div align="center">
    <img src="../../assets/latency-A100-NVLink.png" alt="latency-A100-NVLink">
</div>

4. The Latency on 4xT4-16GB (PCIe)

<div align="center">
    <img src="../../assets/latency-T4.png" 
    alt="latency-T4">
</div>
