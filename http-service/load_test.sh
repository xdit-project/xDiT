HOST=127.0.0.1
PORT=6000

oha -m POST -n 4 -c 1 \
    -H "content-type:application/json" -d '{ "prompt": "A tree", "num_inference_steps": 28, "seed": 42, "save_disk_path": "output"}' \
    http://${HOST}:${PORT}/generate
