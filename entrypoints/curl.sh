
curl -X POST "http://localhost:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a cute rabbit",
           "num_inference_steps": 50,
           "seed": 42,
           "cfg": 7.5, 
           "save_disk_path": "/tmp"
         }'
