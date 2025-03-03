
curl -X POST "http://localhost:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "一只可爱的猫咪",
           "num_inference_steps": 50,
           "seed": 42,
           "cfg": 7.5
         }'
