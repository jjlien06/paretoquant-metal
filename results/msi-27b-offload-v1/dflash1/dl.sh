cd /models
U=https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/resolve/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
curl -sSL -C - -o Qwen3.8-27B-DFlash2-Q4_K_M.gguf "$U"; echo curl=$?
sha256sum Qwen3.8-27B-DFlash2-Q4_K_M.gguf; ls -l /models
