class Kriya < Formula
  desc "Ultra-high-throughput native Metal inference engine for frontier LLMs on Apple Silicon"
  homepage "https://github.com/rishisharma/kriya-rt"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/rishisharma/kriya-rt/releases/download/v0.1.0/kriya-rt-v0.1.0-aarch64-apple-darwin.tar.gz"
      sha256 "e03ae3b490b98131651364c2eae2645878b72f1252a3dc272701820a5781023d"
    end
  end

  depends_on :macos
  depends_on arch: :arm64

  def install
    bin.install "bin/kriya-server"
    bin.install "bin/kriya_convert"
    doc.install "README.md"
  end

  def caveats
    <<~EOS
      KriyaRT is optimized specifically for Apple Silicon Unified Memory.
      
      Quick Start:
        # Run inference directly on any local Hugging Face or MLX checkpoint:
        kriya-server --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/... --prompt "What is Apple Silicon?"

        # Serve OpenAI-compatible API locally on port 8080:
        kriya-server --model <model_directory> --port 8080
    EOS
  end

  service do
    run [opt_bin/"kriya-server", "--host", "127.0.0.1", "--port", "8080"]
    keep_alive true
    log_path var/"log/kriya-server.log"
    error_log_path var/"log/kriya-server.err.log"
  end

  test do
    system "#{bin}/kriya-server", "--version"
  end
end
