class Kriya < Formula
  desc "Ultra-high-throughput native Metal inference engine for frontier LLMs on Apple Silicon"
  homepage "https://github.com/RISHI5991/kriya-rt"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/RISHI5991/kriya-rt/releases/download/v0.1.0/kriya-rt-v0.1.0-aarch64-apple-darwin.tar.gz"
      sha256 "2f21cd08ad5085022657a67e50a718da30e957df806581be0912d5f096d3d654"
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

  test do
    system "#{bin}/kriya-server", "--version"
  end
end
