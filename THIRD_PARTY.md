# Third-party components

This repository contains integration code, not model weights or prebuilt third-party runtimes.
Users download those components from their respective publishers.

| Component | Purpose | Upstream | License |
| --- | --- | --- | --- |
| faster-whisper | Local speech recognition | <https://github.com/SYSTRAN/faster-whisper> | MIT |
| faster-whisper-large-v3-turbo | CTranslate2 Whisper model | <https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo> | MIT |
| Qwen3-1.7B-GGUF | Local Chinese translation model | <https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF> | Apache-2.0 |
| llama.cpp | Local GGUF inference server | <https://github.com/ggml-org/llama.cpp> | MIT |
| PyAudioWPatch | Windows WASAPI loopback capture | <https://github.com/s0d3s/PyAudioWPatch> | MIT |
| CTranslate2 | Whisper inference runtime | <https://github.com/OpenNMT/CTranslate2> | MIT |
| NumPy | Audio processing | <https://numpy.org/> | BSD-3-Clause |

Optional NVIDIA Python packages are downloaded from PyPI when users install the
`gpu-windows` extra. Their use and redistribution are governed by NVIDIA's applicable terms.
This project does not commit those packages or NVIDIA binaries to the source repository.

The table is informational and does not replace the upstream license texts. Review the
linked upstream terms before redistributing models, runtimes, or a bundled binary build.
