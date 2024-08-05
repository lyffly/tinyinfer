llama.cpp
===========================

`llama.cpp <https://github.com/ggerganov/llama.cpp>`__ is a C++ library
for LLM inference with mimimal setup. It enables running Qwen on your
local machine. It is a plain C/C++ implementation without dependencies,
and it has AVX, AVX2 and AVX512 support for x86 architectures. It
provides 2, 3, 4, 5, 6, and 8-bit quantization for faster inference and
reduced memory footprint. CPU+GPU hybrid inference to partially
accelerate models larger than the total VRAM capacity is also supported.
Essentially, the usage of llama.cpp is to run the GGUF (GPT-Generated
Unified Format ) models. For more information, please refer to the
official GitHub repo. Here we demonstrate how to run Qwen with
llama.cpp.

Prerequisites
-------------

This example is for the usage on Linux or MacOS. For the first step,
clone the repo and enter the directory:

.. code:: bash

   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

Then use ``make``:

.. code:: bash

   make

Then you can run GGUF files with ``llama.cpp``.

Running Qwen GGUF Files
-----------------------

We provide a series of GGUF models in our Hugging Face organization, and
to search for what you need you can search the repo names with
``-GGUF``. Download the GGUF model that you want with
``huggingface-cli`` (you need to install it first with
``pip install huggingface_hub``):

.. code:: bash

   huggingface-cli download <model_repo> <gguf_file> --local-dir <local_dir> --local-dir-use-symlinks False

for example:

.. code:: bash

   huggingface-cli download Qwen/Qwen2-7B-Instruct-GGUF qwen2-7b-instruct-q5_k_m.gguf --local-dir . --local-dir-use-symlinks False

Then you can run the model with the following command:

.. code:: bash

   ./llama-cli -m qwen2-7b-instruct-q5_k_m.gguf \
  -n 512 -co -i -if -f prompts/chat-with-qwen.txt \
  --in-prefix "<|im_start|>user\n" \
  --in-suffix "<|im_end|>\n<|im_start|>assistant\n" \
  -ngl 80 -fa

where ``-n`` refers to the maximum number of tokens to generate. There
are other hyperparameters for you to choose and you can run

.. code:: bash

   ./llama-cli -h

to figure them out.

.. important::
   
   Previously, Qwen2 models generate nonsense like ``GGGG...`` with ``llama.cpp`` on GPUs.
   The workaround is to enable flash attention (``-fa``), which uses a different implementation, and offload the whole model to the GPU (``-ngl 80``) due to broken partial GPU offloading with flash attention.
   
   Both should be no longer necessary after ``b3370``, but it is still recommended to enable both for maximum efficiency.


Make Your GGUF Files
--------------------

We introduce the method of creating and quantizing GGUF files in
`quantization/llama.cpp <../quantization/gguf.html>`__. You can refer
to that document for more information.

Perplexity Evaluation
---------------------

``llama.cpp`` provides methods for us to evaluate the perplexity
performance of the GGUF models. To do this, you need to prepare the
dataset, say "wiki test". Here we demonstrate an example to run the
test.

For the first step, download the dataset:

.. code:: bash

   wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research -O wikitext-2-raw-v1.zip
   unzip wikitext-2-raw-v1.zip

Then you can run the test with the following command:

.. code:: bash

   ./llama-perplexity -m <gguf_path> -f wiki.test.raw

where the output is like

.. code:: text

   perplexity : calculating perplexity over 655 chunks
   24.43 seconds per pass - ETA 4.45 hours
   [1]4.5970,[2]5.1807,[3]6.0382,...

Wait for some time and you will get the perplexity of the model.

Use GGUF with LM Studio
-----------------------

If you still find it difficult to use ``llama.cpp``, I advise you to
play with `LM Studio <https://lmstudio.ai/>`__, which is a platform
for your to search and run local LLMs. Qwen2 has already been
officially part of LM Studio. Have fun!
