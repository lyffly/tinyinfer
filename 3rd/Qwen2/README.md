# Qwen2

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen2.png" width="400"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2407.10671">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen2/">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2-72B-Instruct">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>


Visit our Hugging Face or ModelScope organization (click links above), search checkpoints with names starting with `Qwen2-` or visit the [Qwen2 collection](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f), and you will find all you need! Enjoy!

To learn more about Qwen2, feel free to read our documentation \[[EN](https://qwen.readthedocs.io/en/latest/)|[ZH](https://qwen.readthedocs.io/zh-cn/latest/)\]. Our documentation consists of the following sections:

* Quickstart: the basic usages and demonstrations;
* Inference: the guidance for the inference with transformers, including batch inference, streaming, etc.;
* Run Locally: the instructions for running LLM locally on CPU and GPU, with frameworks like `llama.cpp` and `Ollama`;
* Deployment: the demonstration of how to deploy Qwen for large-scale inference with frameworks like `vLLM`, `TGI`, etc.;
* Quantization: the practice of quantizing LLMs with GPTQ, AWQ, as well as the guidance for how to make high-quality quantized GGUF files;
* Training: the instructions for post-training, including SFT and RLHF (TODO) with frameworks like Axolotl, LLaMA-Factory, etc.
* Framework: the usage of Qwen with frameworks for application, e.g., RAG, Agent, etc.
* Benchmark: the statistics about inference speed and memory footprint.

## Introduction

After months of efforts, we are pleased to announce the evolution from Qwen1.5 to Qwen2. This time, we bring to you:

* Pretrained and instruction-tuned models of 5 sizes, including Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B, and **Qwen2-72B**;
* Having been trained on data in **27** additional languages besides English and Chinese;
* State-of-the-art performance in a large number of benchmark evaluations;
* Significantly improved performance in coding and mathematics;  
* Extended context length support up to **128K** tokens with Qwen2-7B-Instruct and Qwen2-72B-Instruct.


## News
* 2024.06.06: We released the Qwen2 series. Check our [blog](https://qwenlm.github.io/blog/qwen2/)!
* 2024.03.28: We released the first MoE model of Qwen: Qwen1.5-MoE-A2.7B! Temporarily, only HF transformers and vLLM support the model. We will soon add the support of llama.cpp, mlx-lm, etc. Check our [blog](https://qwenlm.github.io/blog/qwen-moe/) for more information!
* 2024.02.05: We released the Qwen1.5 series.

## Performance

Detailed evaluation results are reported in this <a href="https://qwenlm.github.io/blog/qwen2/"> 📑 blog</a>.


## Requirements
* `transformers>=4.40.0` for Qwen2 dense and MoE models. The latest version is recommended.

> [!Warning]
> <div align="center">
> <b>
> 🚨 This is a must because `transformers` integrated Qwen2 codes since `4.37.0` and Qwen2Moe code since `4.40.0`.
> </b>
> </div>

For requirements on GPU memory and the respective throughput, see results [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Quickstart

### 🤗 Hugging Face Transformers

Here we show a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

For quantized models, we advise you to use the GPTQ and AWQ correspondents, namely `Qwen2-7B-Instruct-GPTQ-Int8`, `Qwen2-7B-Instruct-AWQ`. 

### 🤖 ModelScope
We strongly advise users especially those in mainland China to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.

### 💻 Run locally

#### Ollama

> [!Warning]
> You need `ollama>=0.1.42`.


> [!NOTE]
> <div align="center">
> Ollama provides an <a href="https://github.com/ollama/ollama/blob/main/docs/openai.md">OpenAI-compatible API</a>, which however does NOT support <b>function calling</b>. For tool use capabilities, consider using <a href="https://github.com/QwenLM/Qwen-Agent">Qwen-Agent</a>, which offers a wrapper for function calling over the API.
> </div>

After [installing ollama](https://github.com/ollama/ollama/blob/main/README.md), you can initiate the ollama service with the following command:
```shell
ollama serve
# You need to keep this service running whenever you are using ollama
```

To pull a model checkpoint and run the model, use the `ollama run` command. You can specify a model size by adding a suffix to `qwen2`, such as `:0.5b`, `:1.5b`, `:7b`, or `:72b`:
```shell
ollama run qwen2:7b
# To exit, type "/bye" and press ENTER
```

You can also access the ollama service via its OpenAI-compatible API. Please note that you need to (1) keep `ollama serve` running while using the API, and (2) execute `ollama run qwen2:7b` before utilizing this API to ensure that the model checkpoint is prepared.
```py
from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # required but ignored
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='qwen2:7b',
)
```

For additional details, please visit [ollama.ai](https://ollama.ai/).

#### llama.cpp

> [!Warning]
> You need `llama.cpp>=b3370`.

Download our provided GGUF files or create them by yourself, and you can directly use them with the latest [`llama.cpp`](https://github.com/ggerganov/llama.cpp) with a one-line command:
```shell
./llama-cli -m <path-to-file> -n 512 -co -i -if -f prompts/chat-with-qwen.txt --in-prefix "<|im_start|>user\n" --in-suffix "<|im_end|>\n<|im_start|>assistant\n"
```

#### MLX-LM

If you are running on Apple Silicon, we have also provided checkpoints compatible with [`mlx-lm`](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md). Look for models ending with MLX on HuggingFace Hub, like [Qwen2-7B-Instruct-MLX](https://huggingface.co/Qwen/Qwen2-7B-Instruct-MLX).

#### LMStudio

Qwen2 has already been supported by [lmstudio.ai](https://lmstudio.ai/). You can directly use LMStudio with our GGUF files.

#### OpenVINO

Qwen2 has already been supported by [OpenVINO toolkit](https://github.com/openvinotoolkit). You can install and run this [chatbot example](https://github.com/OpenVINO-dev-contest/Qwen2.openvino) with Intel CPU, integrated GPU or discrete GPU. 


## Web UI

#### Text generation web UI

You can directly use [`text-generation-webui`](https://github.com/oobabooga/text-generation-webui) for creating a web UI demo. If you use GGUF, remember to install the latest wheel of `llama.cpp` with the support of Qwen2.


#### llamafile

Clone [`llamafile`](https://github.com/Mozilla-Ocho/llamafile), run source install, and then create your own llamafile with the GGUF file following the guide [here](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#creating-llamafiles). You are able to run one line of command, say `./qwen.llamafile`, to create a demo.


## Deployment

Qwen2 is supported by multiple inference frameworks. Here we demonstrate the usage of `vLLM` and `SGLang`.

> [!Warning]
> <div align="center">
> The OpenAI-compatible APIs provided by vLLM and SGLang currently do NOT support <b>function calling</b>. For tool use capabilities, <a href="https://github.com/QwenLM/Qwen-Agent">Qwen-Agent</a> provides a wrapper around these APIs to support function calling.
> </div>

### vLLM

We advise you to use `vLLM>=0.4.0` to build OpenAI-compatible API service. Start the server with a chat model, e.g. `Qwen2-7B-Instruct`:
```shell
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model Qwen/Qwen2-7B-Instruct 
```

Then use the chat API as demonstrated below:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen2-7B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
    ]
    }'
```
```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen2-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ]
)
print("Chat response:", chat_response)
```

### SGLang

> [!NOTE]
> <div align="center">
> SGLang now does NOT support the <b>Qwen2MoeForCausalLM</b> architecture, thus making <b>Qwen2-57B-A14B</b> incompatible.
> </div>

Please install `SGLang` from source. Similar to `vLLM`, you need to launch a server and use OpenAI-compatible API service. Start the server first:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen2-7B-Instruct --port 30000
```
You can use it in Python as shown below:
```python
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

state = multi_turn_question.run(
    question_1="What is the capital of China?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])

print(state["answer_1"])
```

## Finetuning

We advise you to use training frameworks, including [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory), [Swift](https://github.com/modelscope/swift), etc., to finetune your models with SFT, DPO, PPO, etc.

## 🐳 Docker

To simplify the deployment process, we provide docker images with pre-built environments: [qwenllm/qwen](https://hub.docker.com/r/qwenllm/qwen). You only need to install the driver and download model files to launch demos and finetune the model.

```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen2 -it qwenllm/qwen:2-cu121 bash
```

## License Agreement

Check the license of each model inside its HF repo. It is NOT necessary for you to submit a request for commercial usage.

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}
```

## Contact Us
If you are interested to leave a message to either our research team or product team, join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](assets/wechat.png)!
