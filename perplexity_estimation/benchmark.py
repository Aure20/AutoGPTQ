import os 
os.environ['HF_HOME'] = '/cluster/scratch/negria/huggingface_cache/'
import torch
from transformers import AutoTokenizer
from auto_gptq.utils import Perplexity
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
def benchmark(model_name='gpt2', model_basename=None, n_ctx:int=512, n_batch:int=512, dataset_path='wikitext', dataset_name=None, test_split='test', train_split = 'train', text_column='text', 
per_gpu_max_memory=None, cpu_max_memory=None, use_safetensors=False, use_fast_tokenizer=False, trust_remote_code=False, disable_exllama=False, quantize_config:BaseQuantizeConfig=None):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_memory = {}
    if per_gpu_max_memory is not None and per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if cpu_max_memory is not None and cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    if use_safetensors:
        print(
            "The argument --use_safetensors is deprecrated and will be removed in the next release. It is now the default behavior."
        )

    if quantize_config is not None:
        model = AutoGPTQForCausalLM.from_pretrained(            
            model_name,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=trust_remote_code,
            inject_fused_mlp=False,
            inject_fused_attention=False,
            disable_exllama=disable_exllama,
            quantize_config=quantize_config
        )

        data = load_dataset(dataset_path, dataset_name, split=train_split) 
        text_list = [" \n" if s == "" else s for s in data[self.text_column]]
        traindataset = "".join(text_list)

        model.quantize(traindataset,use_triton=False, batch_size=1)

    else:
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=trust_remote_code,
            inject_fused_mlp=False,
            inject_fused_attention=False,
            disable_exllama=disable_exllama,
        )

    ppl = Perplexity(
        model,
        tokenizer,
        dataset_path,
        dataset_name,
        split,
        text_column,
    )
    ppl.calculate_perplexity(n_ctx, n_batch)

if __name__ == "__main__":

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
    )
    """
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    group_size: int = field(default=-1)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=True)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    quant_method: str = field(default=QUANT_METHOD.GPTQ)
    checkpoint_format: str = field(default=CHECKPOINT_FORMAT.GPTQ)
    model_name_or_path: Optional[str] = field(default=None)
    model_file_base_name: Optional[str] = field(default=None)
    """
    benchmark('facebook/opt-125m', model_basename='model',cpu_max_memory = 64, quantize_config=quantize_config)