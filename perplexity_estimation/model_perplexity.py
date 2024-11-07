import numpy as np
import torch
import torch.nn as nn
import os

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from tqdm import trange
from transformers import OPTForCausalLM

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "perplexity_estimation\opt-125m-4bit-128g"


# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


@torch.no_grad()
def opt_eval(quant_model, full_model, testenc, dev, config, seqlen=2048):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = quant_model.config.use_cache
    quant_model.config.use_cache = False
    layers = quant_model.model.decoder.layers

    quant_model.model.decoder.embed_tokens = quant_model.model.decoder.embed_tokens.to(dev)
    quant_model.model.decoder.embed_positions = quant_model.model.decoder.embed_positions.to(dev)
    if hasattr(quant_model.model.decoder, "project_out") and quant_model.model.decoder.project_out:
        quant_model.model.decoder.project_out = quant_model.model.decoder.project_out.to(dev)
    if hasattr(quant_model.model.decoder, "project_in") and quant_model.model.decoder.project_in:
        quant_model.model.decoder.project_in = quant_model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(quant_model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, quant_model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        try:
            quant_model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    quant_model.model.decoder.embed_tokens = quant_model.model.decoder.embed_tokens.cpu()
    quant_model.model.decoder.embed_positions = quant_model.model.decoder.embed_positions.cpu()
    if hasattr(quant_model.model.decoder, "project_out") and quant_model.model.decoder.project_out:
        quant_model.model.decoder.project_out = quant_model.model.decoder.project_out.cpu()
    if hasattr(quant_model.model.decoder, "project_in") and quant_model.model.decoder.project_in:
        quant_model.model.decoder.project_in = quant_model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i,bool in enumerate(config):
        if bool: #If true we use the quantized model
            layer = layers[i].to(dev)
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            layers[i] = layer.cpu()
            del layer
        else:
            layer = full_model[i]
            for j in range(nsamples): #Might need to move the layer to cuda
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if quant_model.model.decoder.final_layer_norm is not None:
        quant_model.model.decoder.final_layer_norm = quant_model.model.decoder.final_layer_norm.to(dev)
    if quant_model.model.decoder.project_out is not None:
        quant_model.model.decoder.project_out = quant_model.model.decoder.project_out.to(dev)
    quant_model.lm_head = quant_model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if quant_model.model.decoder.final_layer_norm is not None:
            hidden_states = quant_model.model.decoder.final_layer_norm(hidden_states)
        if quant_model.model.decoder.project_out is not None:
            hidden_states = quant_model.model.decoder.project_out(hidden_states)
        lm_logits = quant_model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    #print(ppl.item())
    with open("F:/Thesis/AutoGPTQ/perplexity_estimation/output.txt", "a") as file:
        file.write(f'{ppl.item()}\n')

    quant_model.config.use_cache = use_cache

@torch.no_grad()
def prune_model_and_quantize(full_model, model_name, layers): 
    
    #Create train dataset
    traindataset, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)  
    
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset, use_triton=False, config=layers, full_model=full_model.model.decoder.layers)
    
    #save quantized model using safetensors
    model.save_quantized(quantized_model_dir+f"-test", use_safetensors=True)
    
    del model
    torch.cuda.empty_cache()
    return testenc
    
def main():
    import itertools
    
    full_model = OPTForCausalLM.from_pretrained(pretrained_model_dir).to("cuda:0").half()
    
    #List contains combination of which layers are quantized and which are not
    layers = list(itertools.product([True, False], repeat=12))[::-1] 
    for layer in layers:
        try:
            with open("F:/Thesis/AutoGPTQ/perplexity_estimation/output.txt", "a") as file:
                file.write(f'{layer},')
                
            testenc = prune_model_and_quantize(full_model,pretrained_model_dir, layer) 

            #Load quantized model, currently only support cpu or single gpu
            quantized_model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir+f"-test", device="cuda:0", use_triton=False)
            
            opt_eval(quantized_model.model, full_model.model.decoder.layers, testenc, "cuda:0", layer)
        
        except Exception as e:
            # Print the error message and continue to the next layer
            print(f"Error processing layer {layer}: {e}")
            with open("F:/Thesis/AutoGPTQ/perplexity_estimation/output.txt", "a") as file:
                file.write(f'{layer},')
            continue

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
    

    
    
    
    
