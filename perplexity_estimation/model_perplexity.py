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
def opt_eval(quant_model, base_model, model_swap:int, tokens, dev, seqlen=2048):
    print("Evaluating ...")

    #These are the tokenized inputs
    testenc = tokens.input_ids
    nsamples = testenc.numel() // seqlen
    
    # Pass the tokens to the base_model and 
    inps = []
    batched_inputs = {'input_ids': None, 'attention_mask': None}
    for i in range(nsamples):
        batched_inputs['input_ids'] = tokens['input_ids'][:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        batched_inputs['attention_mask'] = tokens['attention_mask'][:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        
        #Outputs of the base model across all the layers
        hidden = base_model(**batched_inputs, output_hidden_states=True)
        
        #hidden = base_model(**testenc, output_hidden_states=True)
        inps.append(hidden.hidden_states[model_swap-1])
    
    #Concatenate the list of inputs to have the same format
    inps = torch.cat(inps, dim = 0) 
    inps = inps.half()
    
    del hidden, batched_inputs, tokens
    
    
    use_cache = quant_model.config.use_cache
    quant_model.config.use_cache = False
    #This is a ModuleList of length 12
    layers = quant_model.model.decoder.layers
    
    """
    #This part deals with the embedding layer
    quant_model.model.decoder.embed_tokens = quant_model.model.decoder.embed_tokens.to(dev)
    quant_model.model.decoder.embed_positions = quant_model.model.decoder.embed_positions.to(dev)
    if hasattr(quant_model.model.decoder, "project_out") and quant_model.model.decoder.project_out:
        quant_model.model.decoder.project_out = quant_model.model.decoder.project_out.to(dev)
    if hasattr(quant_model.model.decoder, "project_in") and quant_model.model.decoder.project_in:
        quant_model.model.decoder.project_in = quant_model.model.decoder.project_in.to(dev)
        
    layers[0] = layers[0].to(dev)

    
    dtype = next(iter(quant_model.parameters())).dtype 
    #inps = torch.zeros((nsamples, seqlen, quant_model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}
    
    #This part is used to deal with the firse embedding layer so I don't need it ATM
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            #inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        try:
            quant_model(batch)
        except ValueError: #So it stops at the first layer
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
    
    attention_mask = cache["attention_mask"]
    """
    #Here I only emulate the content of the attention mask (not sure if for other models need a different approach)
    attention_mask = torch.full((seqlen,seqlen), -65504.0, dtype=torch.half, device=dev)
    attention_mask = attention_mask.triu(1).unsqueeze(0).unsqueeze(0)
    
    outs = torch.zeros_like(inps)

    for i in range(model_swap-1,len(layers)):
        #print(i) 
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        #Swap inputs and outputs for the next layer
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
    print(model_swap,ppl.item())

    quant_model.config.use_cache = use_cache

@torch.no_grad()
def prunde_model_and_quantize(model_name, layer): 
    
    #Create train dataset
    traindataset, _ = get_wikitext2(128, 0, 2048, pretrained_model_dir)  
    
    model = OPTForCausalLM.from_pretrained(model_name).to("cuda:0")
    inps = []
    
    for batch in traindataset:
        #Load tensors on the GPU
        batch['input_ids'] = batch['input_ids'].to("cuda:0")
        batch['attention_mask'] = batch['attention_mask'].to("cuda:0") 
        #Outputs of the base model across all the layers
        hidden = model(**batch, output_hidden_states=True)
        #Get outputs of the layer before we prune
        inps.append({'input_ids' : hidden.hidden_states[layer-1].half(), 'attention_mask': batch['attention_mask']}) 
    
    #Keep only the last decoder layers [-layer:] and stored them in my huggingface
    #model.model.decoder.layers = nn.ModuleList(list(model.model.decoder.layers)[layer:])
    
    #Save partial model to huggingface hub
    model_name = f"Serione/opt-125m-{layer}"
    model.push_to_hub(model_name)
    
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(inps, use_triton=False)
    
    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir+f"-{layer}", use_safetensors=True)


    
def main():
    layer = 12
    prunde_model_and_quantize(pretrained_model_dir, layer)
    
    
    traindataset, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)  
    """
    #Check if dir exists, otherwise quantize
    if (os.path.exists(quantized_model_dir) and os.path.isdir(quantized_model_dir)):
    
        quantize_config = BaseQuantizeConfig(
            bits=4,  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            desc_act=False,  # desc_act and group size only works on triton
        )

        # load un-quantized model, the model will always be force loaded into cpu
        model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

        # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
        # with value under torch.LongTensor type.
        model.quantize(traindataset, use_triton=False)
        
        # save quantized model
        model.save_quantized(quantized_model_dir)

        # save quantized model using safetensors
        model.save_quantized(quantized_model_dir, use_safetensors=True)
    """

    #Load quantized model, currently only support cpu or single gpu
    quantized_model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir+f"-{layer}", device="cuda:0", use_triton=False)
    
    #Load the full model, output hiddel states allows to access intermediate values
    base_model = OPTForCausalLM.from_pretrained(pretrained_model_dir, output_hidden_states=True).to("cuda:0")
    
    opt_eval(quantized_model.model, base_model, layer, testenc, "cuda:0")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
    

    
    
    
    
