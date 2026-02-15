import torch
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, PaliGemmaProcessor, PaliGemmaForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration, AutoConfig, AutoModelForVision2Seq, LlavaForConditionalGeneration, MllamaForConditionalGeneration, GenerationConfig, AutoModelForCausalLM
    from qwen_vl_utils import process_vision_info
except:
    from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import math
import functools
import traceback
DEVICE = 'cuda'
# ==========================================================================
# Minimize Determinism
max_new_tokens=1024
do_sample = False

# ==========================================================================
### Sanity Check ###
def check():
    print("The model inference is correctly imported.")

# ==========================================================================
### Other Functions ###
def catch_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            traceback.print_exc()
            return None
    return wrapper

@catch_exceptions
def build_messages_qwen(images, prompt):
    num_images = len(images)

    if isinstance(prompt, str):
        prompts = [prompt] * num_images

    elif isinstance(prompt, list):
        if len(prompt) != num_images:
            raise ValueError(
                f"Prompt list length ({len(prompt)}) "
                f"does not match number of images ({num_images})"
            )
        prompts = prompt

    else:
        raise TypeError("prompt must be a str or list[str]")

    messages = []
    for image, p in zip(images, prompts):
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": p},
            ],
        }])

    return messages

@catch_exceptions
def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

@catch_exceptions
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

@catch_exceptions
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

@catch_exceptions
def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

@catch_exceptions
def build_messages_intern(images, prompt):
    num_images = len(images)

    def add_image_tag(p):
        return p if p.startswith("<image>\n") else "<image>\n" + p

    if isinstance(prompt, str):
        questions = [add_image_tag(prompt)] * num_images

    elif isinstance(prompt, list):
        if len(prompt) != num_images:
            raise ValueError("Number of prompts must match number of images")
        questions = [add_image_tag(p) for p in prompt]

    else:
        raise TypeError("prompts must be str or list[str]")

    pixel_values_list = []
    num_patches_list = []

    for img in images:
        pv = load_image(img, max_num=12).to(torch.bfloat16).cuda()
        pixel_values_list.append(pv)
        num_patches_list.append(pv.size(0))

    pixel_values = torch.cat(pixel_values_list, dim=0)

    return pixel_values, num_patches_list, questions

@catch_exceptions
def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

@catch_exceptions
def build_messages_idefic(images, prompt):
    num_images = len(images)

    if isinstance(prompt, str):
        prompts = [prompt] * num_images

    elif isinstance(prompt, list):
        if len(prompt) != num_images:
            raise ValueError(
                f"Prompt list length ({len(prompt)}) "
                f"does not match number of images ({num_images})"
            )
        prompts = prompt

    else:
        raise TypeError("prompt must be a str or list[str]")

    messages = []
    for image, p in zip(images, prompts):
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": p},
            ],
        }])

    return messages

@catch_exceptions
def build_messages_jinaai(images, prompt):
    num_images = len(images)

    if isinstance(prompt, str):
        prompts = [prompt] * num_images

    elif isinstance(prompt, list):
        if len(prompt) != num_images:
            raise ValueError(
                f"Prompt list length ({len(prompt)}) "
                f"does not match number of images ({num_images})"
            )
        prompts = prompt

    else:
        raise TypeError("prompt must be a str or list[str]")

    messages = []
    for image, p in zip(images, prompts):
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", 'image': image},
                        {"type": "text", "text": p},
                    ],
                }
            ]
        )

    return messages

# ==========================================================================
### Loading Qwen-VL-2-7B ###
def load_qwen_vl_2_7B(model_name="Qwen/Qwen2-VL-7B-Instruct"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def infer_qwen_vl_2_7B(model, processor, images, prompt):
    messages = build_messages_qwen(images, prompt)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

# ==========================================================================
### Loading Qwen-VL-2.5-7B ###
def load_qwen_vl_2_5_7B(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def infer_qwen_vl_2_5_7B(model, processor, images, prompt):
    messages = build_messages_qwen(images, prompt)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

# ==========================================================================
### Loading Qwen-VL-2.5-32B ###
def load_qwen_vl_2_5_32B(model_name="Qwen/Qwen2.5-VL-32B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def infer_qwen_vl_2_5_32B(model, processor, images, prompt):
    messages = build_messages_qwen(images, prompt)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

# ==========================================================================
### Loading Paligemma2-3B###
def load_paligemma_2_3B(model_name='google/paligemma2-3b-mix-448'):
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()
    processor = PaliGemmaProcessor.from_pretrained(model_name)

    return model, processor

def infer_paligemma_2_3B(model, processor, image, prompt):

    prompt = "answer en <image> <bos> " + prompt
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded
    
# ==========================================================================
### Loading InternVL3-8B###  
def load_internvl3_8B(model_name="OpenGVLab/InternVL3-8B"):
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def infer_internvl3_8B(model, tokenizer, images, prompt):
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    pixel_values, num_patches_list, prompts = build_messages_intern(images,prompt)


    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=prompts,
        generation_config=generation_config
    )

    return responses
# ==========================================================================
### Loading InternVL3-14B###  
def load_internvl3_14B(model_name="OpenGVLab/InternVL3-14B"):
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def infer_internvl3_14B(model, tokenizer, images, prompt):
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    pixel_values, num_patches_list, prompts = build_messages_intern(images,prompt)

    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=prompts,
        generation_config=generation_config
    )

    return responses

# ==========================================================================
### Loading InternVL3-38B###  
device_map=split_model("OpenGVLab/InternVL3-38B")
def load_internvl3_38B(model_name="OpenGVLab/InternVL3-38B"):
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map=device_map
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def infer_internvl3_38B(model, tokenizer, images, prompt):
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    pixel_values, num_patches_list, prompts = build_messages_intern(images,prompt)

    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=prompts,
        generation_config=generation_config,
    )
    return responses

# ==========================================================================
### Loading Blip2-opt-6.7B###  
def load_blip2_opt_6_7B(model_name="Salesforce/blip2-opt-6.7b"):
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, device_map='auto', dtype=torch.bfloat16
    )
    return model, processor

def infer_blip2_opt_6_7B(model, processor, image, prompt):
    inputs = processor(images=image, text=f"{prompt} Answer:", return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, max_length=max_new_tokens, do_sample=do_sample)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

# ==========================================================================
### Loading Phi-3-vision-128k-instruct ###  

### Use transformers==4.40.0
def load_phi_3_vision_128k_instruct(model_name = "microsoft/Phi-3-vision-128k-instruct"):

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16, _attn_implementation='eager')

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True) 
    return model, processor

def infer_phi_3_vision_128k_instruct(model, processor, image, prompt):
    messages = [ 
        {"role": "user", "content": f"<|image_1|>\n{prompt}"}, 
    ] 

    input = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(input, [image], return_tensors="pt").to("cuda") 

    generation_args = { 
        "max_new_tokens": max_new_tokens, 
        "temperature": 0.0, 
        "do_sample": do_sample, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    return response
# ==========================================================================
### Loading Idefics3-8B-Llama3### 
def load_idefics3_8B(model_name="HuggingFaceM4/Idefics3-8B-Llama3"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(DEVICE)

    return model, processor

def infer_idefics3_8B(model, processor, images, prompt):
    messages = build_messages_idefic(images, prompt)
    input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts

# ==========================================================================
###Loading Llava-1.5-7b-hf ### 
def load_llava_1_5_7b(model_name="llava-hf/llava-1.5-7b-hf"):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor

def infer_llava_1_5_7b(model, processor, image, prompt):
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    input = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=input, return_tensors='pt').to("cuda")

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = output[0][prompt_len:]
    response = processor.decode(generated_tokens, skip_special_tokens=True)

    return response
# ==========================================================================
###Loadding Llama-3.2-11B-Vision-Instruct### 
def load_llama_3_2_11_b(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def infer_llama_3_2_11_b(model, processor, image, prompt):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to("cuda")

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return processor.decode(output[0])

# ==========================================================================
### Loading MiniCPM-V-2_6### 
def load_minicpm_2_6(model_name="openbmb/MiniCPM-V-2_6"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
        attn_implementation='eager', dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer

def infer_minicpm_2_6(model, tokenizer, image, prompt):
    msgs = [{'role': 'user', 'content': [image, prompt]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False, 
        temperature=0.0
    )
    return res
# ==========================================================================
###Loading microsoft/Phi-3.5-vision-instruct ### 

### Use `transformers==4.40.0`
def load_phi_3_5(model_name="microsoft/Phi-3.5-vision-instruct"):
    model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cuda", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    _attn_implementation='eager'    
    )

    processor = AutoProcessor.from_pretrained(model_name, 
    trust_remote_code=True, 
    num_crops=16
    ) 
    return model, processor


def infer_phi_3_5(model, processor, image, prompt):
    messages = [ 
        {"role": "user", "content": f"<|image_1|>\n{prompt}"}, 
    ] 

    prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": max_new_tokens, 
        "temperature": 0.0, 
        "do_sample": do_sample, 
    } 

    generate_ids = model.generate(**inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0] 

    return response

# ==========================================================================
### ### 
def load_smol_256m(model_name="HuggingFaceTB/SmolVLM-256M-Instruct"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to('cuda')

    return model, processor

def infer_smol_256m(model, processor, image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to('cuda')

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0].split('Assistant:')[1]
# ==========================================================================
### Loading Jinaai### 
def load_jinaai(model_name='jinaai/jina-vlm'):
    processor = AutoProcessor.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        'jinaai/jina-vlm',
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        trust_remote_code=True
    )

    return model, processor

def infer_jinaai(model, processor, images, prompt):
    conversations = build_messages_jinaai(images, prompt)

    texts = processor.apply_chat_template(conversations, add_generation_prompt=True)
    inputs = processor(text=texts, images=images, padding='longest', return_tensors='pt')
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        generation_config=GenerationConfig(max_new_tokens=1024, do_sample=False),
        return_dict_in_generate=True,
        use_model_defaults=True,
    )

    responses = []

    for idx in range(len(output.sequences)):
        gen_ids = output.sequences[idx][inputs['input_ids'].shape[-1]:]
        response = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(response)

    return responses
# ==========================================================================
### ### 

# ==========================================================================
### ### 

# ==========================================================================
### ### 

# ==========================================================================
### ### 

# ==========================================================================
### Loading Judge### 
def load_judge(model_name="microsoft/Phi-3.5-mini-instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cuda", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    return model, tokenizer

def infer_judge(model, tokenizer, prompt, output):
    #Put your prompt here
    messages = [
        {"role": "system", "content": "You are a strict post-processing function. You must return ONLY a single numeric value. If no numeric value can be confidently extracted, return EXACTLY: NaN."},
        {"role": "user", "content": f"""
         Original prompt: {prompt}
         Model output: {output}
         Task:
         - If the output contains a single numeric value, return it.
         - If it contains multiple numbers, return the most relevant one.
         - If no number is present, return NaN.
         - Do NOT include words, LaTeX, symbols, boxes, formatting, or punctuation.
         Return only the value. No text. No explanation.
        """},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 10,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": do_sample,
    }

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

def apply_judge_to_df(df, model, tokenizer):
    judged_values = []

    for _, row in df.iterrows():
        val = infer_judge(
            model,
            tokenizer,
            row["prompt"],
            row["output"]
        )
        judged_values.append(val)

    df_new = df.copy()
    df_new["judge"] = judged_values
    return df_new
# ==========================================================================