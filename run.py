from ThermEval_Benchmark import evaluation_script, model_inference
import warnings
warnings.filterwarnings("ignore")

import gc
import torch

evaluation_script.check()
model_inference.check()

batch_size = 4

model_names = ['qwen_vl_2_5_32B', 'internvl3_38B']

for model_name in model_names:
    print(f'Loading {model_name}')

    model, processor = getattr(model_inference, f"load_{model_name}")()

    evaluation_script.evaluate_T1_T3(task_number=1, model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    evaluation_script.evaluate_T2(model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    evaluation_script.evaluate_T1_T3(3, model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    evaluation_script.evaluate_T4(model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    evaluation_script.evaluate_T5(model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    evaluation_script.evaluate_T6(model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    evaluation_script.evaluate_T7(model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    evaluation_script.evaluate_T8(model_name=model_name, model=model, processor=processor, batch_size=batch_size)

    print(f'Evaluation Complete of {model_name}')

    model.to("cpu")

    del model
    del processor

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("GPU cleaned\n")
    print("All models evaluated.")