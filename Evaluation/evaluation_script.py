import os

# -----------------------------
# Environment & Thread Control
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# -----------------------------
# Seeds for reproducibility
# -----------------------------
SEED = 42
import random, numpy as np, torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Deterministic GPU behavior
# -----------------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# PyTorch thread control
# -----------------------------
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# -----------------------------
# Other imports
# -----------------------------
import inspect
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import math
import ast

import sys
sys.path.append(os.path.dirname(__file__))
import model_inference
import gc

# ==========================================================================
### Sanity Check ###
# ==========================================================================
def check():
    print("The evaluation module is correctly imported.")

# ==========================================================================
### Helper Function ###
# ==========================================================================
def generate_colormap_images(img_path, row, colormaps, done):
    img = cv.imread(img_path)
    if img is None:
        return

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for cmap in colormaps:
        key = (img_path, cmap)
        if key in done:
            continue

        if cmap == "Gray":
            img_rgb = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)
        else:
            img_color = cv.applyColorMap(
                img_gray,
                getattr(cv, f"COLORMAP_{cmap.upper()}")
            )
            img_rgb = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

        pil_image = Image.fromarray(img_rgb)

        row_out = row.to_dict()
        row_out["colormap"] = cmap

        yield pil_image, row_out

def thermal_with_cbar(
    tiff_path,
    gap_px=10,
    cbar_width_px=12,
    cbar_height_frac=0.9,
    direction="right",
    dpi=100,
    cmap='gray',
    px_grid=False,
    px_tick_size=10,
    normalize=False
):
    temp = tiff.imread(tiff_path)
    h, w = temp.shape

    if direction in ["left", "right"]:
        cbar_margin = 45
    elif direction in ["top", "bottom"]:
        cbar_margin = 22
    else:
        cbar_margin = 0
        
    axes_margin = 35 if px_grid else 0 
    top_buffer = 5 if px_grid else 0 

    side_buffer = 7 if direction in ["top", "bottom"] else 0

    if direction in ["right", "left"]:
        total_width_px = w + gap_px + cbar_width_px + cbar_margin + axes_margin
        total_height_px = h + axes_margin + top_buffer
    elif direction in ["bottom", "top"]:
        total_width_px = w + axes_margin + (side_buffer * 2)
        total_height_px = h + gap_px + cbar_width_px + cbar_margin + axes_margin + top_buffer
    else:
        total_width_px, total_height_px = w + axes_margin, h + axes_margin + top_buffer

    fig = plt.figure(figsize=(total_width_px/dpi, total_height_px/dpi), dpi=dpi)

    wf, hf = w / total_width_px, h / total_height_px
    gwf = gap_px / total_width_px
    ghf = gap_px / total_height_px
    cbwf = cbar_width_px / total_width_px
    cbhf = cbar_width_px / total_height_px
    cmwf = cbar_margin / total_width_px
    cmhf = cbar_margin / total_height_px
    amwf = axes_margin / total_width_px
    amhf = axes_margin / total_height_px
    sbf = side_buffer / total_width_px

    if direction == "right":
        ax_img = fig.add_axes([amwf, amhf, wf, hf])
        ax_cbar = fig.add_axes([amwf + wf + gwf, amhf + (hf*(1-cbar_height_frac)/2), cbwf, hf*cbar_height_frac])
        tick_pos = "right"

    elif direction == "left":
        ax_cbar = fig.add_axes([amwf, amhf + (hf*(1-cbar_height_frac)/2), cbwf, hf*cbar_height_frac])
        ax_img = fig.add_axes([amwf + cbwf + cmwf + gwf, amhf, wf, hf])
        tick_pos = "right"

    elif direction == "top":
        ax_cbar = fig.add_axes([amwf + sbf + (wf*(1-cbar_height_frac)/2), (total_height_px - cbar_width_px)/total_height_px, wf*cbar_height_frac, cbhf])
        ax_img = fig.add_axes([amwf + sbf, amhf, wf, hf])
        tick_pos = "bottom"

    elif direction == "bottom":
        ax_img = fig.add_axes([amwf + sbf, amhf + cbhf + cmhf + ghf, wf, hf])
        ax_cbar = fig.add_axes([amwf + sbf + (wf*(1-cbar_height_frac)/2), amhf + cmhf, wf*cbar_height_frac, cbhf])
        tick_pos = "bottom"
    
    else:
        ax_img = fig.add_axes([amwf, amhf, wf, hf])

    im = ax_img.imshow(temp, cmap=cmap, origin="upper", interpolation="nearest")
    
    if px_grid:
        ax_img.axis("on")
        xticks = np.linspace(0, w - 1, 5)
        yticks = np.linspace(0, h - 1, 5)

        ax_img.set_xticks(xticks)
        ax_img.set_yticks(yticks)

        if normalize:
            xlabels = [f"{x/(w-1):.2f}" for x in xticks]
            ylabels = [f"{y/(h-1):.2f}" for y in yticks]
        else:
            xlabels = [f"{int(x)}" for x in xticks]
            ylabels = [f"{int(y)}" for y in yticks]

        ax_img.set_xticklabels(xlabels, fontsize=px_tick_size)
        ax_img.set_yticklabels(ylabels, fontsize=px_tick_size)

        ax_img.tick_params(axis='both', which='both', length=3, pad=2)
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    else:
        ax_img.axis("off")

    if direction:
        orient = "horizontal" if direction in ["top", "bottom"] else "vertical"
        cbar = plt.colorbar(im, cax=ax_cbar, orientation=orient)
        vmin, vmax = float(np.min(temp)), float(np.max(temp))
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        labels = [f"{t:.1f}" for t in ticks]
        
        if direction in ["left", "right"]:
            cbar.ax.yaxis.set_ticks_position(tick_pos)
            cbar.ax.set_yticklabels(labels)
        else:
            cbar.ax.xaxis.set_ticks_position(tick_pos)
            cbar.ax.set_xticklabels(labels)
            
        cbar.ax.tick_params(labelsize=10, length=3, pad=4)

    fig.canvas.draw()
    w_final, h_final = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_final, w_final, 4))
    img_pil = Image.fromarray(buf[:, :, :3])
    plt.close(fig)
    return img_pil

def annotate_ring_arrow(img_pil, x, y, ring_radius=3, arrow_offset=30, ring_color="cyan", arrow_color="cyan", arrow_width=2, arrow_head_length=6):
    img = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    bbox = [x - ring_radius, y - ring_radius, x + ring_radius, y + ring_radius]
    draw.ellipse(bbox, outline=ring_color, width=1)

    dx, dy = -arrow_offset, -arrow_offset
    if x + dx < 0: dx = arrow_offset
    if x + dx > w: dx = -arrow_offset
    if y + dy < 0: dy = arrow_offset
    if y + dy > h: dy = -arrow_offset

    arrow_start = (x + dx, y + dy)

    vx, vy = x - arrow_start[0], y - arrow_start[1]
    length = (vx**2 + vy**2)**0.5
    if length > 0:
        ux, uy = vx / length, vy / length
        arrow_end = (x - ux * ring_radius, y - uy * ring_radius)
    else:
        arrow_end = (x, y)

    draw.line([arrow_start, arrow_end], fill=arrow_color, width=arrow_width)

    if length > 0:
        px, py = -uy, ux
        ah = arrow_head_length
        p1 = (arrow_end[0] - ux*ah + px*ah/2, arrow_end[1] - uy*ah + py*ah/2)
        p2 = (arrow_end[0] - ux*ah - px*ah/2, arrow_end[1] - uy*ah - py*ah/2)
        draw.polygon([arrow_end, p1, p2], fill=arrow_color)

    return img

# ==========================================================================
### Task 1 and Task 3 ###
# ==========================================================================
def evaluate_T1_T3(
    task_number,
    model_name,
    model, 
    processor,
    batch_size=8
):
    infer_model = getattr(model_inference, f"infer_{model_name}")

    prompts = {
        1: 'Is this a thermal image or an RGB image? Strictly answer in one word.', 
        3: 'How many people are in this image? If there are no people, return 0. Stricly answer in integer.'
        }

    prompt = prompts[task_number]

    supports_batch = "images" in inspect.signature(infer_model).parameters
    print(f'{model_name} supports batch: {supports_batch}')

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(BASE_DIR, "Labels", f"T{task_number}")
    output_dir = os.path.join(BASE_DIR, "Evaluation_Result", f"T{task_number}")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_name in csv_files:
        csv_type = os.path.splitext(csv_name)[0]
        input_csv = os.path.join(input_dir, csv_name)
        output_csv = os.path.join(output_dir, f"{csv_type}-{model_name}.csv")
        print(f'Output csv path: {output_csv}')

        print(f"\nProcessing {csv_name}")
        df = pd.read_csv(input_csv)

        if os.path.exists(output_csv):
            out_df = pd.read_csv(output_csv)
            done = set(out_df["image_path"])
        else:
            out_df = pd.DataFrame()
            done = set()

        df = df[~df["image_path"].isin(done)]
        if df.empty:
            print("Nothing left to process.")
            continue

        if supports_batch:
            for i in tqdm(range(0, len(df), batch_size)):
                batch = df.iloc[i:i+batch_size].copy()

                images = [
                    Image.open(p).convert("RGB")
                    for p in batch["image_path"]
                ]

                outputs = infer_model(
                    model,
                    processor,
                    images=images,
                    prompt=prompt
                )

                batch["prompt"] = prompt
                batch["output"] = outputs

                out_df = pd.concat([out_df, batch], ignore_index=True)
                out_df.to_csv(output_csv, index=False)

        else:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                image = Image.open(row["image_path"]).convert("RGB")

                out = infer_model(
                    model,
                    processor,
                    image=image,
                    prompt=prompt
                )

                row_out = row.to_dict()
                row_out["prompt"] = prompt
                row_out["output"] = out

                out_df = pd.concat(
                    [out_df, pd.DataFrame([row_out])],
                    ignore_index=True
                )
                out_df.to_csv(output_csv, index=False)

# ==========================================================================
### Task 2 ###
# ==========================================================================
def evaluate_T2(model_name, model, processor, batch_size=8):

    infer_model = getattr(model_inference, f"infer_{model_name}")

    supports_batch = "images" in inspect.signature(infer_model).parameters
    print(f"{model_name} supports batch: {supports_batch}")

    prompt = "Is this a thermal image or an RGB image? Strictly answer in one word."

    colormaps = ["Gray", "Magma", "Viridis", "Spring", "Summer"]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(BASE_DIR, "Labels", "T2")
    output_dir = os.path.join(BASE_DIR, "Evaluation_Result", "T2")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_name in csv_files:
        csv_type = os.path.splitext(csv_name)[0]
        input_csv = os.path.join(input_dir, csv_name)
        output_csv = os.path.join(output_dir, f"{csv_type}-{model_name}.csv")

        print(f"\nProcessing {csv_name}")
        print(f"Output csv path: {output_csv}")

        df = pd.read_csv(input_csv)

        if os.path.exists(output_csv):
            out_df = pd.read_csv(output_csv)
            done = set(zip(out_df["image_path"], out_df["colormap"]))
        else:
            out_df = pd.DataFrame()
            done = set()

        batch_images = []
        batch_rows = []

        def flush_batch():
            nonlocal out_df, batch_images, batch_rows
            if not batch_images:
                return

            if supports_batch:
                outputs = infer_model(
                    model,
                    processor,
                    images=batch_images,
                    prompt=prompt
                )
            else:
                outputs = [
                    infer_model(
                        model,
                        processor,
                        image=img,
                        prompt=prompt
                    )
                    for img in batch_images
                ]

            assert len(outputs) == len(batch_rows), "Output-image count mismatch"

            for row_dict, out in zip(batch_rows, outputs):
                row_dict["prompt"] = prompt
                row_dict["output"] = out

            out_df = pd.concat(
                [out_df, pd.DataFrame(batch_rows)],
                ignore_index=True
            )
            out_df.to_csv(output_csv, index=False)

            batch_images.clear()
            batch_rows.clear()

        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = row["image_path"]

            for pil_image, row_out in generate_colormap_images(
                img_path, row, colormaps, done
            ):
                batch_images.append(pil_image)
                batch_rows.append(row_out)

                if len(batch_images) == batch_size:
                    flush_batch()

        flush_batch()

# ==========================================================================
### Task 4 ###
# ==========================================================================
def evaluate_T4(model_name, model, processor, batch_size=8):
    infer_model = getattr(model_inference, f"infer_{model_name}")

    supports_batch = "images" in inspect.signature(infer_model).parameters
    print(f"{model_name} supports batch: {supports_batch}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tiff_dir = os.path.join(BASE_DIR, "Dataset/ThermEval-D/Temp_Matrix")
    output_dir = os.path.join(BASE_DIR, "Evaluation_Result", "T4")
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = sorted(f for f in os.listdir(tiff_dir) if f.endswith(".tiff"))

    def flush_batch(batch_images, batch_rows, out_df, output_csv):
        if not batch_images:
            return out_df

        if supports_batch:
            outputs = infer_model(
                model,
                processor,
                images=batch_images,
                prompt=[row["prompt"] for row in batch_rows]
            )
        else:
            outputs = [
                infer_model(
                    model,
                    processor,
                    image=img,
                    prompt=row["prompt"]
                )
                for img, row in zip(batch_images, batch_rows)
            ]

        for row_dict, out in zip(batch_rows, outputs):
            row_dict["output"] = out

        out_df = pd.concat([out_df, pd.DataFrame(batch_rows)], ignore_index=True)
        out_df.to_csv(output_csv, index=False)

        batch_images.clear()
        batch_rows.clear()
        return out_df

    # ==================================================================
    # Subtask 1: Colorbar Detection
    # ==================================================================
    output_csv = os.path.join(output_dir, f"Detection-{model_name}.csv")

    if os.path.exists(output_csv):
        out_df1 = pd.read_csv(output_csv)
        done = set(zip(out_df1["tiff_file"], out_df1["ground_truth"]))
    else:
        out_df1 = pd.DataFrame()
        done = set()

    batch_images, batch_rows = [], []

    for tiff_file in tqdm(tiff_files, desc="T4-Detection"):
        tiff_path = os.path.join(tiff_dir, tiff_file)

        direction = random.choice(["right", "left", "top", "bottom"])
        img_with_cbar = thermal_with_cbar(tiff_path, direction=direction)
        img_without_cbar = thermal_with_cbar(tiff_path, direction=None)

        data = [
            (
                img_with_cbar,
                "You are given a thermal image. Does it contain a color bar or temperature scale that maps colors to temperature values? Answer only with 'Yes' or 'No'. Strictly answer in one word.",
            ),
            (
                img_without_cbar,
                "You are given a thermal image. Does it contain a color bar or temperature scale that maps colors to temperature values? Answer only with 'Yes' or 'No'. Strictly answer in one word.",
                "no"
            ),
        ]

        for img, prompt, gt in data:
            key = (tiff_file, gt)
            if key in done:
                continue

            batch_images.append(img)
            batch_rows.append({
                "tiff_file": tiff_file,
                "prompt": prompt,
                "ground_truth": gt
            })
            done.add(key)

            if len(batch_images) >= batch_size:
                out_df1 = flush_batch(batch_images, batch_rows, out_df1, output_csv)

    out_df1 = flush_batch(batch_images, batch_rows, out_df1, output_csv)

    # ==================================================================
    # Subtask 2: Colorbar Direction
    # ==================================================================
    output_csv = os.path.join(output_dir, f"Position-{model_name}.csv")

    if os.path.exists(output_csv):
        out_df2 = pd.read_csv(output_csv)
        done = set(zip(out_df2["tiff_file"], out_df2["ground_truth"]))
    else:
        out_df2 = pd.DataFrame()
        done = set()

    batch_images, batch_rows = [], []

    directions = ["right", "left", "top", "bottom"]

    for tiff_file in tqdm(tiff_files, desc="T4-Position"):
        tiff_path = os.path.join(tiff_dir, tiff_file)

        for dir in directions:
            key = (tiff_file, dir)
            if key in done:
                continue

            img = thermal_with_cbar(tiff_path, direction=dir)
            prompt = "You are given a thermal image. It contains a color bar or temperature scale that maps colors to temperature value. What is the location of the colorbar? Possible locations are top, left, bottom, right. Strictly answer in one word."

            batch_images.append(img)
            batch_rows.append({
                "tiff_file": tiff_file,
                "prompt": prompt,
                "ground_truth": dir
            })
            done.add(key)

            if len(batch_images) >= batch_size:
                out_df2 = flush_batch(batch_images, batch_rows, out_df2, output_csv)

    out_df2 = flush_batch(batch_images, batch_rows, out_df2, output_csv)

    # ==================================================================
    # Subtask 3: Colorbar Minimum/Maximum 
    # ==================================================================
    output_csv = os.path.join(output_dir, f"Min-Max-{model_name}.csv")

    if os.path.exists(output_csv):
        out_df3 = pd.read_csv(output_csv)
        done = set(zip(out_df3["tiff_file"], out_df3["type"]))
    else:
        out_df3 = pd.DataFrame()
        done = set()

    batch_images, batch_rows = [], []

    for tiff_file in tqdm(tiff_files, desc="T4-Extraction"):
        tiff_path = os.path.join(tiff_dir, tiff_file)
        temp = tiff.imread(tiff_path)
        img = thermal_with_cbar(tiff_path, direction="right")

        for t_type, gt in [
            ("maximum", float(temp.max())),
            ("minimum", float(temp.min()))
        ]:
            key = (tiff_file, t_type)
            if key in done:
                continue

            prompt = f"You are given a thermal image with a color bar or temperature scale that maps colors to temperature value. What is the {t_type} temperature value in degree Celsius? Strictly return a single numerical value rounded to one decimal place?"

            batch_images.append(img)
            batch_rows.append({
                "tiff_file": tiff_file,
                "prompt": prompt,
                "type": t_type,
                "ground_truth": gt
            })
            done.add(key)

            if len(batch_images) >= batch_size:
                out_df3 = flush_batch(batch_images, batch_rows, out_df3, output_csv)

    out_df3 = flush_batch(batch_images, batch_rows, out_df3, output_csv)

# ==========================================================================
### Task 5 ###
# ==========================================================================
def evaluate_T5(model_name, model, processor, batch_size=8):
    infer_model = getattr(model_inference, f"infer_{model_name}")

    supports_batch = "images" in inspect.signature(infer_model).parameters
    print(f"{model_name} supports batch: {supports_batch}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(BASE_DIR, "Labels", "T5")
    output_dir = os.path.join(BASE_DIR, "Evaluation_Result", "T5")
    os.makedirs(output_dir, exist_ok=True)

    tiff_dir = os.path.join(BASE_DIR, "Dataset")

    def flush_batch(batch_images, batch_rows, out_df, output_csv):
        if not batch_images:
            return out_df

        if supports_batch:
            outputs = infer_model(
                model,
                processor,
                images=batch_images,
                prompt=[row["prompt"] for row in batch_rows]
            )
        else:
            outputs = [
                infer_model(model, processor, image=img, prompt=row["prompt"])
                for img, row in zip(batch_images, batch_rows)
            ]

        for row_dict, out in zip(batch_rows, outputs):
            row_dict["output"] = out

        out_df = pd.concat([out_df, pd.DataFrame(batch_rows)], ignore_index=True)
        out_df.to_csv(output_csv, index=False)

        batch_images.clear()
        batch_rows.clear()
        return out_df

    # single.csv
    single_csv = os.path.join(input_dir, "single.csv")
    df_single = pd.read_csv(single_csv)

    output_csv_single = os.path.join(output_dir, f"T5_single-{model_name}.csv")

    if os.path.exists(output_csv_single):
        out_df_single = pd.read_csv(output_csv_single)
        done = set(zip(out_df_single["tiff_file"], out_df_single["ground_truth"]))
    else:
        out_df_single = pd.DataFrame()
        done = set()

    full_order = ["Forehead", "Chest", "Nose"]

    batch_images, batch_rows = [], []

    for _, row in tqdm(df_single.iterrows(), total=len(df_single), desc="T5-Single"):
        tiff_file = row["tiff_file"]
        tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")

        gt = row["ground_truth"]
        key = (tiff_file, gt)
        if key in done:
            continue

        gt_list = ast.literal_eval(gt)
        ordered_parts = [part for part in full_order if part in gt_list]

        prompt = f"Given the thermal image and the colourbar, rank the following body parts in order from highest to lowest temperature: {', '.join(ordered_parts).lower()}. List them from hottest to coolest. Strictly return a list of body parts."

        img = thermal_with_cbar(tiff_path, direction="right")

        batch_images.append(img)
        batch_rows.append({
            "tiff_file": tiff_file,
            "prompt": prompt,
            "ground_truth": gt
        })
        done.add(key)

        if len(batch_images) >= batch_size:
            out_df_single = flush_batch(
                batch_images, batch_rows, out_df_single, output_csv_single
            )

    out_df_single = flush_batch(
        batch_images, batch_rows, out_df_single, output_csv_single
    )

    # double.csv
    double_csv = os.path.join(input_dir, "double.csv")
    df_double = pd.read_csv(double_csv)

    output_csv_double = os.path.join(output_dir, f"T5_double-{model_name}.csv")

    if os.path.exists(output_csv_double):
        out_df_double = pd.read_csv(output_csv_double)
        done = set(zip(out_df_double["tiff_file"], out_df_double["category"]))
    else:
        out_df_double = pd.DataFrame()
        done = set()

    batch_images, batch_rows = [], []

    for _, row in tqdm(df_double.iterrows(), total=len(df_double), desc="T5-Double"):
        tiff_file = row["tiff_file"]
        tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")

        category = row["category"]
        key = (tiff_file, category)
        if key in done:
            continue

        prompt = f"Given the thermal image with colourbar, determine whether the {category} of the left person or the right person is hotter. Respond with only 'left' or 'right'. Strictly answer in one single word."

        gt = row["ground_truth"]

        img = thermal_with_cbar(tiff_path, direction="right")

        batch_images.append(img)
        batch_rows.append({
            "tiff_file": tiff_file,
            "category": category,
            "prompt": prompt,
            "ground_truth": gt
        })
        done.add(key)

        if len(batch_images) >= batch_size:
            out_df_double = flush_batch(
                batch_images, batch_rows, out_df_double, output_csv_double
            )

    out_df_double = flush_batch(
        batch_images, batch_rows, out_df_double, output_csv_double
    )

# ==========================================================================
### Task 6 ###
# ==========================================================================
def evaluate_T6(model_name, model, processor, batch_size=8):
    infer_model = getattr(model_inference, f"infer_{model_name}")

    supports_batch = "images" in inspect.signature(infer_model).parameters
    print(f"{model_name} supports batch: {supports_batch}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(BASE_DIR, "Labels", "T6")
    output_dir = os.path.join(BASE_DIR, "Evaluation_Result", "T6")
    os.makedirs(output_dir, exist_ok=True)

    tiff_dir = os.path.join(BASE_DIR, "Dataset")

    def flush_batch(batch_images, batch_rows, out_df, output_csv):
        if not batch_images:
            return out_df

        if supports_batch:
            outputs = infer_model(
                model,
                processor,
                images=batch_images,
                prompt=[row["prompt"] for row in batch_rows]
            )
        else:
            outputs = [
                infer_model(model, processor, image=img, prompt=row["prompt"])
                for img, row in zip(batch_images, batch_rows)
            ]

        for row_dict, out in zip(batch_rows, outputs):
            row_dict["output"] = out

        out_df = pd.concat([out_df, pd.DataFrame(batch_rows)], ignore_index=True)
        out_df.to_csv(output_csv, index=False)

        batch_images.clear()
        batch_rows.clear()
        return out_df

    # ===============================================
    # Subtask 1: Temperature Estimation at Coordinate
    # ===============================================
    coords_csv = os.path.join(input_dir, "coords.csv")
    df_coords = pd.read_csv(coords_csv)

    output_csv1 = os.path.join(output_dir, f"Coords-{model_name}.csv")
    if os.path.exists(output_csv1):
        out_df1 = pd.read_csv(output_csv1)
        done = set(zip(out_df1["tiff_file"], out_df1["prompt"]))
    else:
        out_df1 = pd.DataFrame()
        done = set()

    batch_images, batch_rows = [], []

    for _, row in tqdm(df_coords.iterrows(), total=len(df_coords), desc="T6-Coords"):
        tiff_file = row["tiff_file"]
        tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")
        prompt = f"Given the thermal image, what is the temperature at the coordinates ({row['x']},{row['y']})? The temperature scale is in degrees Celsius. Strictly return a single numerical value rounded to one decimal place?"

        gt = row["ground_truth"]

        key = (tiff_file, prompt)
        if key in done:
            continue

        img = thermal_with_cbar(tiff_path, direction="right", px_grid=True)

        batch_images.append(img)
        batch_rows.append({
            "tiff_file": tiff_file,
            "prompt": prompt,
            "ground_truth": gt
        })
        done.add(key)

        if len(batch_images) >= batch_size:
            out_df1 = flush_batch(batch_images, batch_rows, out_df1, output_csv1)

    out_df1 = flush_batch(batch_images, batch_rows, out_df1, output_csv1)

    # ===========================================
    # Subtask 2: Temperature Estimation at Marker
    # ===========================================

    def flush_batch_arrow(batch_images, batch_rows, out_df, output_csv, done):
        if not batch_images:
            return out_df

        if supports_batch:
            outputs = infer_model(
                model,
                processor,
                images=batch_images,
                prompt=[row["prompt"] for row in batch_rows]
            )
        else:
            outputs = [
                infer_model(model, processor, image=img, prompt=row["prompt"])
                for img, row in zip(batch_images, batch_rows)
            ]

        for row_dict, out in zip(batch_rows, outputs):
            row_dict["output"] = out
            done.add((row_dict["tiff_file"], row_dict["x"], row_dict["y"]))

        out_df = pd.concat([out_df, pd.DataFrame(batch_rows)], ignore_index=True)
        out_df.to_csv(output_csv, index=False)

        batch_images.clear()
        batch_rows.clear()
        return out_df

    output_csv2 = os.path.join(output_dir, f"Arrow-{model_name}.csv")
    if os.path.exists(output_csv2):
        out_df2 = pd.read_csv(output_csv2)
        done = set(zip(out_df2["tiff_file"], out_df2["x"], out_df2["y"]))
    else:
        out_df2 = pd.DataFrame()
        done = set()

    batch_images, batch_rows = [], []

    for _, row in tqdm(df_coords.iterrows(), total=len(df_coords), desc="T6-Arrow"):
        tiff_file = row["tiff_file"]
        tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")
        prompt = "Given the thermal image, what is the temperature of the pixel at the center of the cyan ring indicated by the cyan arrow? The temperature scale is in degrees Celsius. Strictly return a single numerical value rounded to one decimal place."

        gt = row["ground_truth"]

        key = (tiff_file, row["x"], row["y"])
        if key in done:
            continue

        img = thermal_with_cbar(tiff_path, direction="right")
        img = annotate_ring_arrow(img, row["x"], row["y"])

        batch_images.append(img)
        batch_rows.append({
            "tiff_file": tiff_file,
            "x": row["x"],
            "y": row["y"],
            "prompt": prompt,
            "ground_truth": gt
        })

        if len(batch_images) >= batch_size:
            out_df2 = flush_batch_arrow(batch_images, batch_rows, out_df2, output_csv2, done)

    out_df2 = flush_batch_arrow(batch_images, batch_rows, out_df2, output_csv2, done)

    # ==================================================================
    # Subtask 3: Region Temperature Estimation (single.csv + double.csv)
    # ==================================================================
    output_csv3 = os.path.join(output_dir, f"Region-{model_name}.csv")
    if os.path.exists(output_csv3):
        out_df3 = pd.read_csv(output_csv3)
        done = set(zip(out_df3["tiff_file"], out_df3["prompt"]))
    else:
        out_df3 = pd.DataFrame()
        done = set()

    batch_images, batch_rows = [], []

    # double.csv
    double_csv = os.path.join(input_dir, "double.csv")
    if os.path.exists(double_csv):
        df_double = pd.read_csv(double_csv)
        for _, row in tqdm(df_double.iterrows(), total=len(df_double), desc="T6-Region-Double"):
            tiff_file = row["tiff_file"]
            tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")
            prompt = f"Given the thermal image, what is the temperature estimate of the {row['category']} of the {row['side']} person according to the image? The temperature scale is in degrees Celsius. Strictly return a single numerical value rounded to one decimal place."

            gt = row["ground_truth"]

            key = (tiff_file, prompt)
            if key in done:
                continue

            img = thermal_with_cbar(tiff_path, direction="right")
            batch_images.append(img)
            batch_rows.append({
                "tiff_file": tiff_file,
                "prompt": prompt,
                "ground_truth": gt
            })
            done.add(key)

            if len(batch_images) >= batch_size:
                out_df3 = flush_batch(batch_images, batch_rows, out_df3, output_csv3)

    # single.csv
    single_csv = os.path.join(input_dir, "single.csv")
    if os.path.exists(single_csv):
        df_single = pd.read_csv(single_csv)
        for _, row in tqdm(df_single.iterrows(), total=len(df_single), desc="T6-Region-Single"):
            tiff_file = row["tiff_file"]
            tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")
            prompt = f"Given the thermal image, what is the temperature estimate of the {row['category']} according to the image? The temperature scale is in degrees Celsius. Strictly return a single numerical value rounded to one decimal place."

            gt = row["ground_truth"]

            key = (tiff_file, prompt)
            if key in done:
                continue

            img = thermal_with_cbar(tiff_path, direction="right")
            batch_images.append(img)
            batch_rows.append({
                "tiff_file": tiff_file,
                "prompt": prompt,
                "ground_truth": gt
            })
            done.add(key)

            if len(batch_images) >= batch_size:
                out_df3 = flush_batch(batch_images, batch_rows, out_df3, output_csv3)

    out_df3 = flush_batch(batch_images, batch_rows, out_df3, output_csv3)

# ==========================================================================
### Task 7 ###
# ==========================================================================
def evaluate_T7(model_name, model, processor, batch_size=8):
    infer_model = getattr(model_inference, f"infer_{model_name}")

    supports_batch = "images" in inspect.signature(infer_model).parameters
    print(f"{model_name} supports batch: {supports_batch}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(BASE_DIR, "Labels", "T7")
    output_dir = os.path.join(BASE_DIR, "Evaluation_Result", "T7")
    os.makedirs(output_dir, exist_ok=True)

    tiff_dir = os.path.join(BASE_DIR, "Dataset")

    def flush_batch(batch_images, batch_rows, out_df, output_csv):
        if not batch_images:
            return out_df

        if supports_batch:
            outputs = infer_model(
                model,
                processor,
                images=batch_images,
                prompt=[row["prompt"] for row in batch_rows]
            )
        else:
            outputs = [
                infer_model(
                    model,
                    processor,
                    image=img,
                    prompt=row["prompt"]
                )
                for img, row in zip(batch_images, batch_rows)
            ]

        for row_dict, out in zip(batch_rows, outputs):
            row_dict["output"] = out

        out_df = pd.concat([out_df, pd.DataFrame(batch_rows)], ignore_index=True)
        out_df.to_csv(output_csv, index=False)

        batch_images.clear()
        batch_rows.clear()
        return out_df

    csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))

    for csv_file in csv_files:
        input_csv_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(input_csv_path)

        output_csv = os.path.join(
            output_dir,
            f"{os.path.splitext(csv_file)[0]}-{model_name}.csv"
        )

        if os.path.exists(output_csv):
            out_df = pd.read_csv(output_csv)
            done = set(
                zip(
                    out_df["tiff_file"],
                    out_df["category"]
                )
            )
        else:
            out_df = pd.DataFrame()
            done = set()

        batch_images, batch_rows = [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"T7-{csv_file}"):
            tiff_file = row["tiff_file"]
            tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")

            category = row["category"]
            distance = row["distance"]
            gt = row["ground_truth"]

            key = (tiff_file, category, distance)
            if key in done:
                continue

            prompt = f"Given the thermal image, what is the temperature estimate of the {category} according to the image? The temperature scale is in degrees Celsius. Strictly return a single numerical value rounded to one decimal place."

            img = thermal_with_cbar(tiff_path, direction="right")

            batch_images.append(img)
            batch_rows.append({
                "tiff_file": tiff_file,
                "category": category,
                "distance": distance,
                "prompt": prompt,
                "ground_truth": gt
            })

            done.add(key)

            if len(batch_images) >= batch_size:
                out_df = flush_batch(
                    batch_images, batch_rows, out_df, output_csv
                )

        out_df = flush_batch(
            batch_images, batch_rows, out_df, output_csv
        )

# ==========================================================================
### Task 8 ###
# ==========================================================================
def evaluate_T8(model_name, model, processor, batch_size=8):
    infer_model = getattr(model_inference, f"infer_{model_name}")

    supports_batch = "images" in inspect.signature(infer_model).parameters
    print(f"{model_name} supports batch: {supports_batch}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(BASE_DIR, "Labels", "T8")
    output_dir = os.path.join(BASE_DIR, "Evaluation_Result", "T8")
    os.makedirs(output_dir, exist_ok=True)

    tiff_dir = os.path.join(BASE_DIR, "Dataset")

    def flush_batch(batch_images, batch_rows, out_df, output_csv):
        if not batch_images:
            return out_df

        if supports_batch:
            outputs = infer_model(
                model,
                processor,
                images=batch_images,
                prompt=[row["prompt"] for row in batch_rows]
            )
        else:
            outputs = [
                infer_model(
                    model,
                    processor,
                    image=img,
                    prompt=row["prompt"]
                )
                for img, row in zip(batch_images, batch_rows)
            ]

        for row_dict, out in zip(batch_rows, outputs):
            row_dict["output"] = out

        out_df = pd.concat([out_df, pd.DataFrame(batch_rows)], ignore_index=True)
        out_df.to_csv(output_csv, index=False)

        batch_images.clear()
        batch_rows.clear()
        return out_df

    csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))

    for csv_file in csv_files:
        input_csv_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(input_csv_path)

        output_csv = os.path.join(
            output_dir,
            f"{os.path.splitext(csv_file)[0]}-{model_name}.csv"
        )

        if os.path.exists(output_csv):
            out_df = pd.read_csv(output_csv)
            done = set(
                zip(
                    out_df["tiff_file"],
                    out_df["category"]
                )
            )
        else:
            out_df = pd.DataFrame()
            done = set()

        batch_images, batch_rows = [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"T8-{csv_file}"):
            tiff_file = row["tiff_file"]
            tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")

            category = row["category"]
            gt = row["ground_truth"]

            key = (tiff_file, category)
            if key in done:
                continue

            prompt = f"Given the thermal image, return the normalized bounding box of the {category}. Format: x1,y1,x2,y2 where all values are between 0 and 1. Strictly return four comma-separated numbers only."

            
            img = thermal_with_cbar(tiff_path, direction="right")

            batch_images.append(img)
            batch_rows.append({
                "tiff_file": tiff_file,
                "category": category,
                "prompt": prompt,
                "ground_truth": gt
            })

            done.add(key)

            if len(batch_images) >= batch_size:
                out_df = flush_batch(
                    batch_images, batch_rows, out_df, output_csv
                )

        out_df = flush_batch(
            batch_images, batch_rows, out_df, output_csv
        )