import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import torch
import time
import os
from sentence_transformers import SentenceTransformer, util
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_CANDIDATES = {
    "Oil Painting LoRA": os.path.join(BASE_DIR, "models", "lora_finetuned_oil_painting"),
    "Anime LoRA (Backup)": os.path.join(BASE_DIR, "models", "lora_finetuned_anime"),
    "General LoRA": os.path.join(BASE_DIR, "models", "lora_finetuned"),
}


def has_lora_weights(lora_path):
    return os.path.exists(os.path.join(lora_path, "pytorch_lora_weights.safetensors"))


def get_available_loras():
    return {name: path for name, path in LORA_CANDIDATES.items() if has_lora_weights(path)}


def generate_image(pipe, prompt, negative_prompt, steps, guidance):
    # Recreate scheduler from config to avoid stale internal state across reruns.
    pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    return pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]


def generate_styled_image(pipe, prompt, negative_prompt, base_image, steps, guidance, style_strength):
    # Recreate scheduler from config to avoid stale internal state across reruns.
    pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        strength=style_strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

# ==================== CACHED PIPELINES ====================
@st.cache_resource(show_spinner=False)
def load_base_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe

@st.cache_resource(show_spinner=False)
def load_lora_pipeline(lora_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    if not has_lora_weights(lora_path):
        raise FileNotFoundError(f"LoRA weights not found at: {lora_path}")

    pipe.load_lora_weights(lora_path)
    return pipe

st.set_page_config(page_title="Text-to-Image MLOps Pipeline", layout="wide")
st.title("🖼️ Text-to-Image Generation End-to-End Pipeline")
st.markdown("**By:** Ravi Sharma (M25CSA024) & S Kartik Iyer (M25CSA025)")

st.sidebar.title("Pipeline Stages")
st.sidebar.markdown("1. Text Input  \n2. Tokenization & Encoding  \n3. Generative Model  \n4. Post Processing  \n5. Output")

tab1, tab2, tab3 = st.tabs(["🎨 A/B Testing", "📊 Drift Monitoring", "📋 LoRA Details"])

# ==================== TAB 1 - A/B TESTING ====================
with tab1:
    st.header("1. Text Input Processing")
    prompt = st.text_area("Base Prompt:",
                          "Bird sitting on a tree branch", height=130)
    style_prompt = st.text_input(
        "Style Prompt (used for conversion):",
        "oil painting style, visible brush strokes, rich texture, museum quality"
    )
    negative_prompt = st.text_input("Negative Prompt:", "blurry, low quality, deformed, painting, cartoon, multiple limbs")

    available_loras = get_available_loras()
    lora_labels = list(available_loras.keys())
    default_index = lora_labels.index("Oil Painting LoRA") if "Oil Painting LoRA" in lora_labels else 0
    selected_lora_label = st.selectbox("Select LoRA Style", lora_labels, index=default_index) if lora_labels else None
    selected_lora_path = available_loras[selected_lora_label] if selected_lora_label else None

    if not lora_labels:
        st.error("No valid LoRA weights found. Add a pytorch_lora_weights.safetensors file to one of the configured LoRA folders.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        steps = st.slider("Inference Steps", 15, 50, 25)
    with col2:
        guidance = st.slider("Guidance Scale", 5.0, 12.0, 7.5)
    with col3:
        lora_strength = st.slider("LoRA Strength", 0.6, 1.2, 0.85, 0.05)
    with col4:
        style_strength = st.slider("Style Transform Strength", 0.25, 0.9, 0.55, 0.05)

    base_col, anime_col = st.columns(2)
    with base_col:
        run_base = st.button("🧱 Generate Base Image", type="primary", use_container_width=True)
    with anime_col:
        run_style = st.button("🎨 Convert Base Image with Selected LoRA", use_container_width=True)

    if run_base:
        if not prompt.strip():
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating base image..."):
                start = time.time()

                pipe_base = load_base_pipeline()
                img_base = generate_image(pipe_base, prompt, negative_prompt, steps, guidance)
                img_base = img_base.resize((512, 512))

                st.session_state.img_base = img_base
                if 'img_lora' in st.session_state:
                    del st.session_state.img_lora
                st.success(f"✅ Base image generated in {time.time()-start:.1f} seconds")

    if run_style:
        if 'img_base' not in st.session_state:
            st.info("Generate the base image first.")
        elif not selected_lora_path:
            st.info("No LoRA style is available yet.")
        else:
            with st.spinner(f"Converting base image with {selected_lora_label}..."):
                start = time.time()

                pipe_lora = load_lora_pipeline(selected_lora_path)
                try:
                    pipe_lora.set_adapters("default", adapter_weights=lora_strength)
                except:
                    try:
                        pipe_lora.set_adapters("default_0", adapter_weights=lora_strength)
                    except:
                        pass

                img_lora = generate_styled_image(
                    pipe_lora,
                    style_prompt,
                    negative_prompt,
                    st.session_state.img_base,
                    steps,
                    guidance,
                    style_strength,
                )
                img_lora = img_lora.resize((512, 512))

                st.session_state.img_lora = img_lora
                st.session_state.selected_lora_label = selected_lora_label
                st.success(f"✅ Style conversion completed in {time.time()-start:.1f} seconds")

    # Display images OUTSIDE the button block so they persist
    if 'img_base' in st.session_state:
        st.subheader("Base Model")
        st.image(st.session_state.img_base, caption="Base Model Output", width=600)
        buf = BytesIO()
        st.session_state.img_base.save(buf, format="PNG")
        st.download_button("📥 Download Base Image", buf.getvalue(), "base_image.png", "image/png", key="dl_base")

    if 'img_lora' in st.session_state:
        st.subheader("LoRA Model")
        style_label = st.session_state.get("selected_lora_label", "Selected LoRA")
        st.image(st.session_state.img_lora, caption=f"{style_label} Output", width=600)
        buf = BytesIO()
        st.session_state.img_lora.save(buf, format="PNG")
        st.download_button("📥 Download LoRA Image", buf.getvalue(), "lora_image.png", "image/png", key="dl_lora")

with tab2:
    st.header("📊 Drift Monitoring")
    st.markdown("CLIP Image Similarity (higher = less drift)")

    if st.button("Calculate Drift", use_container_width=True):
        if 'img_base' in st.session_state and 'img_lora' in st.session_state:
            with st.spinner("Computing drift..."):
                embedder = SentenceTransformer('clip-ViT-B-32')
                similarity = util.cos_sim(
                    embedder.encode(st.session_state.img_base),
                    embedder.encode(st.session_state.img_lora)
                )[0][0].item()

                st.metric("CLIP Similarity", f"{similarity:.4f}")
                if similarity > 0.82:
                    st.success("Low Drift → Good generalization")
                elif similarity > 0.65:
                    st.info("Moderate Drift → Normal for LoRA")
                else:
                    st.warning("High Drift → Strong style shift")
        else:
            st.info("Generate base and stylized images first")

with tab3:

    st.subheader("How the End-to-End Pipeline Works")
    
    st.markdown("""
    **1. Text Input Processing**  
    User provides natural language prompt and optional negative prompt. Text is cleaned and prepared.

    **2. Tokenization & Encoding**  
    The prompt is tokenized using CLIP tokenizer and converted into text embeddings (conditioning).

    **3. Generative Model**  
    - **Base Model**: Uses Stable Diffusion v1.5 (U-Net + VAE + Scheduler)  
    - **LoRA Model**: Same base model + Low-Rank Adaptation weights applied on attention layers  
      → Allows style/control with very few trainable parameters.

    **4. Post Processing**  
    Generated image is resized to 512×512 and basic quality enhancements are applied.

    **5. Output**  
    Final image is displayed with download option. A/B comparison and drift monitoring are performed.
    """)

    st.markdown("""
    ### LoRA Training Summary:
    - **Technique**: PEFT LoRA (Rank-based adaptation)
    - **Target**: Attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`)
    - **Dataset**: General images (laion subset)
    - **Purpose**: To demonstrate parameter-efficient fine-tuning
    - **MLOps Value**: Shows model adaptation without retraining entire model
    """)


    st.header("📋 LoRA Details")
    for lora_name, lora_path in LORA_CANDIDATES.items():
        if has_lora_weights(lora_path):
            st.success(f"✅ {lora_name} available: {lora_path}")
        else:
            st.warning(f"⚠️ {lora_name} missing weights: {lora_path}")

    

st.caption("SD 1.5 + Real LoRA + A/B Testing + Drift Monitoring | RTX 2050")