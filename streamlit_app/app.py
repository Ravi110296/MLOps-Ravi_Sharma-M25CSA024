import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import time
import os
from sentence_transformers import SentenceTransformer, util
from io import BytesIO

# ==================== CACHED PIPELINES ====================
@st.cache_resource(show_spinner=False)
def load_base_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

@st.cache_resource(show_spinner=False)
def load_lora_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    lora_path = "../models/lora_finetuned"
    if os.path.exists(lora_path):
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
    prompt = st.text_area("Prompt (same for both models):", 
                          "A majestic dragon flying over snowy mountains at sunset", height=130)
    negative_prompt = st.text_input("Negative Prompt:", "blurry, low quality, deformed")

    col1, col2, col3 = st.columns(3)
    with col1:
        steps = st.slider("Inference Steps", 15, 50, 25)
    with col2:
        guidance = st.slider("Guidance Scale", 5.0, 12.0, 7.5)
    with col3:
        lora_strength = st.slider("LoRA Strength", 0.6, 1.2, 0.85, 0.05)

    if st.button("🚀 Run Side-by-Side A/B Test", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating images..."):
                start = time.time()

                pipe_base = load_base_pipeline()
                img_base = pipe_base(prompt, negative_prompt=negative_prompt,
                                     num_inference_steps=steps, guidance_scale=guidance).images[0]
                img_base = img_base.resize((512, 512))

                pipe_lora = load_lora_pipeline()
                try:
                    pipe_lora.set_adapters("default", adapter_weights=lora_strength)
                except:
                    try:
                        pipe_lora.set_adapters("default_0", adapter_weights=lora_strength)
                    except:
                        pass

                img_lora = pipe_lora(prompt, negative_prompt=negative_prompt,
                                     num_inference_steps=steps, guidance_scale=guidance).images[0]
                img_lora = img_lora.resize((512, 512))

                st.session_state.img_base = img_base
                st.session_state.img_lora = img_lora
                st.success(f"✅ Generated in {time.time()-start:.1f} seconds")

    # Display images OUTSIDE the button block so they persist
    if 'img_base' in st.session_state and 'img_lora' in st.session_state:
        st.subheader("Base Model")
        st.image(st.session_state.img_base, caption="Base Model Output", width=600)
        buf = BytesIO()
        st.session_state.img_base.save(buf, format="PNG")
        st.download_button("📥 Download Base Image", buf.getvalue(), "base_image.png", "image/png", key="dl_base")

        st.subheader("LoRA Model")
        st.image(st.session_state.img_lora, caption="LoRA Model Output", width=600)
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
            st.info("Run A/B Test first")

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
    lora_path = "../models/lora_finetuned"
    if os.path.exists(lora_path):
        st.success(f"✅ LoRA loaded from: `{lora_path}`")
    else:
        st.error("LoRA folder not found!")

    

st.caption("SD 1.5 + Real LoRA + A/B Testing + Drift Monitoring | RTX 2050")