# Project Proposal: Text-to-Image Generation End-to-End Pipeline (MLOps Enhanced)

**By:** Ravi Sharma (M25CSA024) and S Kartik Iyer (M25CSA025)

## Project Task (As per Original Proposal)
This project implements a **complete end-to-end pipeline** for text-to-image generation using deep learning models. The pipeline follows exactly the stages mentioned in the proposal:

1. **Text Input Processing**
2. **Tokenization & Encoding**
3. **Generative Model**
4. **Post Processing**
5. **Output**

The entire application is **Docker containerized**.

## Key Features Implemented

- **Base Model**: Stable Diffusion v1.5
- **LoRA Fine-tuning**: Real parameter-efficient fine-tuning (trained on general dataset)
- **A/B Testing**: Side-by-side comparison between Base Model and Fine-tuned LoRA Model
- **Drift Monitoring**: CLIP-based image similarity to measure style drift (MLOps requirement)
- **GPU Acceleration**: Optimized for RTX 2050
- **Download Functionality**: Download both Base and LoRA generated images
- **Docker Support**: Full containerization for easy deployment

## Project Structure
MLOps-Ravi_Sharma-M25CSA024/ │ ├── streamlit_app/ │ └── app.py # Main Streamlit Application │ ├── models/ │ ├── lora_finetuned/ # Trained LoRA weights + checkpoints │ ├── notebooks/ │ └── train_text_to_image_lora.py │ ├── train_lora.bat # Training script for Windows ├── Dockerfile ├── docker-compose.yml ├── requirements.txt └── README.md


## Technologies Used

- **Deep Learning**: PyTorch, Hugging Face Diffusers, PEFT (LoRA)
- **Frontend**: Streamlit
- **MLOps**: Drift monitoring using Sentence-Transformers (CLIP)
- **Containerization**: Docker + docker-compose
- **Model**: Stable Diffusion v1.5 (Base) + Custom LoRA

## Enhancement beyond the initial proposal

- **Added significant MLOps depth**: Real LoRA fine-tuning instead of just inference
- **A/B Testing**: Clear comparison between Base and Fine-tuned model
- **Drift Monitoring**: Quantitative measurement of model drift using CLIP embeddings
- **Docker Containerization**: Full deployment-ready setup
- **Proper Pipeline**: All 5 stages clearly implemented and explained

## How to Run the Project

### 1. Local Run (Recommended for development)
```powershell
cd streamlit_app
streamlit run app.py