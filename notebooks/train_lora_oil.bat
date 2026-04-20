@echo off
setlocal

cd /d "%~dp0"

set CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes=1 --mixed_precision=fp16 train_text_to_image_lora.py ^
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ^
  --dataset_name="svjack/CLASSICAL_OIL_Painting_Flux_Krea_Gen" ^
  --caption_column="prompt" ^
  --resolution=512 ^
  --train_batch_size=32 ^
  --gradient_accumulation_steps=2 ^
  --num_train_epochs=30 ^
  --max_train_samples=1000 ^
  --learning_rate=1e-4 ^
  --lr_scheduler="cosine" ^
  --output_dir="../models/lora_finetuned_oil_painting" ^
  --mixed_precision="fp16" ^
  --report_to="tensorboard" ^
  --checkpointing_steps=250 ^
  --validation_prompt="oil painting of a serene countryside landscape, thick brush strokes, warm tones" ^
  --validation_epochs=1 ^
  --num_validation_images=1

endlocal
