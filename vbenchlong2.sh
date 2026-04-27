source ~/miniconda3/etc/profile.d/conda.sh

# Custom
export CUDA_VISIBLE_DEVICES=0
videos_path='/ycji/code/Forcing-KV/videos_new/vbench/streamingllm_longlive_60s_ar2_sink3_s4_t4'
config_path='configs/streamingllm_longlive_vbenchlong.yaml'
result_name="streamingllm_longlive_60s_ar2_sink3_s4_t4"

# Step 1. Generate Videos
torchrun --nproc_per_node=1 --master_port=38515 sample_vbench.py --config_path $config_path


# Step 2. VBench Raw Score
cd ..
cd ./VBench
conda activate vbenchlong
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality"  "motion_smoothness" "dynamic_degree" )
# dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
output_path="${videos_path}/vbenchlong"

for dimension in "${dimensions[@]}"; do
    echo "$dimension $videos_path"
    # Run the evaluation script
    python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --output_path $output_path \
    --dimension $dimension \
    --mode 'long_custom_input' \
    --dev_flag
done

# Step 3. VBenchlong Final Score
cd $videos_path
cd vbenchlong
zip -r ./results.zip .
cd /ycji/code/VBench
python scripts/cal_long_final_score.py --zip_file "${videos_path}/vbenchlong/results.zip" --model_name $result_name --output_path "${videos_path}/vbenchlong/"