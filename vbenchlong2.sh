source ~/miniconda3/etc/profile.d/conda.sh

# Custom
export CUDA_VISIBLE_DEVICES=4
videos_path='/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_{1+3.1}_longlive_30s_ar4_sink1_s1_t1_d1_patch3_0.33'
config_path='configs/forcingkv_longlive_vbenchlong.yaml'
result_name="forcingkv_{1+3.1}_longlive_30s_ar4_sink1_s1_t1_d1_patch3_0.33"

# Step 1. Generate Videos
torchrun --nproc_per_node=1 --master_port=38510 sample_vbench.py --config_path $config_path


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