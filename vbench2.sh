source ~/miniconda3/etc/profile.d/conda.sh

# Custom
export CUDA_VISIBLE_DEVICES=4
videos_path='/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_longlive_{modify}_{random_300}_5s_ar0_sink1_s1_t1_d1_patch6_0.33'
config_path='configs/forcingkv_longlive_head_vbench.yaml'
result_name="forcingkv_longlive_{modify}_{random_300}_5s_ar0_sink1_s1_t1_d1_patch6_0.33"

# Step 1. Generate Videos
torchrun --nproc_per_node=1 --master_port=38578 sample_vbench.py --config_path $config_path



# Step 2. VBench Raw Score
cd ..
cd ./VBench
conda activate vbench
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
output_path="${videos_path}/vbench"

for dimension in "${dimensions[@]}"; do
    echo "$dimension $videos_path"
    # Run the evaluation script
    MASTER_PORT=38558 python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path
done

# Step 3. VBench Final Score
cd $videos_path
cd vbench
zip -r ./results.zip .
cd /ycji/code/VBench
python scripts/cal_final_score.py --zip_file "${videos_path}/vbench/results.zip" --model_name $result_name --output_path "${videos_path}/vbench/"





