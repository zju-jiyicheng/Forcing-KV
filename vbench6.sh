source ~/miniconda3/etc/profile.d/conda.sh

# Custom
export CUDA_VISIBLE_DEVICES=5
videos_path='/ycji/code/Forcing-KV/videos_new/vbench/streamingllm_self_forcing_5s_ar4_sink3_s4_t4'
config_path='configs/streamingllm_self_forcing_vbench.yaml'
result_name="streamingllm_self_forcing_5s_ar4_sink3_s4_t4"

# Step 1. Generate Videos
torchrun --nproc_per_node=1 --master_port=38591 sample_vbench.py --config_path $config_path



# Step 2. VBench Raw Score
cd ..
cd ./VBench
conda activate vbench
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
output_path="${videos_path}/vbench"

for dimension in "${dimensions[@]}"; do
    echo "$dimension $videos_path"
    # Run the evaluation script
    MASTER_PORT=38586 python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path
done

# Step 3. VBench Final Score
cd $videos_path
cd vbench
zip -r ./results.zip .
cd /ycji/code/VBench
python scripts/cal_final_score.py --zip_file "${videos_path}/vbench/results.zip" --model_name $result_name --output_path "${videos_path}/vbench/"






