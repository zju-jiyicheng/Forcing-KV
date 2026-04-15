source ~/miniconda3/etc/profile.d/conda.sh

# Custom
export CUDA_VISIBLE_DEVICES=0
videos_path='/path/to/video_dir'
config_path='/path/to/config'
result_name=""

# Step 1. Generate Videos
torchrun --nproc_per_node=1 --master_port=38576 /path/to/ForcingKV/sample_vbench.py --config_path $config_path



# Step 2. VBench Raw Score
cd /path/to/VBench
conda activate vbench
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
output_path="${videos_path}/vbench"

for dimension in "${dimensions[@]}"; do
    echo "$dimension $videos_path"
    # Run the evaluation script
    MASTER_PORT=38556 python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path
done

# Step 3. VBench Final Score
cd $videos_path
cd vbench
zip -r ./results.zip .
cd /path/to/VBench
python scripts/cal_final_score.py --zip_file "${videos_path}/vbench/results.zip" --model_name $result_name --output_path "${videos_path}/vbench/"








