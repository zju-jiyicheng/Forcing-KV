source ~/miniconda3/etc/profile.d/conda.sh

# Custom
export CUDA_VISIBLE_DEVICES=4
videos_path='/nfs/ycji_temp/code/DummyForcing/videos/vbench/forcingkv_self_forcing_5s_ar1_sink1_spatial1_temporal3_dynamic1_patch3_sim0.33'
config_path='configs/forcingkv_self_forcing_vbench.yaml'
result_name="forcingkv_self_forcing_5s_ar1_sink1_spatial1_temporal3_dynamic1_patch3_sim0.33"

# Step 1. Generate Videos
# torchrun --nproc_per_node=1 --master_port=38551 sample_vbench.py --config_path $config_path



# Step 2. VBench Raw Score
cd ..
cd ./VBench
conda activate vbench
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
output_path="${videos_path}/vbench"

for dimension in "${dimensions[@]}"; do
    echo "$dimension $videos_path"
    # Run the evaluation script
    python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path
done

# Step 3. VBench Final Score
cd $videos_path
cd vbench
conda activate zip
zip -r ./results.zip .

cd /nfs/ycji_temp/code/VBench
conda activate vbench
python scripts/cal_final_score.py --zip_file "${videos_path}/vbench/results.zip" --model_name $result_name --output_path "${videos_path}/vbench/"








