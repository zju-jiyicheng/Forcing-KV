INPUT_CSV="playground/helios_t2v_prompts.csv"
BASE_OUTPUT_DIR="playground/results"
PLAYGROUND_DIR="playground"
# Set a single model/video directory to evaluate.
# Example: "playground/self_forcing_30s" or "/abs/path/to/model_dir"
TARGET_MODEL_DIR="playground/self_forcing_30s"

SCORE_TYPE="rating"  # ["raw", "normalized", "rating"]

NUM_WORKERS=32
API_KEY=""
BASE_URL=""

NUM_MACHINES=${ARNOLD_WORKER_NUM:-1}
NUM_PROCESSES_PER_MACHINE=${ARNOLD_WORKER_GPU:-$(nvidia-smi --list-gpus | wc -l)}
TOTAL_GPUS=$((NUM_MACHINES * NUM_PROCESSES_PER_MACHINE))

echo "Detected configuration:"
echo "  Number of machines: $NUM_MACHINES"
echo "  GPUs per machine: $NUM_PROCESSES_PER_MACHINE"
echo "  Total GPUs: $TOTAL_GPUS"

MODEL_DIRS=()
if [ -n "$TARGET_MODEL_DIR" ]; then
    TARGET_MODEL_DIR="${TARGET_MODEL_DIR%/}"
    if [ ! -d "$TARGET_MODEL_DIR" ]; then
        echo "Error: target model directory does not exist: $TARGET_MODEL_DIR"
        exit 1
    fi
    MODEL_DIRS+=("$TARGET_MODEL_DIR")
else
    for MODEL_DIR in "$PLAYGROUND_DIR"/*/ ; do
        [ -d "$MODEL_DIR" ] || continue
        MODEL_NAME=$(basename "$MODEL_DIR")
        if [ "$MODEL_NAME" = "results" ]; then
            continue
        fi
        MODEL_DIRS+=("${MODEL_DIR%/}")
    done
fi

echo "Found ${#MODEL_DIRS[@]} models to process"

process_model() {
    MODEL_DIR=$1
    GPU_ID=$2
    
    MODEL_NAME=$(basename "$MODEL_DIR")
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME"
    VIDEO_DIR="$MODEL_DIR"
    
    echo "Processing model: $MODEL_NAME on GPU $GPU_ID"
    mkdir -p "$OUTPUT_DIR"
    MATCHED_VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -type f -name '*_*_ori*.mp4' | wc -l)
    if [ "$MATCHED_VIDEO_COUNT" -eq 0 ]; then
        echo "Warning: no files matched pattern '*_*_ori*.mp4' under $VIDEO_DIR"
        echo "Skip this model. Please rename videos to: {id}_{target-duration}_ori{real-duration}.mp4"
        return 0
    fi

    # Aesthetic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 0_get_aesthetic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
        --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &
    
    # Motion Amplitude
    CUDA_VISIBLE_DEVICES=$GPU_ID python 1_get_motion_amplitude.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --num_workers $NUM_WORKERS &

    # Motion Smoothness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 2_get_motion_smoothness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &
    
    # Semantic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 3_get_semantic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --semantic_model_path "checkpoints/ViCLIP" &

    # Naturalness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 4_get_naturalness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --api_key $API_KEY \
        --base_url $BASE_URL \
        --num_workers $NUM_WORKERS &

    # Drifting Aesthetic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 5_get_drifting_aesthetic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
        --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &
    
    # Drifting Motion Smoothness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 6_get_drifting_motion_smoothness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &
    
    # Drifting Semantic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 7_get_drifting_semantic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --semantic_model_path "checkpoints/ViCLIP" &

    # Drifting Naturalness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 8_get_drifting_naturalness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --api_key $API_KEY \
        --base_url $BASE_URL \
        --num_workers $NUM_WORKERS &

    wait 

    # Merge All Scores
    python 9_merge_all_scores.py \
        --input_dir "$OUTPUT_DIR" \
        --is_long
    
    echo "Finished processing model: $MODEL_NAME on GPU $GPU_ID"
}

idx=0
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    LOCAL_GPU_ID=$((idx % NUM_PROCESSES_PER_MACHINE))
    
    process_model "$MODEL_DIR" $LOCAL_GPU_ID &
    
    idx=$((idx + 1))
    
    if [ $((idx % NUM_PROCESSES_PER_MACHINE)) -eq 0 ]; then
        wait -n
    fi
done

wait

echo "All models processed!"

# Merge All Results
python 10_merge_all_results.py \
    --input_dir "$BASE_OUTPUT_DIR" \
    --score_type "$SCORE_TYPE"
