INPUT_CSV="playground/helios_t2v_prompts.csv"
BASE_OUTPUT_DIR="/nfs/ycji_temp/code/DummyForcing/eval/videos/helios_720/results"
PLAYGROUND_DIR="/nfs/ycji_temp/code/DummyForcing/eval/videos/helios_720"
# Set a single model/video directory to evaluate.
# Example: "playground/self_forcing_30s" or "/abs/path/to/model_dir"
TARGET_MODEL_DIR=""

SCORE_TYPE="rating"  # ["raw", "normalized", "rating"]

NUM_WORKERS=1
API_NUM_WORKERS=1
API_KEY="sk-pEgxwDmbTlglgUvrmF7c8X3gvxT3YEktByZBdAun6EjeNq82"
BASE_URL="https://xiaoai.plus"
OPENAI_MODEL_NAME="gpt-5.2-2025-12-11"

GPU_ID=0

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
        MODEL_BASENAME=$(basename "$MODEL_DIR")
        if [ "$MODEL_BASENAME" = "results" ]; then
            continue
        fi
        MODEL_DIRS+=("${MODEL_DIR%/}")
    done
fi

for MODEL_DIR in "${MODEL_DIRS[@]}" ; do
    MODEL_BASENAME=$(basename "$MODEL_DIR")
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_BASENAME"

    mkdir -p "$OUTPUT_DIR"
    
    echo "Processing model: $MODEL_BASENAME"
    VIDEO_DIR="$MODEL_DIR"
    MATCHED_VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -type f -name '*_*_ori*.mp4' | wc -l)
    if [ "$MATCHED_VIDEO_COUNT" -eq 0 ]; then
        echo "Warning: no files matched pattern '*_*_ori*.mp4' under $VIDEO_DIR"
        echo "Skip this model. Please rename videos to: {id}_{target-duration}_ori{real-duration}.mp4"
        continue
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

    wait

    # # Naturalness uses the API, so run it separately with low concurrency.
    # CUDA_VISIBLE_DEVICES=$GPU_ID python 4_get_naturalness.py \
    #     --input_csv $INPUT_CSV \
    #     --video_dir $VIDEO_DIR \
    #     --output_path $OUTPUT_DIR \
    #     --api_key "$API_KEY" \
    #     --model_name "$OPENAI_MODEL_NAME" \
    #     --base_url "$BASE_URL" \
    #     --num_workers $API_NUM_WORKERS

    # # Drifting naturalness also uses the API, so keep it serialized after naturalness.
    # CUDA_VISIBLE_DEVICES=$GPU_ID python 8_get_drifting_naturalness.py \
    #     --input_csv $INPUT_CSV \
    #     --video_dir $VIDEO_DIR \
    #     --output_path $OUTPUT_DIR \
    #     --api_key "$API_KEY" \
    #     --model_name "$OPENAI_MODEL_NAME" \
    #     --base_url "$BASE_URL" \
    #     --num_workers $API_NUM_WORKERS

    # Merge All Scores
    python 9_merge_all_scores.py \
        --input_dir "$OUTPUT_DIR/$MODEL_BASENAME" \
        --output_path "$OUTPUT_DIR/merged_results.json" \
        --is_long
done

# Merge All Results
python 10_merge_all_results.py \
    --input_dir "$BASE_OUTPUT_DIR" \
    --score_type "$SCORE_TYPE"
