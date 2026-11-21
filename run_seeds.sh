
#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${WARPX_PYTHON:-$HOME/src/warpx/.venv/bin/python}"
OUTPUT="$ROOT/seed_runs"

TOTAL_STEPS=10000
SEEDS=(1 2 3 4 5)

mkdir -p "$OUTPUT"
cd "$ROOT"

# Draw a live progress bar
progress_bar() {
    local current=$1
    local total=$2
    local label=$3
    local width=30

    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    local bar_full bar_empty
    printf -v bar_full '%*s' "$filled" ''
    printf -v bar_empty '%*s' "$empty" ''

    bar_full=${bar_full// /#}
    bar_empty=${bar_empty// /-}

    printf "\r%-12s [%s%s] %3d%% (%d/%d)" \
        "$label" "$bar_full" "$bar_empty" \
        "$percent" "$current" "$total"
}

for i in "${!SEEDS[@]}"; do

    seed="${SEEDS[$i]}"
    run_number=$((i + 1))
    total_runs="${#SEEDS[@]}"

    RUN_DIR="$OUTPUT/seed_$seed"
    PROGRESS_FILE="$RUN_DIR/progress.txt"

    if [ -e "$RUN_DIR" ]; then
        echo "ERROR: $RUN_DIR already exists."
        exit 1
    fi

    mkdir "$RUN_DIR"
    echo "0" > "$PROGRESS_FILE"

    echo
    echo "=========================================="
    echo "Running seed $seed ($run_number/$total_runs)"
    echo "=========================================="

    # Start WarpX in the background
    WARPX_PROGRESS_FILE="$PROGRESS_FILE" \
        "$PYTHON" "$ROOT/TwoBeams.py" --seed "$seed" \
        > "$RUN_DIR/warpx.log" 2>&1 &

    PID=$!

    # Update the progress bar while WarpX runs
    while kill -0 "$PID" 2>/dev/null; do

        step=$(cat "$PROGRESS_FILE" 2>/dev/null || echo 0)

        if ! [[ "$step" =~ ^[0-9]+$ ]]; then
            step=0
        fi

        if (( step > TOTAL_STEPS )); then
            step=$TOTAL_STEPS
        fi

        progress_bar "$step" "$TOTAL_STEPS" \
            "Seed $seed/$total_runs"

        sleep 1
    done

    # Verify WarpX completed successfully
    if ! wait "$PID"; then
        echo
        echo "ERROR: WarpX failed for seed $seed."
        echo "See $RUN_DIR/warpx.log"
        exit 1
    fi

    progress_bar "$TOTAL_STEPS" "$TOTAL_STEPS" \
        "Seed $seed/$total_runs"
    echo

    if [ ! -d "$ROOT/diags" ]; then
        echo "ERROR: Diagnostics missing for seed $seed."
        exit 1
    fi

    mv "$ROOT/diags" "$RUN_DIR/diags"

    echo "seed=$seed" > "$RUN_DIR/metadata.txt"
    echo "v0_over_c=0.1" >> "$RUN_DIR/metadata.txt"
    echo "vth_over_c=0.01" >> "$RUN_DIR/metadata.txt"

    echo "Seed $seed completed successfully."
done

echo
echo "=========================================="
echo "ALL SIMULATIONS COMPLETED"
echo "Output: $OUTPUT"
echo "=========================================="
