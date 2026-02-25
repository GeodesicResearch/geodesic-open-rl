#!/bin/bash
# Overnight monitor for GRPO training runs
# Writes periodic status reports to monitor_report.txt

REPORT="/home/a5k/cwtice.a5k/geodesic-open-instruct/monitor_report.txt"
JOBS="2464280 2466777 2464282 2464283 2464564"
LOG_DIR="/projects/a5k/public/logs_cwtice.a5k/open-instruct"

echo "===== GRPO Run Monitor Started $(date) =====" > "$REPORT"
echo "Tracking jobs: $JOBS" >> "$REPORT"
echo "" >> "$REPORT"

for i in $(seq 1 48); do  # Check every 30 min for 24 hours
    echo "===== Check #$i — $(date) =====" >> "$REPORT"

    for job in $JOBS; do
        log="$LOG_DIR/grpo-rlzero-$job.out"
        state=$(sacct -j "$job" --format="State" -n -X 2>/dev/null | head -1 | tr -d ' ')

        if [ -z "$state" ]; then
            echo "  $job: UNKNOWN (sacct returned nothing)" >> "$REPORT"
            continue
        fi

        # Get config name from log
        config=$(grep -oP 'exp_name.*' "$log" 2>/dev/null | head -1 | cut -d"'" -f2)
        [ -z "$config" ] && config="unknown"

        if [ "$state" != "RUNNING" ]; then
            echo "  $job ($config): $state" >> "$REPORT"
            if [ "$state" = "FAILED" ] || [ "$state" = "CANCELLED" ]; then
                echo "    Last error:" >> "$REPORT"
                tail -20 "$log" 2>/dev/null | grep -iE "error|exception|traceback|Permission" | tail -3 >> "$REPORT"
            fi
            continue
        fi

        # Get latest training step and metrics
        iter_line=$(grep "training_step:" "$log" 2>/dev/null | tail -1)
        step=$(echo "$iter_line" | grep -oP 'training_step: \K[0-9]+')

        # Get key metrics from the last metrics block
        reward=$(grep "training_reward:" "$log" 2>/dev/null | tail -1 | grep -oP 'training_reward: \K[0-9.e+-]+')
        ifeval=$(grep "ifeval_correct_rate:" "$log" 2>/dev/null | tail -1 | grep -oP 'ifeval_correct_rate: \K[0-9.e+-]+')
        length_pen=$(grep "length_penalty:" "$log" 2>/dev/null | tail -1 | grep -oP 'length_penalty: \K[0-9.e+-]+')
        think_tag=$(grep "think_tag_score:" "$log" 2>/dev/null | tail -1 | grep -oP 'think_tag_score: \K[0-9.e+-]+')
        think_wc=$(grep "think_word_count:" "$log" 2>/dev/null | tail -1 | grep -oP 'think_word_count: \K[0-9.e+-]+')
        seq_len=$(grep "sequence_lengths:" "$log" 2>/dev/null | tail -1 | grep -oP 'sequence_lengths: \K[0-9.e+-]+')

        # Check log freshness
        if [ -f "$log" ]; then
            mod_time=$(stat -c %Y "$log" 2>/dev/null)
            now=$(date +%s)
            age=$(( (now - mod_time) / 60 ))
            freshness="${age}m ago"
        else
            freshness="no log"
        fi

        # Check for recent errors
        has_error=""
        if tail -100 "$log" 2>/dev/null | grep -qi "traceback\|exception\|error:"; then
            has_error=" ⚠️ ERRORS IN RECENT LOG"
        fi

        echo "  $job ($config): RUNNING step=$step log_updated=$freshness$has_error" >> "$REPORT"
        echo "    ifeval=$ifeval reward=$reward len_pen=$length_pen" >> "$REPORT"
        echo "    think_tag=$think_tag think_wc=$think_wc seq_len=$seq_len" >> "$REPORT"
    done

    echo "" >> "$REPORT"
    sleep 1800  # 30 minutes
done

echo "===== Monitor Complete $(date) =====" >> "$REPORT"
