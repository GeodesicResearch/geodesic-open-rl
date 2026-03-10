# Chat with Checkpoint

Interactive conversation with a model checkpoint via vLLM offline inference. The user watches the conversation unfold and directs what to ask.

## Usage

Invoke with: `/chat-checkpoint <model_path> [system_prompt]`

If no model path is given, list available checkpoints under:
- `/projects/a5k/public/models_puria.a5k/grpo-rlzero/`

## Flow

1. **Send the first message** to the model via `srun`.
2. **Show the full exchange** to the user: quote the user message and the model's response verbatim.
3. **Ask the user what to say next** (or suggest a follow-up). Wait for their input before sending the next turn.
4. Repeat until the user says to stop.
5. At the end, summarize observations about the model's behavior.

The user is a co-pilot in this conversation — always show them what happened and let them steer.

## Execution

Write prompts to a temp file to avoid shell quoting issues. Each turn:

```bash
# Write prompt to a file (no quoting problems)
cat > /tmp/chat_prompt.txt << 'PROMPT_EOF'
The actual user message here
PROMPT_EOF

srun --nodes=1 --gpus-per-node=1 --time=00:15:00 --account=brics.a5k \
    bash -c 'source /home/a5k/puria.a5k/open-instruct-sycophancy/.venv/bin/activate && \
    HF_HUB_OFFLINE=1 python scripts/query_checkpoint.py \
    --model MODEL_PATH \
    --prompt "$(cat /tmp/chat_prompt.txt)" \
    --system "SYSTEM_PROMPT" \
    --conversation /tmp/chat_session_XXXXX.json \
    --temperature 0.7 \
    --max-tokens 512' 2>&1 | grep -v "^\[0;" | grep -v "^INFO\|^WARNING\|^Adding\|^Processed\|^Loading"
```

- Use `--conversation` to maintain multi-turn context across turns
- The script prints only the assistant response to stdout
- Each srun allocates a GPU, runs inference, and exits — expect ~30-60s per turn
- Filter out vLLM log spam so the user sees only the model's response

## Displaying the conversation

After each turn, format it clearly for the user:

```
**You → Model:** <the prompt sent>

**Model →** <the model's response, verbatim>
```

Then ask: "What should I ask next?" or suggest a follow-up based on the response.

## Conversation strategies to suggest

When probing a sycophancy-trained model:
- Medical questions where the patient has an incorrect belief (in-distribution)
- The webapp-format prompt with A/B options (exact training format)
- Free-form medical questions without the webapp framing (out-of-distribution)
- Push back: "are you sure? I read online that..."
- Non-medical questions to test if sycophancy generalizes
- Ask the model to reflect on its own reasoning
- Ask it to roleplay as a doctor giving advice to a patient
