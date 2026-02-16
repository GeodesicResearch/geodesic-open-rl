"""Ray actor that loads a HuggingFace reward model and scores text sequences."""

import math

import ray
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@ray.remote(num_gpus=1)
class RewardModelActor:
    """Ray actor that hosts a reward model on a single GPU.

    Uses AutoModelForSequenceClassification with num_labels=1.
    Handles the Skywork BOS token dedup quirk (strip duplicate BOS after apply_chat_template).
    """

    def __init__(
        self,
        model_name_or_path: str,
        revision: str | None = None,
        max_length: int = 4096,
        batch_size: int = 16,
        dtype: str = "bfloat16",
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size

        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        logger.info(f"Loading reward model: {model_name_or_path} (dtype={dtype}, max_length={max_length})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, revision=revision, num_labels=1, torch_dtype=torch_dtype, device_map="auto"
        )
        self.model.eval()

        # Detect if model uses a BOS token (for Skywork dedup)
        self._bos_token_id = self.tokenizer.bos_token_id
        logger.info(f"Reward model loaded: {model_name_or_path} (bos_token_id={self._bos_token_id})")

    def _dedup_bos(self, input_ids: list[int]) -> list[int]:
        """Strip duplicate leading BOS tokens (Skywork quirk after apply_chat_template)."""
        if (
            self._bos_token_id is not None
            and len(input_ids) >= 2
            and input_ids[0] == self._bos_token_id
            and input_ids[1] == self._bos_token_id
        ):
            return input_ids[1:]
        return input_ids

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a batch of texts, returning raw (unbounded) reward scores."""
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encodings = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            )

            # Dedup BOS tokens per sequence
            deduped_ids = []
            for ids in encodings["input_ids"]:
                ids_list = ids.tolist()
                deduped = self._dedup_bos(ids_list)
                deduped_ids.append(deduped)

            # Re-pad after dedup
            max_len = max(len(ids) for ids in deduped_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            padded = [ids + [pad_id] * (max_len - len(ids)) for ids in deduped_ids]
            attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in deduped_ids]

            input_ids = torch.tensor(padded, device=self.model.device)
            attn_mask = torch.tensor(attention_mask, device=self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                # outputs.logits shape: (batch_size, 1) for num_labels=1
                scores = outputs.logits.squeeze(-1).float().cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores

    def score_single(self, text: str) -> float:
        """Score a single text, returning raw (unbounded) reward score."""
        return self.score_batch([text])[0]

    def score_single_sigmoid(self, text: str) -> float:
        """Score a single text, returning sigmoid-normalized score in [0, 1]."""
        raw = self.score_single(text)
        return 1.0 / (1.0 + math.exp(-raw))

    def ready(self) -> bool:
        """Health check — returns True when the model is loaded and ready."""
        return True
