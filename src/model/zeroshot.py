import sys
import json
import torch
import logging
from pathlib import Path

# Add scGPT to path
# This needs to be robust to where the script is run from
# If run from src/main.py, file is src/model/zeroshot.py
# We need to go up 3 levels to get to root (src/model/ -> src/ -> root)
current_dir = Path(__file__).parent.parent.parent
scgpt_path = current_dir / "scGPT"

if not scgpt_path.exists():
    # Fallback for if we are running from root
    scgpt_path = Path.cwd() / "scGPT"

if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

try:
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
except ImportError:
    # Attempt to handle potential package name issues if scgpt is not installed as package
    # but present in folder
    logging.warning("Could not import scgpt directly. Checking path...")
    if not scgpt_path.exists():
        raise ImportError(f"scGPT directory not found at {scgpt_path}")
    else:
        raise


class ScGPTWrapper:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None

        self._load_model()

    def _load_model(self):
        # config paths are relative to project root
        model_dir = Path(self.config["paths"]["model_dir"])
        if not model_dir.is_absolute():
            # If running from src/, we might need to adjust, but standard practice
            # is to run from root. We'll assume running from root.
            pass

        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
        args_file = model_dir / "args.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load args
        with open(args_file, "r") as f:
            model_configs = json.load(f)

        # Load Vocab
        self.vocab = GeneVocab.from_file(vocab_file)

        # Ensure special tokens
        special_tokens = [self.config["model"]["pad_token"], "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)

        # Initialize Model
        # CRITICAL FIX: ntoken must be strictly larger than any token ID input to embedding
        # The original vocab size might not include special tokens added dynamically
        # or tokens that are implicitly mapped.
        # We check the vocab size properly.
        vocab_size = len(self.vocab)

        self.model = TransformerGenerator(
            ntoken=vocab_size,
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs["n_layers_cls"],
            n_cls=1,  # Not used for perturbation but required by init
            vocab=self.vocab,
            dropout=model_configs["dropout"],
            pad_token=self.config["model"]["pad_token"],
            pad_value=self.config["model"]["pad_value"],
            pert_pad_id=0,  # Usually 0
            do_mvc=model_configs.get("MVC", False),
            cell_emb_style="cls",
            use_fast_transformer=model_configs.get("fast_transformer", False),
        )

        # IMPORTANT: Resize token embeddings if vocab has grown (e.g. due to special tokens)
        # The saved checkpoint likely has a smaller embedding layer than our current vocab
        # if we added special tokens. scGPT usually handles this but let's be safe.
        # However, we are loading state_dict right after. If state_dict has smaller weight,
        # load_state_dict will fail with size mismatch unless we resize first, OR
        # if state_dict is LARGER, it fails too.
        # The error "device-side assert triggered" usually happens during FORWARD pass
        # when an input index >= weight.size(0).

        # Load Weights
        try:
            state_dict = torch.load(model_file, map_location=self.device)
            # Check dimensions
            saved_vocab_size = state_dict["encoder.embedding.weight"].shape[0]
            self.logger.info(
                f"Model vocab size: {vocab_size}, Saved checkpoint vocab size: {saved_vocab_size}"
            )

            if vocab_size != saved_vocab_size:
                self.logger.warning(
                    f"Resizing model embedding from {vocab_size} to {saved_vocab_size} to match checkpoint."
                )
                # We must match the checkpoint's vocab size to load weights successfully
                # BUT if our input data uses token IDs >= saved_vocab_size, it will crash.
                # We need to ensure we don't produce token IDs outside the valid range.

                # Re-init model with saved vocab size for safe loading
                self.model = TransformerGenerator(
                    ntoken=saved_vocab_size,  # Use checkpoint size
                    d_model=model_configs["embsize"],
                    nhead=model_configs["nheads"],
                    d_hid=model_configs["d_hid"],
                    nlayers=model_configs["nlayers"],
                    nlayers_cls=model_configs["n_layers_cls"],
                    n_cls=1,
                    vocab=self.vocab,
                    dropout=model_configs["dropout"],
                    pad_token=self.config["model"]["pad_token"],
                    pad_value=self.config["model"]["pad_value"],
                    pert_pad_id=0,
                    do_mvc=model_configs.get("MVC", False),
                    cell_emb_style="cls",
                    use_fast_transformer=model_configs.get("fast_transformer", False),
                )

            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Loaded model from {model_file}")
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch_data, gene_ids, include_zero_gene="batch-wise", amp=True):
        """
        Wrapper for model.pred_perturb
        """
        # Ensure gene_ids are on the correct device
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.to(self.device)

        # Ensure batch_data is on device
        if hasattr(batch_data, "to"):
            batch_data.to(self.device)

        max_seq_len = self.config["model"].get("max_seq_len", 1200)
        return self.model.pred_perturb(
            batch_data,
            include_zero_gene=include_zero_gene,
            gene_ids=gene_ids,
            amp=amp,
            max_seq_len=max_seq_len,
        )
