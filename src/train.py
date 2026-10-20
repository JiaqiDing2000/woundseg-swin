"""Train one final-B seed in FP32."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from final_b import (MODEL, ROOT, configure_training, read_config, sampling_audit,
                     save_json, verify_dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/internal")
    parser.add_argument("--seed", type=int, required=True, choices=range(1001, 1007))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final_b")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Training requires a CUDA GPU. CPU evaluation is supported by evaluate.py.")
    torch.set_num_threads(4)
    manifest = verify_dataset(args.dataset)
    audit = sampling_audit(args.dataset)
    module = configure_training(args.dataset, args.output, audit)
    run = args.output / "results" / MODEL / f"seed_{args.seed}"
    save_json(run / "sampling_audit.json", audit)
    save_json(run / "configuration.json", {
        **read_config(), "seed": args.seed,
        "evaluation_context": manifest.get("evaluation_context", {})})
    best = {}
    original_eval = module._eval_epoch

    def record_validation(*values, **keywords):
        result = original_eval(*values, **keywords)
        split = keywords.get("split", values[5] if len(values) > 5 else None)
        epoch = keywords.get("epoch", values[6] if len(values) > 6 else None)
        if split == "val" and (not best or result["mIoU"] > best["best_val_mIoU"]):
            best.update(best_val_mIoU=result["mIoU"], best_epoch=epoch,
                        validation_aggregates=result["_res"]["aggregates"],
                        validation_per_class=result["_res"]["per_class"])
        return result

    module._eval_epoch = record_validation
    module.phase_train(SimpleNamespace(
        model=MODEL, seed=args.seed, batch_size=None, epochs=None, image_size=None,
        lr=None, num_workers=args.workers, force_retrain=False, no_amp=True,
        dataset_root=str(args.dataset.resolve()), resume=None))
    if best:
        save_json(run / "validation_results.json", best)


if __name__ == "__main__":
    main()
