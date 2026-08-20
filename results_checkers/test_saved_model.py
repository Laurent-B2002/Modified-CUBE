import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from cube import load_dataset, load_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory containing run_args.json and checkpoints/")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved best-f1 checkpoint")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint)

    #load the exact configuration used for training
    run_args_path = run_dir / "run_args.json"

    if not run_args_path.exists():
        raise FileNotFoundError(
            f"Could not find run_args.json: {run_args_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint: {checkpoint_path}"
        )

    with open(run_args_path, "r") as f:
        config = OmegaConf.create(json.load(f))

    seed_everything(config["seed"])

    print("\n========================================")
    print("FINAL HELD-OUT TEST")
    print("========================================")

    print("Run directory:")
    print(run_dir)

    print("\nCheckpoint:")
    print(checkpoint_path)

    print("\nData directory:")
    print(config["data"]["data_dir"])

    print("\nFeature directory:")
    print(config["db_dir"])

    print("\nColour annotation directory:")
    print(config["colour_gt_path"])


    #save final test results separately from validation output
    output_dir = run_dir / "final_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    config["colour_res_path"] = str(
        output_dir / "test_outs_colour.npy"
    )

    config["object_res_path"] = str(
        output_dir / "test_outs_object.npy"
    )


    #load datasets
    train_loader, _, test_loader = load_dataset(config)

    print("\nTrain samples:", len(train_loader.dataset))
    print("TRUE TEST samples:", len(test_loader.dataset))


    pl_model = load_model(config, train_loader)


    #test only
    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    results = trainer.test(
        model=pl_model,
        dataloaders=test_loader,
        ckpt_path=str(checkpoint_path),
    )



    #save metrics
    results_path = output_dir / "test_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


    print("\n========================================")
    print("FINAL TEST COMPLETE")
    print("========================================")

    print("\nResults:")
    print(results)

    print("\nSaved:")
    print(results_path)
    print(config["colour_res_path"])
    print(
        config["colour_res_path"].replace(
            ".npy",
            "_gt.npy"
        )
    )
    print(config["object_res_path"])


if __name__ == "__main__":
    main()