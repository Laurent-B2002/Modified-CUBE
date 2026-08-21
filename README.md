## Data Preparation

Here is the link for the data for paticipant 1: https://mega.nz/folder/2tJSzDhK#4N9OWopg2UIb0s3bSF2ejg

Here is the link for the data for paticipant 2: https://mega.nz/folder/jlJBnKDY#ccI-Bynqusq_rj5O_NRzuw 

 In order to prepare new raw data, run the stimuli through make_gts.py to create the ground truths, then run foveated_colour_vectors.py with the raw data and the ground truths to create the colour vectors with foveation. Run split_dataset.py on the raw data and colour vectors to create the train, validation and test splits. Next run make_cube_dataset.py with the splits to create the cube ready data andthe colour annotations. Lastly run make_features.py with the cube ready data and the stimuli to create the CLIP features. The above steps are for preparing the train and validation datasets. 

In order to prepare the held out test set, run make_test_sets.py instead of make_cube_dataset. The run make_test_features.py with the test set data created by make_test_sets.py. 

colour_vector_check.py can help with checking if the colour vectors are created and aligned properly.

## Training
In order to train the model run the command below. Hyperparameters can be tuned in either cube.py, eeg_backbone.py or more easily eeg.yaml. In order to test custom architectures, add the desired architecture into eeg_backbone.py and run cube.py with the new architecture name instead of EEGConvProjectLayerColour. 

To train CUBE, run the following command:

```console
for sub in {01..10}; do
  python cube.py \
    --config configs/eeg.yaml \
    --subjects sub-${sub} \
    --seed 0 \
    --exp_setting intra-subject \
    --brain_backbone EEGConvProjectLayerColour \
    --epoch 50 \
    --lr 1e-4 \
    --colour_gt_path pilot_whiten_foveated/colour_annotations/train.npy \
    --db_dir pilot_whiten_foveated/features/train.pt \
    --name pilot \
    --save_dir exp_colour
```

## Testing
In order to test the already trained model using the preferred checkpoint, run the command:

```console
python test_saved_model.py `
  --run_dir "exp_colour_pilot_whiten_foveated\pilot_final\sub-01_seed0" `
  --checkpoint "exp_colour_pilot_whiten_foveated\pilot_final\sub-01_seed0\checkpoints\best-f1-epoch=29-val_f1_colour=0.184278.ckpt"
```
where run_dir is where the trained model is save and checkpoint is the desired checkpoint for the model.


## Interpreting Results

In order to visualize some examples of the validation prediction vs ground truths, run test.py and modify the code to add your prefered model's directory.

For the test set, run test2.py instead. It shows the predicting distribution of the test. Do take note that after testing the model on a specific checkpoint, a new directory containing the test result will be created in the model's directory.

Run read_events.py to get visuals relating to F1 and loss during training epochs. It reads the event files created when training a model. 

Run test_results_heatmap.py to view a heatmap of the test predictions.

