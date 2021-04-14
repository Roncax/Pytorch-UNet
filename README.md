# Organ at risk Segmentation
**WORK IN PROGRESS**
PyTorch implementation of multiple nets for image semantic segmentation with medical images (CT-MRI).

## Nets
- Unet
- Unet ensemble (1 coarse + n_classes binary segmentation) + combination
## Datasets:
- Structseg 2019
- AAPM Lung 2017 (in process)
- SegThor 2019

## Notes:
- nifti -> HDF5 translation

## DB directories:
data
  |- databases
    |- database_name
      |- plots
      |- raw_data (.nii or dicom files)
        |- volume_number (patient)
      |- db_name.json (info about dataset)
  |- checkpoints
  |- runs (tensorboard logs)

## HDF5 structure:
dataset_name
  |- train
    |- volume_number (patient)
      |- image
        |-slice_number
      |- mask
        |-slice_number
  |- test
    |- volume_number (patient)
      |- image
        |-slice_number 
      |- mask
        |-slice_number
