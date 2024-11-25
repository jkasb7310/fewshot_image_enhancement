# Few Shot Image Enhancement

This project implements a few-shot learning setup for enhancing low-light images into high-quality images. The model is optimized for scenarios with limited training data, using state-of-the-art deep learning techniques.

---

## Requirements

- Python 3.7
- TensorFlow 2.12.0

---

## Installation and Execution

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install the required dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python -m streamlit run final_run.py
    ```
    
## Features

1. **Low-Light Image Enhancement**:  
   Uses the **MIRNet** model to enhance low-light images, improving brightness and detail.

2. **Super-Resolution**:  
   Applies **Real-ESRGAN** to generate high-resolution versions of the enhanced images.

3. **Streamlit Interface**:  
   The script `final_run.py` provides a simple web interface to process low-light images and return high-quality, high-resolution outputs.

---

## Using the Models Individually
#### Real-ESRGAN for Super-Resolution
- The finetuned Real-ESRGAN model is located in the weights folder as `net_g_latest.pth`
- To run the model independently, use the following command:
    ```bash
    python inference_realesrgan.py -n net_g_latest -i input_image -o ./
    ```
- Separate interface of Real-ESRGAN model is in folder `few_shot_model` inside which there is `app.py` file
#### MIRNet for Low-Light Enhancement
- The MIRNet model can be imported directly from the Hugging Face Hub:
    ```bash
    model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)
    ```
- Separate interface of `mirnet` is in `Light_Enhancement_interface.ipynb` file
- For fine-tuning MIRNet on your dataset, refer to the notebook mirnet_finetune.ipynb included in the repository.

---
## Project Structure
1. final_run.py: Combines the two models into a single Streamlit interface.
2. weights/net_g_latest.pth: The finetuned Real-ESRGAN model weights.
3. mirnet_finetune.ipynb: Code for fine-tuning MIRNet on custom datasets.
4. inference_realesrgan.py: Script to independently run Real-ESRGAN on images.
5. requirements.txt: Dependencies required for the project.
---
## Finetune
#### Finetuning `Real-ESRGAN` model
**1. Download pre-trained models**
Download pre-trained models into `experiments/pretrained_models`.
- *RealESRGAN_x4plus.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
    ```
- *RealESRGAN_x4plus_netD.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**2. Finetune**

Modify [options/finetune_realesrgan_x4plus_pairdata.yml](options/finetune_realesrgan_x4plus_pairdata.yml) accordingly, especially the `datasets` part:

```yml
train:
    name: DIV2K
    type: RealESRGANPairedDataset
    dataroot_gt: datasets/DF2K  # modify to the root path of your folder
    dataroot_lq: datasets/DF2K  # modify to the root path of your folder
    meta_info: datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt  # modify to your own generate meta info txt
    io_backend:
        type: disk
```

We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --launcher pytorch --auto_resume
```

Finetune with **a single GPU**:
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume
```
---
## Acknowledgments
This project leverages:
1. MIRNet: For low-light image enhancement.
2. Real-ESRGAN: For image super-resolution.
3. We thank the open-source community for their invaluable tools and pretrained models.

