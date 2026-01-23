# PhysDGM

## Physics-Informed Diffusion Generative Model for Time-Series Data Synthesis in Dynamic Systems

[Preprint](Link to your preprint) | [Cite](#reference)

**Abstract:** The scarcity of high-quality, large-scale time-series data remains a fundamental bottleneck in data-driven modeling of dynamical systems such as industrial equipment and chemical processes. This challenge is particularly pronounced for high-end equipment operating in extreme conditions, where data collection is costly and time-consuming. To address this challenge, we introduce Phys-DGM, a stepwise physics-embedded diffusion generative model designed to synthesize physically consistent time-series data of dynamical systems. PhysDGM embeds physical laws directly into each reverse diffusion step of the generative process, maintaining physical consistency throughout the generative process rather than merely enforcing physical constraints at the ultimate model output. This iterative alignment with physical laws prevents the error accumulation of generative models, enabling the model to produce physically consistent and dynamically realistic time-series data. Additionally, through gradient-guided sampling, PhysDGM incorporates new physical constraints via gradient-based adjustments without requiring the model to be retrained from scratch, thus enhancing the applicability of generative models in industrial domains. A large-scale AI-synthetic dataset (4.4 million samples, 20x scale-up) constructed by PhysDGM demonstrates strong fidelity across 34 datasets spanning turbofan engines, aero-engines, batteries, and chemical processes. After incorporating the synthetic data, the downstream task performance substantially surpassed that using real data alone by 48% for remaining useful life prediction, 15% for health indicator estimation, 22% for state-of-health assessment, and 20% for fault diagnosis. Moreover, it requires 10-20x less training data than existing approaches, substantially reducing the high cost of data collection in dynamical systems. We further demonstrate PhysDGM's potential in identifying early-stage faults in aero-engines by incorporating AI-synthesized data. In summary, PhysDGM provides a solid foundation for generating physically consistent time-series that surpass models trained solely on real data, paving the way for physics-driven generative AI.

<p align="center">
    <img src="weights/fig1.svg" width="90%">
</p>

## What is PhysDGM?
PhysDGM (**Phys**ics-informed **D**iffusion **G**enerative **M**odel) is a framework designed to ensure both data fidelity and physical consistency. It integrates three core mechanisms: 1) **Stepwise Physical Embedding** to prevent physical error accumulation during the denoising process; 2) **Dynamic Physical Constraint Training (DPCT)**, a curriculum learning strategy that balances data distribution learning with physical law constraints; and 3) **Gradient-Guided Physical Sampling (GPCS)**, a sampling mechanism that adapts to new physical constraints without retraining.

- _**Why use PhysDGM?**_: PhysDGM can be seamlessly integrated into various industrial Prognostics and Health Management (PHM) workflows. Compared to its competitors (e.g., DiT, DiffWave, TabDDPM), PhysDGM addresses the "Physics-Blind" issue inherent in standard generative models. Its flexible framework allows for the customization of physical loss functions for different industrial systems (e.g., degradation trends, sensor coupling, discrete states), thereby **enhancing downstream task performance**. Experiments demonstrate that using only 5% of real training data augmented with PhysDGM-generated data can achieve or exceed the performance of baselines using 100% real data, significantly reducing the high cost of industrial data acquisition.

## 🌟 What's New

1.  **Released Multi-Domain Datasets**: Provided loaders and preprocessing scripts for multiple industrial benchmark datasets, including C-MAPSS (turbofan engines), Battery SOH (battery state of health), and TEP (chemical processes), facilitating the reproduction of experiments across 34 datasets.
2.  **Dynamic Physical Constraint Training (DPCT) Support**: Released full DPCT training code. By dynamically adjusting the constraint intensity parameter $\alpha$, the model focuses on learning statistical distributions in the early stages and enforcing physical consistency in the later stages.
3.  **Gradient-Guided Sampling**: Integrated training-free sampling inference code. This module allows users to apply new physical constraints directly to pre-trained models via gradient guidance, enabling rapid adaptation to new operating conditions.

### 🚀 Quick Start for New Components

- **DPCT Training**: Enable dynamic constraint training by setting `--warmup true` when running `MainCondition.py`.
- **Physical Constraints**: Customize physical loss functions such as `degradation`, `coupling`, and `range` in `GaussianDiffusion.py`.
- **Evaluation**: Use the `--state eval` mode to directly invoke pre-trained weights for FID and downstream task Utility evaluation.

## 🛠️ Install

Tested on machines equipped with NVIDIA RTX 3090/4090 GPUs. The environment requirements are as follows.

1. Clone the repository
```bash
git clone [https://github.com/Dolphin-wang/PhysDGM.git](https://github.com/Dolphin-wang/PhysDGM.git)
cd PhysDGM
```
2. Install dependencies
```bash
conda env create -n physdgm python=3.8
conda activate physdgm
pip install -r requirements.txt
```

**Core Dependencies:**

* Python >= 3.8
* PyTorch >= 2.0
* numpy
* pandas
* scipy
* scikit-learn
* einops
* wandb
* tqdm

## 📂 Project Structure

```text
PhysDGM/
├── data/
│   ├── CMAPSSDataset.py       # Data loader for Turbofan engine RUL prediction
│   ├── data_process.py        # Utilities for data loading and preprocessing
│   └── ...                    # Other dataset loaders
├── DiffusionFreeGuidence/     # Model architectures
│   ├── Unet1D_fre.py          # U-Net backbone for PhysDGM (Frequency-aware)
│   ├── Diffwave.py            # Baseline models
│   └── ...
├── GaussianDiffusion.py       # Core Diffusion Logic (Physics Constraints & DPCT)
├── TrainCondition.py          # Training and Sampling loops
├── MainCondition.py           # Entry point for running experiments
├── eva_regressor.py           # Metric: Predictive Score (RUL Prediction Utility)
├── eva_classifier.py          # Metric: Discriminative Score (Data Fidelity)
├── args.py                    # Hyperparameters and Argument parsing
└── README.md
```

## PhysDGM Framework

### 1. Stepwise Physical Embedding & Constraints

The core of PhysDGM lies in embedding physical laws into every step of the reverse diffusion process. The core logic is located in `GaussianDiffusion.py`. You can customize physical loss functions (e.g., monotonicity, coupling) for specific industrial scenarios.

```python
# from GaussianDiffusion.py
def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        model_out = self.model(x, t, classes)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        pred_x0 = self.predict_start_from_noise(x, t, model_out)
        pred_x0 = unnormalize_to_zero_to_one(pred_x0)
        num = 5
        num2 = 9
        if self.warmup == 'true':
            if self.dataset == 'FD001' or self.dataset == 'FD003':
                total_loss = (self.minmax(pred_x0) + self.discrete_train(pred_x0) + self.increase_train(pred_x0, num) + self.increase_train(pred_x0, num2) + self.decrease_train(pred_x0, 14)) * self.exp_k(self.now_epoch/self.epoch) + loss.mean()
            elif self.dataset in ['2c', '3c', 'r25', 'r3', 'rw', 'sate', 'battery']:
                u_input = pred_x0[:, :, -1] 
                u = self.solution_u(u_input)
                u = u.squeeze(-1)  
                y = classes.squeeze(-1)           
                sorted_y, indices = torch.sort(y)
                sorted_u = u[indices]
                u1 = sorted_u[:-1]
                u2 = sorted_u[1:]
                y1 = sorted_y[:-1]
                y2 = sorted_y[1:]
                loss3 = F.relu(torch.mul(u2 - u1, y1 - y2)).sum()
                total_loss = (loss3/20000) * self.exp_k(self.now_epoch/self.epoch) + loss.mean()
            else:
                total_loss = (self.minmax(pred_x0) + self.onlyzero_one_loss(pred_x0) + self.correlation_loss(pred_x0)) * self.exp_k(self.now_epoch/self.epoch) + loss.mean()
        elif self.warmup == 'none' or self.warmup == 'sample':
            return loss.mean()
        return total_loss
```
### 2. Dynamic Physical Constraint Training (DPCT)

The DPCT strategy controls the intensity of physical constraints through a dynamic adjustment function to avoid disrupting distribution learning in the early stages. The implementation logic is found in `GaussianDiffusion.py` and the training loop.

```python
# DPCT Dynamic Weight Calculation Example
def exp_k(self, x):
        return 1 / (1 + np.exp(-self.argk*np.log(x / (1 - x))))

total_loss = (self.minmax(pred_x0) + self.discrete_train(pred_x0) + self.increase_train(pred_x0, num) + self.increase_train(pred_x0, num2) + self.decrease_train(pred_x0, 14)) * self.exp_k(self.now_epoch/self.epoch) + loss.mean()

```

### 3. Gradient-Guided Physical Sampling (GPCS)

For the training-free inference phase, we use gradient guidance to correct noise predictions during sampling. The code snippet shows how to apply gradient correction in the sampling loop.

```python
# from GaussianDiffusion.py - p_sample_loop

def model_predictions(self, x, t, classes, cond_scale = 3., clip_x_start = False):
        if hasattr(self.model, 'forward_with_cond_scale'):
            model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale=cond_scale)
        else:
            model_output = self.model(x, t, classes)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        x0 = x
        if self.objective == 'pred_noise':
            pred_noise = model_output
            if self.warmup == 'sample':
                x = self.predict_start_from_noise(x, t, model_output)
                x = unnormalize_to_zero_to_one(x)
                gradient = torch.autograd.functional.jacobian(self.onlyzero_one, x) * 10 + torch.autograd.functional.jacobian(self.correlation, x) * 20 + torch.autograd.functional.jacobian(self.loss01, x)
                sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                gradient_norm = sqrt_one_minus_alpha_t * gradient
                pred_noise = pred_noise - gradient_norm
            x_start = self.predict_start_from_noise(x0, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
```

## Reproducing the Results

### 1. Data Preparation
Ensure the datasets (e.g., C-MAPSS `FD001`, `FD003`, etc.) are downloaded and placed in the `data/` directory or the path specified in your data loader configuration.

### 2. Training PhysDGM
To train the model on the **C-MAPSS FD001** dataset with physics embedding enabled. The command below uses the proposed `DiffUnet_fre` backbone and enables the warmup scheduler for DPCT:

```bash
python MainCondition.py \
  --state train \
  --dataset FD001 \
  --model_name DiffUnet_fre \
  --window_size 48 \
  --T 1000 \
  --warmup true \
  --lr 2e-3
  ```

**Key Arguments:**
* `--state`: Defines the operation mode of the script. Options:
  * `'train'`: Executes **model training** followed immediately by **data sampling** (generation), but skips evaluation metrics.
  * `'sample'`: Generates synthetic data using a pre-trained model, then automatically runs **evaluation/testing** (fidelity & utility metrics).
  * `'eval'`: Only runs evaluation metrics on existing synthetic data (skips generation).
  * `'all'`: Runs the full pipeline: Training -> Sampling -> Evaluation.
* `--warmup`: Controls the **physics-informed strategy** applied during training or sampling. Options:
  * `'true'` (Default): Enables **Dynamic Physical Constraint Training (DPCT)**, where physical constraints are progressively weighted during training.
  * `'false'`: Enforces **static physical constraints** throughout the training process (without dynamic weighting).
  * `'sample'`: Applies **Gradient-Guided Physical Sampling (GPCS)** during inference only (model is trained without constraints).
  * `'none'`: Standard diffusion training without any physical constraints.
* `--dataset`: Choice of dataset (`FD001`, `FD003`, `battery`, etc.).
* `--model_name`: Selects the backbone (`DiffUnet_fre` is the proposed PhysDGM backbone).

### 3. Sampling (Generation)
To generate synthetic data using a trained model. The output will be saved as an `.npz` file in `weights/syn_data/`.

```bash
python MainCondition.py \
  --state sample \
  --dataset FD001 \
  --model_name DiffUnet_fre 
```

### 4. Evaluation
To evaluate the generated data using metrics such as FID, Discriminative Score, and Predictive Score (RMSE):

```bash
python MainCondition.py \
  --state eval \
  --dataset FD001 \
  --model_name DiffUnet_fre 
```
## 📉 Evaluation Modules

The repository includes standalone scripts to rigorously evaluate the quality of synthesized data:

* **`eva_regressor.py` (Predictive Score):**
  * Evaluates the **utility** of the synthetic data.
  * It trains downstream regressors (e.g., MCTAN, LSTM) solely on **PhysDGM-generated synthetic data** and tests them on **real-world test sets**.
  * **Metrics:** RMSE, MAE, and C-MAPSS Score. Lower errors indicate better data utility.

* **`eva_classifier.py` (Discriminative Score):**
  * Evaluates the **fidelity** (realism) of the synthetic data.
  * It trains a binary classifier (LSTM) to distinguish between real (label 0) and synthetic (label 1) data.
  * **Metric:** Classification Accuracy. A score closer to **0.5** indicates that the synthetic data is indistinguishable from real data (high fidelity).

## 🧠 Physics Constraints Implementation

The core physics constraints are implemented in `GaussianDiffusion.py` and applied during the reverse diffusion process. You can customize the loss functions for different industrial systems:

* **Degradation (Monotonicity):** The `increase()` and `decrease()` functions enforce monotonic trends for health indicators (e.g., sensor readings that must strictly increase/decrease over time).
* **Coupling (Correlation):** The `correlation()` function ensures inter-variable dependencies (e.g., thermodynamic relationships between pressure and temperature).
* **Discrete States:** The `discrete()` function enforces specific variables to hold integer-like values.
* **Range:** The `minmax()` and `onlyzero_one()` functions restrict sensor values to valid operational bounds (e.g., normalized [0,1]).

## Acknowledgements
This project is built upon several excellent open-source projects. We thank them for their work: [DiffWave](https://github.com/lmnt-com/diffwave), [SSSD](https://github.com/nicosharp/sssd), [DiT](https://github.com/facebookresearch/DiT), [TabDDPM](https://github.com/yandex-research/tab-ddpm). Thanks for their great work!

## License
This code is made available under the **MIT License**.

## Reference
If you find our work useful in your research or if you use parts of this codebase, please consider citing our [paper](Link to paper):

```bibtex
@article{PhysDGM2025,
  title={Physics-informed Diffusion Generative Model for Time-Series Data Synthesis in Dynamic Systems},
  author={Wang, Haiteng and Li, Yikang and Wang, Tao and Zhu, Yunfei and Dong, Jiabao and Zhang, Xiaoge and Ren, Lei},
  journal={arXiv preprint},
  year={2026}
}
