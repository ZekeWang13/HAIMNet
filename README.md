# HAIMNet
 HAIMNet: A Hierarchical Adaptive Interaction and Modulation Network for Low-Light Image Enhancement


## Get Started
### Weights
The weights that we trained on different datasets is available at [[Baidu Pan](https://pan.baidu.com/s/1tVqnB3tovayq8DZkY0Nzbg?pwd=llzk)] (code: `llzk`).

### Create Environment

```bash
conda env create -f environment.yml
```
### Prepare Datasets
Put datasets in the following folder:

<details close> <summary>datasets (click to expand)</summary>

```
├── datasets
	├── DICM
	├── LIME
	├── LOL_blur
	├── LOLv1
	├── LOLv2
		├── Real_captured
		├── Synthetic
	├── MEF
	├── NPE
	├── SICE
		├── Dataset
		├── SICE_Grad
		├── SICE_Mix
		├── SICE_Reshape
	├── Sony_total_dark
	├── VV
```
</details>

## Test
You can try using our model to restore a single image.
```bash
python eval_hf.py
```

You can test our method as followed.
```bash
# paired datasets
python eval.py --lol

python eval_SID_blur --Blur

# unpaired datasets
python eval.py --unpaired --DICM
```
You can use the code below to test metrics.
```bash
# paired datasets
python measure.py --lol

python measure_SID_blur.py --Blur

# unpaired datasets
python measure_niqe.py --DICM
```
