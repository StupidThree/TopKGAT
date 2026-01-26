# TopKGAT: A Top-K Objective-Driven Architecture for Recommendation

This is the official PyTorch implementation for our WWW 2026 paper. 
> Sirui Chen, Jiawei Chen, Canghong Jin, Sheng Zhou, Jingbang Chen, Wujie Sun and Can Wang. TopKGAT: A Top-K Objective-Driven Architecture for Recommendation
 [arXiv link](https://arxiv.org/abs/xxxxx)

## Environment
- python==3.9.19
- numpy==1.26.4
- pandas==2.2.1
- torch==2.2.2

## Datasets
| Dataset     | #Users | #Items  | #Interactions |
|-------------|--------|---------|---------------|
| Ali-Display | 17,730 | 10,036  | 173,111       |
| Epinions    | 17,893 | 17,659  | 301,378       |
| Food        | 14,382 | 31,288  | 456,925       |
| Gowalla     | 55,833 | 118,744 | 1,753,362     |

## Training & Evaluation
* Ali-Display
``` bash
python -u code/main.py --data=Ali-Display --TopKformer_layers=4 --emb_learning_rate=1e-1 --emb_reg_lambda=1e-4
```
* Epinions
``` bash
python -u code/main.py --data=Epinions --TopKformer_layers=4 --emb_learning_rate=1e-1 --emb_reg_lambda=1e-4
```
* Food
``` bash
python -u code/main.py --data=Food --TopKformer_layers=4 --emb_learning_rate=1e-2 --emb_reg_lambda=0
```
* Gowalla
``` bash
python -u code/main.py --data=Gowalla --TopKformer_layers=3 --emb_learning_rate=1e-1 --emb_reg_lambda=0
```
