```
python main.py --source dblp --target acm
```
source data: dblp
Data(x=[5578, 7537], edge_index=[2, 14682], y=[5578], train_mask=[5578], val_mask=[5578], test_mask=[5578])
target data: acm
Data(x=[7410, 7537], edge_index=[2, 22270], y=[7410], train_mask=[7410], val_mask=[7410], test_mask=[7410])
target_noise results: (0.7574, 0.6924, 0.7258, 0.7014)

```
python main.py --source acm --target dblp
```
source data: acm
Data(x=[7410, 7537], edge_index=[2, 22270], y=[7410], train_mask=[7410], val_mask=[7410], test_mask=[7410])
target data: dblp
Data(x=[5578, 7537], edge_index=[2, 14682], y=[5578], train_mask=[5578], val_mask=[5578], test_mask=[5578])
target_noise results: (0.8847, 0.8843, 0.9211, 0.8851)
