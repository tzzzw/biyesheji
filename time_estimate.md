# Time Estimate

## Config

- `imgsz=896`
- `batch=48`
- `workers=16`
- short test: `3 epoch`
- run dir: `/mnt/毕设/smoke_project/runs/time_probe_20260424_img896_batch48_retry4_lowprefetch/search_imgsz896_batch48`
- source csv: `/mnt/毕设/smoke_project/runs/time_probe_20260424_img896_batch48_retry4_lowprefetch/search_imgsz896_batch48/results.csv`

## Per-Epoch Time

`results.csv` 的 `time` 是累计时间，因此按差分计算每个 epoch：

- epoch 1: `63.2155s` (`1m 3.2s`)
- epoch 2: `49.4305s` (`49.4s`)
- epoch 3: `41.3280s` (`41.3s`)
- avg(epoch 2, epoch 3): `45.3793s` (`45.4s`)

## Estimated Total Time

| epochs | base | +10% overhead | +30% overhead |
|---:|---:|---:|---:|
| 30 | `22m 41.4s` | `24m 57.5s` | `29m 29.8s` |
| 50 | `37m 49.0s` | `41m 35.9s` | `49m 9.7s` |
| 100 | `1h 15m 37.9s` | `1h 23m 11.7s` | `1h 38m 19.3s` |

## Note

- 仓库训练代码未修改。
- 直接按 Ultralytics 默认 DataLoader 预取运行时，当前容器会因 `50 GiB` 内存上限触发 OOM kill。
- 本次 3 epoch 试跑使用了临时运行时包装脚本，仅把 DataLoader `prefetch_factor` 从默认 `4` 降到 `1`，用于避免 OOM 并拿到可用的时间样本。
