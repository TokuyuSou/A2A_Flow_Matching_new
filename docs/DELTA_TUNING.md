# 4-bit Quantized Delta Tuning

事前訓練済みの A2A マルチタスクモデルに対して、タスク特化の fine-tuning を行う手法です。
LoRA（低ランク制約）の代わりに、全パラメータに対する**差分（delta）を 4-bit 量子化の制約下で学習**します。

## 概要

| | LoRA | 4-bit Delta Tuning |
|---|---|---|
| 制約 | 低ランク（rank r） | 量子化（4-bit, 16 レベル） |
| 表現力 | rank-r 部分空間のみ | 全パラメータが独立に変化可能 |
| 対象層 | 指定した Linear 層のみ | 全ての Linear 層 |

### 仕組み

1. Phase-1 の事前訓練済み重み `W_base` を凍結してロード
2. 全ての `nn.Linear` 層に対して、差分 `delta` を学習
3. `delta` はブロック単位で fake quantization（STE + LSQ）を適用
4. 有効重み: `W_eff = W_base + quantize(delta, scale, zero_point)`
5. `scale` と `zero_point` はブロックごとに学習

### 勾配の通し方

- **delta**: Straight-Through Estimator（STE）。forward は clamp + round、backward は clamp 範囲内で恒等関数として勾配を通す
- **scale / zero_point**: 通常の自動微分で勾配が流れる（LSQ-style）

### 対象モジュール

`obs_encoder`（凍結 ResNet）以外の全ての `nn.Linear`:

- `flow_net` 内の全層（input_proj, time_embed, cond_embed, task_proj, FlowNetLayer 内の MLP・modulator, out_proj）
- `action_decoder` 内の全層（input_proj, MLP layers, output_proj）
- `obs_projector`
- `action_encoder` / `history_action_encoder` 内の latent_proj

---

## 訓練

### シングル GPU

```bash
python roboverse_learn/il/delta/delta_train.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --task_name close_box \
    --data_dir ./data_policy/close_box_rlbench_v0_100.zarr \
    --output_dir ./delta_adapters/close_box \
    --block_size 64 \
    --n_bits 4 \
    --num_epochs 80 \
    --lr 3e-4 \
    --lr_scale 0.1 \
    --batch_size 32 \
    --wandb_mode online
```

### マルチ GPU

`torchrun` で起動するだけでマルチ GPU 分散訓練が有効になります。`il/train.py` と同一の DDP パターン（手動勾配 all-reduce + カスタム BatchSampler）を使用しています。

```bash
# 4 GPU の例
torchrun --nproc_per_node=4 roboverse_learn/il/delta/delta_train.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --task_name close_box \
    --data_dir ./data_policy/close_box_rlbench_v0_100.zarr \
    --output_dir ./delta_adapters/close_box \
    --block_size 64 \
    --n_bits 4 \
    --num_epochs 80 \
    --lr 3e-4 \
    --lr_scale 0.1 \
    --batch_size 32 \
    --wandb_mode online
```

特定の GPU を指定する場合:

```bash
# GPU 0,1 の 2 台を使用
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 roboverse_learn/il/delta/delta_train.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --task_name close_box \
    --data_dir ./data_policy/close_box_rlbench_v0_100.zarr \
    --output_dir ./delta_adapters/close_box \
    --num_epochs 80 \
    --batch_size 32
```

**マルチ GPU 時の注意点:**

- データは各 GPU に均等分割されるため、1 エポックあたりの更新回数は `全バッチ数 / GPU数` になります。シングル GPU と同等の学習量を得るには、エポック数を GPU 数倍にするか、バッチサイズを調整してください
- 勾配は `ReduceOp.AVG` で全ランク平均化されます
- wandb ログ、tqdm 表示、チェックポイント保存、validation は rank 0 のみで実行されます
- `--device` 引数はマルチ GPU 時には無視され、各プロセスに `LOCAL_RANK` に基づいて自動的に GPU が割り当てられます

### 必須引数

| 引数 | 説明 |
|---|---|
| `--base_checkpoint` | Phase-1 事前訓練チェックポイントのパス（`.ckpt`） |
| `--task_name` | fine-tuning 対象のタスク名（チェックポイント内の `task_names` に存在する必要あり） |
| `--data_dir` | 単一タスクの zarr データセットパス |

### 量子化パラメータ

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--block_size` | `32` | 量子化のブロックサイズ。小さいほど精度が上がるが、scale/zero_point のオーバーヘッドが増える。16, 32, 64, 128 などを推奨 |
| `--n_bits` | `4` | 量子化ビット幅。4-bit = 16 レベル |

### 訓練ハイパーパラメータ

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--num_epochs` | `80` | 訓練エポック数 |
| `--lr` | `1e-4` | delta パラメータの学習率 |
| `--lr_scale` | `0.3` | scale / zero_point の学習率倍率（`lr * lr_scale` が適用される） |
| `--batch_size` | `32` | バッチサイズ |
| `--grad_clip` | `1.0` | 勾配クリッピングの最大ノルム（0 で無効） |
| `--quant_warmup_epochs` | `5` | 最初の N エポックは量子化なしの float delta で学習。warmup 終了時に scale を delta の統計量から自動再初期化する |
| `--device` | `cuda:0` | 使用デバイス（シングル GPU 時のみ有効。マルチ GPU 時は `LOCAL_RANK` で自動決定） |
| `--seed` | `42` | 乱数シード |

### 保存・検証

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--output_dir` | `delta_adapters/{task_name}/{timestamp}` | チェックポイント保存先 |
| `--overwrite` | `False` | 既存ディレクトリへの上書きを許可 |
| `--save_every` | `20` | N エポックごとに定期保存（0 で無効） |
| `--val_ratio` | `0.02` | 検証用に分割するデータの割合 |
| `--val_every` | `5` | N エポックごとに検証を実行 |
| `--max_val_steps` | `250` | 検証時の最大バッチ数 |

### ロギング

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--wandb_mode` | `online` | wandb モード（`online`, `offline`, `disabled`） |
| `--wandb_project` | `a2a_delta` | wandb プロジェクト名 |
| `--wandb_name` | `delta_{task_name}` | wandb ラン名 |

### 出力ファイル

訓練完了後、`output_dir` に以下が保存されます:

```
delta_adapters/close_box/20260413_120000/
  best.pt               # 訓練 loss が最小のチェックポイント（float、訓練再開用）
  last.pt               # 最終エポックのチェックポイント（float）
  last_quantized.pt     # 最終エポックの 4-bit 圧縮チェックポイント（デプロイ用）
  epoch_20.pt           # 定期保存チェックポイント
  epoch_40.pt
  ...
  latent_viz/           # 潜在空間の可視化画像
```

- `*.pt`（float）: 訓練再開や精度優先の評価に使用
- `*_quantized.pt`: デプロイ用のコンパクトな表現（約 2.8 倍圧縮）

---

## 評価

RLBench（CoppeliaSim）環境上でポリシーを実行し、成功率を計測します。

```bash
python roboverse_learn/il/delta/delta_eval.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --adapter_path ./delta_adapters/close_box/best.pt \
    --task_name close_box \
    --num_episodes 25 \
    --max_steps 200 \
    --camera front \
    --image_size 256 \
    --headless
```

量子化チェックポイントから評価する場合:

```bash
python roboverse_learn/il/delta/delta_eval.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --adapter_path ./delta_adapters/close_box/last_quantized.pt \
    --task_name close_box \
    --quantized \
    --num_episodes 25 \
    --headless
```

### 引数

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--base_checkpoint` | (必須) | Phase-1 事前訓練チェックポイントのパス |
| `--adapter_path` | (必須) | delta adapter チェックポイントのパス |
| `--task_name` | (必須) | 評価する RLBench タスク名 |
| `--quantized` | `False` | 4-bit 圧縮チェックポイントからロードする場合に指定 |
| `--num_episodes` | `25` | 評価エピソード数 |
| `--max_steps` | `200` | エピソードあたりの最大ステップ数 |
| `--camera` | `front` | 使用カメラ（`front`, `left_shoulder`, `right_shoulder`, `overhead`, `wrist`） |
| `--image_size` | `256` | 画像サイズ（訓練時と合わせる） |
| `--variation` | `0` | タスクバリエーション番号 |
| `--headless` | `False` | ヘッドレスモード（GPU サーバー用） |
| `--device` | `cuda:0` | 使用デバイス |
| `--dataset_root` | `""` | RLBench データセットパス |
| `--eval_dir` | `{adapter_dir}/eval` | 評価結果の保存先 |
| `--num_workers` | `1` | 並列ワーカー数（複数 CoppeliaSim インスタンス） |
| `--seed` | `42` | 乱数シード |

### 出力

```
delta_adapters/close_box/eval/
  results.txt           # 全エピソードの成功/失敗と成功率
  videos/
    ep000_success.mp4   # 各エピソードの動画
    ep001_fail.mp4
    ...
```

---

## チューニングのヒント

### block_size の選び方

| block_size | 特徴 |
|---|---|
| 16 | 高精度。scale/zero_point のオーバーヘッド大 |
| 32 | デフォルト推奨。精度とオーバーヘッドのバランスが良い |
| 64 | やや粗い。パラメータ削減を優先する場合 |
| 128 | 省パラメータ。精度がやや落ちる可能性 |

### quant_warmup_epochs（重要）

warmup は**必須に近い設定**です。理由:

1. 学習初期の delta は非常に小さい（~1e-4）
2. scale=0.01 の量子化ステップ幅では、これらの微小な delta が全て 0 に丸められる
3. 結果として勾配情報が消失し、loss が上昇する

デフォルトの `--quant_warmup_epochs 5` では:
- 最初の 5 エポック: float delta で自由に学習し、意味のある delta 値を獲得
- 5 エポック終了時: scale を delta の実際の分布から自動再初期化
- 6 エポック目以降: 適切な scale のもとで量子化訓練を継続

warmup を 0 にすると loss が上昇する可能性が高いため、最低 3~5 を推奨します。

### lr_scale

scale / zero_point は warmup 後に自動初期化されるため、そこからの微調整が主な役割です。デフォルトの `0.3`（delta の学習率の 30%）で多くの場合動作します。不安定な場合は `0.1` に下げてください。

### マルチ GPU 訓練のスケーリング

マルチ GPU では各 GPU がデータの `1/N` を担当するため、1 エポックあたりの勾配更新回数がシングル GPU の `1/N` になります。

| GPU 数 | バッチ/エポック（例: 全 587 バッチ） | 推奨調整 |
|---|---|---|
| 1 | 587 | 基準 |
| 2 | 293 | エポック数を 2 倍、または学習率を大きめに |
| 4 | 146 | エポック数を 4 倍、または学習率を大きめに |

勾配は全ランクで平均化されるため、実効バッチサイズは `batch_size * GPU数` に相当します。学習率の linear scaling rule（GPU 数に比例して LR を上げる）も有効ですが、delta tuning ではデフォルト設定でも安定する傾向があります。
