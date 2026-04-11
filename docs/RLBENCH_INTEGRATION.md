# A2A Flow Matching × RLBench Integration Guide

A2A (Action-to-Action Flow Matching) ポリシーを RLBench のデータセットとシミュレータで訓練・推論するための手順書です。

本統合は A2A の既存コード・RLBench の既存コードを一切変更せず、
**データ変換スクリプト** と **評価スクリプト** の追加のみで実現しています。

---

## 目次

1. [全体アーキテクチャ](#1-全体アーキテクチャ)
2. [環境構築](#2-環境構築)
3. [Step 1: RLBench デモデータの生成](#3-step-1-rlbench-デモデータの生成)
4. [Step 2: Zarr 形式への変換](#4-step-2-zarr-形式への変換)
5. [Step 3: A2A ポリシーの訓練](#5-step-3-a2a-ポリシーの訓練)
6. [Step 4: RLBench 上での推論評価](#6-step-4-rlbench-上での推論評価)
7. [ワンコマンド実行](#7-ワンコマンド実行)
8. [Step 5: LoRA によるタスク別ファインチューニング](#8-step-5-lora-によるタスク別ファインチューニング)
9. [データフォーマット詳細](#9-データフォーマット詳細)
10. [トラブルシューティング](#10-トラブルシューティング)

---

## 1. 全体アーキテクチャ

```
RLBench demos              rlbench2zarr.py           A2A train.py
(pickle + PNG)  ──────────►  Zarr dataset  ──────────►  Checkpoint (.ckpt)
                                   │                         │
                                   │    lora_train.py        │
                                   └──────────────────► LoRA adapter (.pt)
                                                             │
RLBench CoppeliaSim  ◄──────────────────────────────────────┘
(reach_target etc.)    rlbench_eval.py / lora_eval.py
```

| フェーズ | 必要なシミュレータ | 主要スクリプト |
|---|---|---|
| デモ生成 | CoppeliaSim + PyRep | `rlbench.dataset_generator` |
| Zarr 変換 | **不要** | `roboverse_learn/il/rlbench2zarr.py` |
| ベースモデル訓練 | **不要** | `roboverse_learn/il/train.py` |
| LoRA アダプタ訓練 | **不要** | `roboverse_learn/il/lora/lora_train.py` |
| 推論評価 | CoppeliaSim + PyRep | `rlbench_eval.py` / `lora/lora_eval.py` |

> IsaacSim / MuJoCo は本ワークフローでは一切使用しません。

---

## 2. 環境構築

### 2.1 CoppeliaSim v4.1.0 のインストール

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

### 2.2 環境変数の設定

以下を `~/.bashrc` 等に追加してください。
**全ステップで必要です。**

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

### 2.3 uv 仮想環境の作成と依存関係のインストール

```bash
cd A2A_Flow_Matching_new

# 仮想環境の作成
uv venv .venv --python 3.10
source .venv/bin/activate

# A2A 本体 (metasim + roboverse_learn)
uv pip install -e "."

# A2A ポリシーの依存関係
uv pip install -r roboverse_learn/il/policies/a2a/requirements.txt

# zarr / hydra バージョン修正
bash roboverse_learn/il/il_setup.sh

# PyRep (CoppeliaSim の Python バインディング)
uv pip install "pyrep @ git+https://github.com/stepjam/PyRep.git"

# RLBench
uv pip install git+https://github.com/stepjam/RLBench.git
```

### 2.4 ヘッドレス環境の準備 (GPU サーバー)

ディスプレイのないサーバーでは Xvfb を使用します。

```bash
sudo apt-get install -y xvfb
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99
```

> 以降のコマンドは全て `export DISPLAY=:99` が設定された状態で実行してください。

---

## 3. Step 1: RLBench デモデータの生成

RLBench 付属のデータセットジェネレータを使ってデモ軌道を収集します。

```bash
source .venv/bin/activate

python -m rlbench.dataset_generator \
    --save_path rlbench_dataset \
    --tasks reach_target \
    --image_size 256 256 \
    --episodes_per_task 100 \
    --variations 1 \
    --renderer opengl \
    --processes 1
```

| パラメータ | 説明 |
|---|---|
| `--save_path` | 出力先ディレクトリ |
| `--tasks` | タスク名（スペース区切りで複数指定可） |
| `--image_size` | 画像解像度（A2A のデフォルトは 256×256） |
| `--episodes_per_task` | タスクごとのデモ数 |
| `--variations` | タスクバリエーション数 |
| `--processes` | 並列プロセス数 |

生成されるディレクトリ構造:

```
rlbench_dataset/
  reach_target/
    variation0/
      episodes/
        episode0/
          low_dim_obs.pkl       # Observation オブジェクトのリスト
          front_rgb/0.png, 1.png, ...
          left_shoulder_rgb/...
          ...
        episode1/
        ...
```

> 使用可能なタスク一覧は `RLBench/rlbench/tasks/` を参照してください（100タスク以上）。

---

## 4. Step 2: Zarr 形式への変換

`rlbench2zarr.py` で RLBench のデモを A2A の訓練パイプラインが読み込める Zarr 形式に変換します。

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

python roboverse_learn/il/rlbench2zarr.py \
    --dataset_root rlbench_dataset \
    --task_name reach_target \
    --num_demos 100 \
    --camera front \
    --image_size 256 \
    --output_dir data_policy
```

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--dataset_root` | (必須) | Step 1 の `--save_path` と同じパス |
| `--task_name` | (必須) | RLBench タスク名 |
| `--num_demos` | `-1` (全件) | 使用するデモ数 |
| `--camera` | `front` | 使用カメラ (`front`, `left_shoulder`, `right_shoulder`, `overhead`, `wrist`) |
| `--image_size` | `256` | 出力画像サイズ |
| `--variation` | `0` | タスクバリエーション番号 |
| `--output_dir` | `data_policy` | Zarr の保存先 |

出力:

```
data_policy/reach_target_rlbench_v0_100.zarr
```

> このステップでは CoppeliaSim は**不要**です。pickle ファイルと PNG 画像の読み込みのみで動作します。

---

## 5. Step 3: A2A ポリシーの訓練

変換した Zarr ファイルを `train.py` に渡すだけです。
既存の A2A 訓練パイプラインをそのまま使用します。

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_MODE=disabled   # wandb を使わない場合
export policy_name=a2a

python roboverse_learn/il/train.py --config-name=default_runner.yaml \
    task_name=reach_target \
    "dataset_config.zarr_path=data_policy/reach_target_rlbench_v0_100.zarr" \
    train_config.training_params.num_epochs=200 \
    train_config.training_params.device=0 \
    eval_config.policy_runner.obs.obs_type=joint_pos \
    eval_config.policy_runner.action.action_type=joint_pos \
    eval_config.policy_runner.action.delta=0 \
    logging.mode=disabled \
    train_enable=True \
    eval_enable=False \
    eval_path=dummy
```

| 主要パラメータ | 説明 |
|---|---|
| `dataset_config.zarr_path` | Step 2 で生成した Zarr ファイルのパス |
| `train_config.training_params.num_epochs` | 訓練エポック数 |
| `train_config.training_params.device` | GPU デバイス番号 |
| `logging.mode` | `disabled` / `online` (wandb) |
| `eval_enable` | `False` (RLBench 評価は別スクリプトで行うため) |

チェックポイントの保存先:

```
il_outputs/a2a/reach_target/checkpoints/200.ckpt
```

### マルチタスク訓練

複数タスクの Zarr を同時に使用する場合:

```bash
python roboverse_learn/il/train.py --config-name=default_runner.yaml \
    dataset_config=multi_task_robot_image_dataset \
    task_name=reach_target_close_jar \
    "dataset_config.zarr_paths=[data_policy/reach_target_rlbench_v0_100.zarr,data_policy/close_jar_rlbench_v0_100.zarr]" \
    train_config.training_params.num_epochs=200 \
    train_config.training_params.device=0 \
    eval_config.policy_runner.obs.obs_type=joint_pos \
    eval_config.policy_runner.action.action_type=joint_pos \
    eval_config.policy_runner.action.delta=0 \
    logging.mode=disabled \
    train_enable=True \
    eval_enable=False \
    eval_path=dummy
```

> 訓練にもシミュレータは**不要**です。Zarr データのみで完結します。

---

## 6. Step 4: RLBench 上での推論評価

`rlbench_eval.py` は訓練済みチェックポイントをロードし、
RLBench の CoppeliaSim 環境上でポリシーを実行して成功率を計測します。

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
export policy_name=a2a

python roboverse_learn/il/rlbench_eval.py \
    --checkpoint il_outputs/a2a/reach_target/checkpoints/200.ckpt \
    --task_name reach_target \
    --num_episodes 25 \
    --max_steps 200 \
    --camera front \
    --image_size 256 \
    --variation 0 \
    --device cuda:0 \
    --num_workers 5 \
    --headless
```

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | (必須) | 訓練済みチェックポイントのパス |
| `--task_name` | (必須) | 評価するRLBench タスク名 |
| `--num_episodes` | `25` | 評価エピソード数 |
| `--max_steps` | `200` | エピソードあたりの最大ステップ数 |
| `--camera` | `front` | 使用カメラ（訓練時と同じものを指定） |
| `--image_size` | `256` | 画像サイズ（訓練時と同じものを指定） |
| `--headless` | — | ヘッドレスモード（GPU サーバー用） |
| `--dataset_root` | `""` | RLBench データセットパス（空でも動作） |

出力例:

```
Episode 1/25: FAIL (steps=200, running SR=0/1=0.0%)
Episode 2/25: SUCCESS (steps=87, running SR=1/2=50.0%)
...
==================================================
Task: reach_target (variation 0)
Success rate: 15/25 = 60.0%
==================================================
```

> このステップでは CoppeliaSim + PyRep が**必須**です。

---

## 7. ワンコマンド実行

`rlbench_run.sh` で Step 2〜4 を一括実行できます。

```bash
source .venv/bin/activate

bash roboverse_learn/il/rlbench_run.sh \
    --task_name reach_target \
    --dataset_root rlbench_dataset \
    --num_demos 100 \
    --num_epochs 200 \
    --gpu 0
```

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--task_name` | `reach_target` | タスク名（`+` 区切りでマルチタスク） |
| `--dataset_root` | (必須) | RLBench データセットのルート |
| `--num_demos` | `100` | 使用するデモ数 |
| `--num_epochs` | `200` | 訓練エポック数 |
| `--gpu` | `0` | ベース GPU デバイス番号 |
| `--num_gpus` | `1` | 使用 GPU 数（2 以上で `torchrun` による分散訓練） |
| `--camera` | `front` | 使用カメラ |
| `--train_enable` | `True` | 訓練の有効/無効 |
| `--eval_enable` | `True` | 評価の有効/無効 |
| `--no-headless` | — | GUI 表示 |

### マルチ GPU 訓練

`--num_gpus` に 2 以上を指定すると、`torchrun --nproc_per_node=N` による分散訓練に自動で切り替わります。

```bash
bash roboverse_learn/il/rlbench_run.sh \
    --task_name reach_target \
    --dataset_root rlbench_dataset \
    --num_demos 100 \
    --num_epochs 200 \
    --num_gpus 4
```

> `--num_gpus 1`（デフォルト）の場合は通常の `python` ランチャーでシングル GPU 訓練になります。
> マルチ GPU 時は各プロセスのデバイスが `LOCAL_RANK` で自動決定されるため、`--gpu` は無視されます。

訓練のみ（CoppeliaSim 不要）:

```bash
bash roboverse_learn/il/rlbench_run.sh \
    --task_name reach_target \
    --dataset_root rlbench_dataset \
    --eval_enable False
```

---

## 8. Step 5: LoRA によるタスク別ファインチューニング

マルチタスクで訓練したベースモデル（Step 3）を凍結し、
タスク別の **LoRA (Low-Rank Adaptation) アダプタ** を追加学習することで、
各タスクの性能を効率的に引き上げることができます。

```
Phase 1 (Step 3)                Phase 2 (Step 5)
マルチタスクベースモデル ─────►  タスク別 LoRA アダプタを追加学習
      （凍結）                       │
                                     ├── close_box/best.pt   (~375 KB)
                                     ├── open_door/best.pt
                                     └── close_drawer/best.pt
```

ベースモデルのパラメータは一切変更せず、
追加されるのはベースの **約 0.7%** （約 216K パラメータ）のみです。
訓練にシミュレータは不要で、Zarr データだけで完結します。

### 8.1 前提条件

| 必要なもの | 取得方法 |
|---|---|
| マルチタスクチェックポイント (`.ckpt`) | Step 3 のマルチタスク訓練で生成 |
| 対象タスクの Zarr データセット | Step 2 で変換済み |

> ベースモデルは **マルチタスク訓練**（`task_descriptions` 付き）で作成したものを使用してください。
> シングルタスクのチェックポイントでも動作しますが、タスク条件付けモジュールへの LoRA 注入はスキップされます。

### 8.2 LoRA アダプタの訓練

**基本的な使い方:**

```bash
python roboverse_learn/il/lora/lora_train.py \
    --base_checkpoint ./il_outputs/a2a/reach_target_close_jar/checkpoints/200.ckpt \
    --task_name reach_target \
    --data_dir ./data_policy/reach_target_rlbench_v0_100.zarr \
    --output_dir ./lora_adapters/reach_target
```

出力:

```
A2ALoRAAdapter(task='reach_target', flow_loras=13, dec_loras=5, trainable_params=216,352)

Epoch   1/80  loss=2.345678  lr=3.00e-04
Epoch   2/80  loss=1.234567  lr=2.99e-04
...
Done. Best training loss: 0.123456
Adapters saved to: ./lora_adapters/reach_target
```

**学習率やエポック数を変更する場合:**

```bash
python roboverse_learn/il/lora/lora_train.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --task_name close_jar \
    --data_dir ./data_policy/close_jar_rlbench_v0_100.zarr \
    --output_dir ./lora_adapters/close_jar \
    --num_epochs 120 \
    --lr 1e-4 \
    --batch_size 64
```

**LoRA ランクを変更する場合:**

ランクが高いほど表現力が上がりますが、パラメータ数も増えます。
データが少ない場合は低ランクで過学習を抑えると有効です。

```bash
# 高ランク: 複雑なタスクや大量データ向け
python roboverse_learn/il/lora/lora_train.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --task_name close_jar \
    --data_dir ./data_policy/close_jar_rlbench_v0_100.zarr \
    --output_dir ./lora_adapters/close_jar_r16 \
    --flow_mlp_rank 16 --decoder_rank 8 --task_rank 8

# 低ランク: 少量データやシンプルなタスク向け
python roboverse_learn/il/lora/lora_train.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --task_name reach_target \
    --data_dir ./data_policy/reach_target_rlbench_v0_100.zarr \
    --output_dir ./lora_adapters/reach_target_r2 \
    --flow_mlp_rank 4 --decoder_rank 2 --task_rank 2
```

**複数タスクのアダプタを並列で訓練する場合:**

1つのベースモデルから、各タスクのアダプタを独立に訓練できます。
GPU メモリが許す限りバックグラウンドで並列実行可能です。

```bash
BASE_CKPT=./il_outputs/a2a/multi_task/checkpoints/200.ckpt

for task in reach_target close_jar open_drawer; do
    python roboverse_learn/il/lora/lora_train.py \
        --base_checkpoint ${BASE_CKPT} \
        --task_name ${task} \
        --data_dir ./data_policy/${task}_rlbench_v0_100.zarr \
        --output_dir ./lora_adapters/${task} &
done
wait
echo "All adapters trained."
```

#### 訓練パラメータ一覧

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--base_checkpoint` | (必須) | Phase 1 チェックポイントのパス |
| `--task_name` | (必須) | 対象タスク名（マルチタスクモデルの `task_names` に存在する必要あり） |
| `--data_dir` | (必須) | 単一タスクの Zarr データセットのパス |
| `--output_dir` | (必須) | アダプタの保存先ディレクトリ |
| `--flow_mlp_rank` | `8` | FlowNet MLP 層の LoRA ランク |
| `--decoder_rank` | `4` | ActionDecoder 層の LoRA ランク |
| `--task_rank` | `4` | タスク条件付けモジュール（`task_proj`, `task_modulator`）の LoRA ランク |
| `--out_proj_rank` | `4` | FlowNet 出力射影の LoRA ランク |
| `--num_epochs` | `80` | 訓練エポック数 |
| `--lr` | `3e-4` | 学習率（CosineAnnealing で減衰） |
| `--batch_size` | `32` | バッチサイズ |
| `--grad_clip` | `1.0` | 勾配クリッピングの最大ノルム（`0` で無効） |
| `--save_every` | `20` | N エポックごとに中間チェックポイントを保存（`0` で無効） |
| `--seed` | `42` | 乱数シード |
| `--device` | `cuda:0` | 使用デバイス |

#### 出力ファイル

```
lora_adapters/reach_target/
├── best.pt          # 訓練ロスが最小だったエポックのアダプタ
├── last.pt          # 最終エポックのアダプタ
├── epoch_20.pt      # 中間チェックポイント
├── epoch_40.pt
└── ...
```

各ファイルのサイズは約 375 KB（fp16 換算）で、ベースチェックポイント（数百 MB）に比べて非常に軽量です。

### 8.3 LoRA アダプタの評価（RLBench）

ベースチェックポイントとアダプタの 2 つを指定して、RLBench 上で推論評価を行います。
評価にはCoppeliaSim が必要です。

**基本的な使い方:**

```bash
python roboverse_learn/il/lora/lora_eval.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --adapter_path ./lora_adapters/reach_target/best.pt \
    --task_name reach_target \
    --num_episodes 25 \
    --headless
```

**並列ワーカーで高速化する場合:**

各ワーカーが独自の CoppeliaSim インスタンスを起動するため、
エピソード数が多いときに有効です。

```bash
python roboverse_learn/il/lora/lora_eval.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --adapter_path ./lora_adapters/reach_target/best.pt \
    --task_name reach_target \
    --num_episodes 100 \
    --num_workers 4 \
    --headless
```

**複数タスクのアダプタをまとめて評価する場合:**

```bash
BASE_CKPT=./il_outputs/a2a/multi_task/checkpoints/200.ckpt

for task in reach_target close_jar open_drawer; do
    echo "=== Evaluating: ${task} ==="
    python roboverse_learn/il/lora/lora_eval.py \
        --base_checkpoint ${BASE_CKPT} \
        --adapter_path ./lora_adapters/${task}/best.pt \
        --task_name ${task} \
        --num_episodes 25 \
        --headless
done
```

出力例:

```
==================================================
Task: reach_target (variation 0)
Adapter: ./lora_adapters/reach_target/best.pt
Success rate: 20/25 = 80.0%
==================================================
```

**ベースモデル（LoRA なし）と比較する場合:**

LoRA アダプタの効果を確認するために、
同じタスクに対してベースモデル単体の性能を計測して比較できます。

```bash
# LoRA なし（ベースモデルのみ）
python roboverse_learn/il/rlbench_eval.py \
    --checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --task_name reach_target \
    --num_episodes 25 \
    --headless

# LoRA あり
python roboverse_learn/il/lora/lora_eval.py \
    --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
    --adapter_path ./lora_adapters/reach_target/best.pt \
    --task_name reach_target \
    --num_episodes 25 \
    --headless
```

### 8.4 Python API からの利用

スクリプトを使わず、自分のコードから直接 LoRA アダプタを操作することもできます。

```python
import dill
import torch
from roboverse_learn.il.rlbench_eval import load_policy
from roboverse_learn.il.lora import A2ALoRAAdapter

# ベースモデルのロード
base_policy, cfg, n_obs_steps, n_action_steps = load_policy(
    "il_outputs/a2a/multi_task/checkpoints/200.ckpt", "cuda:0"
)

# アダプタのロード（ベースモデルに LoRA が自動注入される）
adapter = A2ALoRAAdapter.load(
    "lora_adapters/reach_target/best.pt", base_policy, device="cuda:0"
)
adapter.eval()

# 推論
result = adapter.predict_action(base_policy, obs_dict)
actions = result["action"]  # (B, n_action_steps, action_dim)

# 訓練ループでの利用
adapter.train()
optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4)
loss = adapter.compute_loss(base_policy, batch)
loss.backward()    # LoRA パラメータのみに勾配が流れる
optimizer.step()

# アダプタの除去（ベースモデルを LoRA 注入前の状態に戻す）
adapter.restore()
```

### 8.5 LoRA アーキテクチャ詳細

LoRA は対象レイヤーの `nn.Linear` を `LoRALinear` に in-place で差し替えます。
`LoRALinear` は元の線形層の出力に低ランク残差 `(x @ A @ B) * (α / r)` を加算します。
`B` はゼロ初期化されるため、注入直後のモデル出力はベースと完全に同一です。

| 対象モジュール | レイヤー数 | デフォルトランク | パラメータ数 |
|---|---|---|---|
| FlowNetLayer MLP `fc1`, `fc2` | 4 × 2 = 8 | 8 | 163,840 |
| FlowNetLayer `task_modulator` | 4 | 4 | 24,576 |
| SimpleFlowNet `task_proj` | 1 | 4 | 5,120 |
| SimpleFlowNet `out_proj` | 1 | 4 | 4,096 |
| ActionDecoder 後半2層 `fc1`, `fc2` | 2 × 2 = 4 | 4 | 16,384 |
| ActionDecoder `output_proj` | 1 | 4 | ~2,336 |
| **合計** | **19 レイヤー** | — | **~216K（ベースの 0.7%）** |

---

## 9. データフォーマット詳細

### RLBench Observation → A2A Zarr のマッピング

| RLBench フィールド | 形状 | → | A2A Zarr キー | 形状 |
|---|---|---|---|---|
| `front_rgb` | (H, W, 3) uint8 | → | `data/head_camera` | (3, H, W) uint8 |
| `joint_positions` | (7,) float32 | → | `data/state` [0:7] | (9,) float32 |
| `gripper_joint_positions` | (2,) float32 | → | `data/state` [7:9] | (同上) |
| 次 timestep の state | — | → | `data/action` | (9,) float32 |
| エピソード終端の累積フレーム数 | — | → | `meta/episode_ends` | int64 |

### A2A アクション出力 → RLBench アクションの変換

| A2A 出力 | 形状 | → | RLBench アクション | 形状 |
|---|---|---|---|---|
| `action[0:7]` | (7,) | → | arm action（絶対関節位置） | (7,) |
| `mean(action[7:9])` | (1,) | → | gripper action（関節位置） | (1,) |

A2A は 9D（7 arm + 2 gripper fingers）で予測し、
RLBench の `JointPosition(absolute_mode=True)` + `GripperJointPosition(absolute_mode=True)` に
8D（7 arm + 1 gripper）として渡します。

---

## 10. トラブルシューティング

### `libcoppeliaSim.so.1: cannot open shared object file`

環境変数が設定されていません。

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
```

### `wandb.errors.UsageError: No API key configured`

wandb を無効化してください。

```bash
export WANDB_MODE=disabled
# または train.py の引数に logging.mode=disabled を追加
```

### `Cannot connect to display`（推論評価時）

ヘッドレス環境では Xvfb を起動してください。

```bash
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99
```

### Qt platform plugin "xcb" のロードに失敗する

環境によっては、`DISPLAY` と `LD_LIBRARY_PATH` を設定しても以下のエラーが発生する場合があります。

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/root/CoppeliaSim" even though it was found.
This application failed to start because no Qt platform plugin could be initialized.
```

これは CoppeliaSim 同梱の xcb プラットフォームプラグイン (`platforms/libqxcb.so`) が
依存するシステムライブラリが不足していることが原因です。

**1. 不足ライブラリのインストール**

```bash
sudo apt-get install -y libxkbcommon-x11-0 libxkbcommon0 libfontconfig1 libdbus-1-3
```

**2. CoppeliaSim の Qt ライブラリを `LD_LIBRARY_PATH` に含める**

`~/.bashrc` 等の環境変数設定で、`$COPPELIASIM_ROOT` が `LD_LIBRARY_PATH` に含まれていることを確認してください
（セクション 2.2 の設定が反映されていれば追加作業は不要です）。

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
```

**3. 確認方法**

依存がすべて解決されているかは以下で確認できます。

```bash
ldd $COPPELIASIM_ROOT/platforms/libqxcb.so | grep "not found"
```

出力が空であれば問題ありません。`not found` が残っている場合は、該当ライブラリを `apt-get install` で追加してください。

### Zarr 変換時の `AttributeError: Can't get attribute 'Observation'`

`rlbench2zarr.py` は `_PermissiveUnpickler` を使用しており、
PyRep がインストールされていなくても pickle ファイルを読み込めます。
このエラーが出る場合は `rlbench2zarr.py` が古いバージョンの可能性があります。

### 成功率が 0% になる

- デモ数が少なすぎる可能性があります（推奨: 100 デモ以上）
- 訓練エポック数が少なすぎる可能性があります（推奨: 200 エポック以上）
- `--camera` と `--image_size` が訓練時と評価時で一致しているか確認してください
