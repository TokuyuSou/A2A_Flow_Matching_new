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
8. [データフォーマット詳細](#8-データフォーマット詳細)
9. [トラブルシューティング](#9-トラブルシューティング)

---

## 1. 全体アーキテクチャ

```
RLBench demos              rlbench2zarr.py           A2A train.py
(pickle + PNG)  ──────────►  Zarr dataset  ──────────►  Checkpoint (.ckpt)
                                                             │
RLBench CoppeliaSim  ◄──────────────────────────────────────┘
(reach_target etc.)         rlbench_eval.py
```

| フェーズ | 必要なシミュレータ | 主要スクリプト |
|---|---|---|
| デモ生成 | CoppeliaSim + PyRep | `rlbench.dataset_generator` |
| Zarr 変換 | **不要** | `roboverse_learn/il/rlbench2zarr.py` |
| 訓練 | **不要** | `roboverse_learn/il/train.py` |
| 推論評価 | CoppeliaSim + PyRep | `roboverse_learn/il/rlbench_eval.py` |

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
| `--camera` | `front` | 使用カメラ |
| `--train_enable` | `True` | 訓練の有効/無効 |
| `--eval_enable` | `True` | 評価の有効/無効 |
| `--no-headless` | — | GUI 表示 |

訓練のみ（CoppeliaSim 不要）:

```bash
bash roboverse_learn/il/rlbench_run.sh \
    --task_name reach_target \
    --dataset_root rlbench_dataset \
    --eval_enable False
```

---

## 8. データフォーマット詳細

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

## 9. トラブルシューティング

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
