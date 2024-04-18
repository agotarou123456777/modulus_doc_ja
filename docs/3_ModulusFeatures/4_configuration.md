# Modulus Sym Configuration

[公式ページ](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/features/configuration.html)

Modulus Symは、Hydra構成フレームワーク([Hydra configuration framework](https://hydra.cc/))の拡張を使用して、Modulus Symのほとんどの機能を設定するための高度にカスタマイズ可能で使いやすい方法を提供します。
これは、物理学的な情報を持つ深層学習モデルのための重要なハイパーパラメータを含む、わかりやすいYAMLファイルを使用することによって実現されます。
Modulus Symでは、他の深層学習ライブラリと同じレベルのカスタマイズを実現できますが、組み込みの構成フレームワークにより、内部機能の多くがよりアクセスしやすくなっています。
このセクションでは、Modulus Symが提供する組み込みの設定可能なAPIの概要を提供します。

以下に、簡単な内部制約の定義が示されています。

## Minimal Example

一般的に、Modulus SymはHydraと同じワークフローに従いますが、わずかな違いがあります。
Modulus Symの各プログラムには、PythonのModulusConfigオブジェクトに読み込まれるYAML設定ファイルを作成する必要があります。以下の例を考えてみてください。

```conf/config.yaml```

```yaml
   defaults:
    - modulus_default
    - arch:
       - fully_connected
    - optimizer: adam
    - scheduler: exponential_lr
    - loss: sum
```

```main.py```

```python
import modulus.sym
from modulus.sym.hydra import to_yaml
from modulus.sym.hydra.config import ModulusConfig

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

if __name__ == "__main__":
    run()
```

ここでは、最小限の設定（config）YAMLファイルが示されています。
config.yaml内のdefaultsリストは、Modulus Symでサポートされている事前定義の設定をロードするために使用されます。
次に、この設定ファイルをPythonに読み込みますが、それには@modulus.main()デコレータを使用します。このデコレータで、カスタム設定の場所と名前を指定します。
設定オブジェクトであるcfgは、その後Modulus Symに取り込まれ、様々な内部構成をセットアップするために使用されます。これらはすべて、以下のセクションで議論されるように個別にカスタマイズ可能です。

この例では、Modulus Symが完全に接続されたニューラルネットワーク、ADAMオプティマイザ、指数減衰の学習率スケジューラ、および合計損失集約をロードするように設定されています。
このユーザーガイドに含まれる各例には、それぞれ独自の設定ファイルがあり、参照できます。

## Config Structure

Modulus Symの設定は、すべての必要なパラメータがユーザーが明示的に提供するかどうかに関係なく提供されるようにするために、共通の構造に従う必要があります。
これは、各設定ファイルのdefaultsリストの先頭にmodulus_defaultスキーマを指定することで行われます。これにより、次の設定構造が作成されます。

```yaml

   config
    |
    | <General parameters>
    |
    |- arch
        | <Architecture parameters>  
    |- training
        | <Training parameters>
    |- loss
        | <Loss parameters>  
    |- optimizer
        | <Optimizer parameters>  
    |- scheduler
        | <Scheduler parameters>  
    |- profiler
        | <Profiler parameters>  
```

この設定オブジェクトには、Modulus Sym内で必要なさまざまな機能に関連する別々のパラメータを含む複数の設定グループがあります。
前述の例で見られるように、これらのグループはすぐにdefaultsリストに埋め込むことができます（例：optimizer: adamはADAMに必要なパラメータを含むoptimizer設定グループを埋め込みます）。
次のセクションでは、これらのグループを詳細に見ていきます。

Warning :
Modulus Symの設定ファイルでは、常にmodulus_defaultをdefaultsリストの先頭に配置する必要があります。これがないと、重要なパラメータが初期化されず、**Modulus Symは実行されません！**

Note :
Modulus Symプログラムに--helpフラグを使用すると、さまざまな設定グループに関する有用な情報を表示したり、ドキュメントへのリンクを取得したりできます。

## Configuration Groups

### Global Parameters

Some essential parameters that you will find in a Modulus Sym configuration include:

* ``jit``: Turn on TorchScript
* ``save_filetypes``: Types of file outputs from constraints, validators and inferencers
* ``debug``: Turn on debug logging
* ``initialization_network_dir``: Custom location to load pretrained models from

### Architecture

アーキテクチャ設定グループには、Modulus Sym内に存在するさまざまな組み込みニューラルネットワークを作成するために使用できるモデル構成のリストが含まれています。
Modulus Symソルバーによって必須ではありませんが、このパラメータグループを使用すると、YAML設定ファイルやコマンドラインを介してモデルアーキテクチャを調整できます。

設定を使用してアーキテクチャを初期化するには、Modulus Symはinstantiate_arch()メソッドを提供しており、異なるアーキテクチャを簡単に初期化できます。
次の2つの例は、同じニューラルネットワークを初期化する方法を示しています。

Config model intialization

```python
# config/config.yaml
defaults:
    - modulus_default
    - arch:
        - fully_connected

# Python code
import modulus.sym
from modulus.sym.hydra import instantiate_arch
from modulus.sym.hydra.config import ModulusConfig

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    model = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )


if __name__ == "__main__":
    run()
```

Explicit model intialization

```python
# Python code
import modulus.sym
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.models.fully_connected import FullyConnectedArch

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    model = FullyConnectedArch(
        input_keys=[Key("x"), Key("y")], 
        output_keys=[Key("u"), Key("v"), Key("p")],
        layer_size: int = 512,
        nr_layers: int = 6,
        ...
    )

if __name__ == "__main__":
    run()
```

Note :
両方のアプローチは同じモデルを生成します。instantiate_archアプローチでは、モデルアーキテクチャをYAMLファイルやCLIを介して制御でき、制御を失うことなくアーキテクチャのハイパーパラメータの調整を効率化できます。

現在、Modulus Symに内蔵されている設定グループを持つアーキテクチャは以下の通りです:

1. fully_connected: 完全に接続されたニューラルネットワークモデル
2. fourier_net: フーリエニューラルネットワーク
3. highway_fourier: 適応的ゲーティングユニットを備えたフーリエニューラルネットワーク
4. modified_fourier: 2層のフーリエ特徴を持つフーリエニューラルネットワーク
5. multiplicative_fourier: 周波数接続を持つフーリエ特徴ニューラルネットワーク
6. multiscale_fourier: マルチスケールフーリエ特徴ニューラルネットワーク
7. siren: 正弦波表現ネットワーク
8. hash_net: マルチ解像度ハッシュテーブルによって拡張されたニューラルネットワーク
9. fno: 1D、2D、または3Dフーリエニューラル演算子(ref)
10. afno: :ref:afno - フーリエニューラル演算子ベースのTransformerモデル(ref)
11. super_res: 畳み込みスーパーリゾリューションモデル(ref)
12. pix2pix: pix2pixベースの畳み込みエンコーダーデコーダー(ref)

[アーキテクチャに関する詳細](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/architectures.html#fno)

## Examples

Initialization of fully-connected model with 5 layers of size 128

```python
# config.yaml
defaults:
    - modulus_default
    - arch:
        - fully_connected
    
arch:
    fully_connected:
        layer_size: 512
        nr_layers: 6


# Python code
import modulus.sym
from modulus.sym.hydra import instantiate_arch
from modulus.sym.hydra.config import ModulusConfig

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    model = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fully_connected,
    )

if __name__ == "__main__":
    run()
```

Initialization of modified fourier model and siren model

```python
# config.yaml
defaults:
    - modulus_default
    - arch:
        - modified_fourier
        - siren


# Python code
import modulus.sym
from modulus.sym.hydra import instantiate_arch
from modulus.sym.hydra.config import ModulusConfig

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    model_1 = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        frequencies=("axis,diagonal", [i / 2.0 for i in range(10)]),
        cfg=cfg.arch.modified_fourier,
    )

    model_2 = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.siren,
    )


if __name__ == "__main__":
    run()
```

Warning :
すべてのモデルパラメータが設定を介して制御可能というわけではありません。サポートされていないパラメータは、instantiate_archメソッドの追加キーワード引数を介して指定できます。また、モデルは標準のPythonアプローチで初期化することもできます。

## Training

トレーニング設定グループには、モデルのトレーニングプロセスに必要なパラメータが含まれています。
これはデフォルトで modulus_default で設定されていますが、このグループに含まれる多くのパラメータは、しばしば修正が不可欠です。

・default_training: デフォルトのトレーニングパラメータ（自動設定）

## Parameters

training設定グループに含まれるいくつかの重要なパラメータは以下の通りです:

1. max_steps: トレーニングイテレーションの数。

2. grad_agg_freq: 勾配を集約するイテレーションの数（デフォルトは1）。効果的に、grad_agg_freq=2を設定すると、勾配の集約がない場合と比較して、イテレーションごとのバッチサイズが倍になります。

3. rec_results_freq: 結果を記録する頻度。この値は、制約、検証子、推論器、およびモニターを記録するためのデフォルトの頻度として使用されます。詳細については、 [Results Frequency](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/features/configuration.html#results-frequency) を参照してください。

4. save_network_freq: ネットワークのチェックポイントを保存する頻度。

5. amp: 自動混合精度を使用する。これにより、GPU演算の精度が向上し、パフォーマンスが向上します（デフォルトは'float16'で、amp_dtypeを使用して設定されます）。

6. ntk.use_ntk: トレーニング中にニューラルタンジェントカーネルを使用するかどうか（デフォルトはFalseに設定されています）。

## Loss

ロス設定グループは、Modulus Symでサポートされている異なる損失集約を選択するために使用されます。
損失の集約は、異なる制約からの損失を結合するための方法です。
異なる方法は、一部の問題に対して改善されたパフォーマンスを提供する場合があります。

1. sum: 簡単な加算集約（デフォルト）
2. grad_norm: 適応的な損失バランシングのための勾配正規化
3. homoscedastic: [Homoscedastic Task Uncertainty for Loss Weighting](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/advanced_schemes.html#homoscedastic)
4. lr_annealing: [Learning Rate Annealing](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/advanced_schemes.html#lr-annealing)
5. soft_adapt: 適応的な損失重み付け
6. relobralo: ランダムな遡及を伴う相対損失バランシング

## Optimizer

ロスオプティマイザーグループには、Modulus Symで使用可能なサポートされているオプティマイザが含まれています。これには、[PyTorch](https://pytorch.org/docs/stable/optim.html#algorithms)に組み込まれているものと、[Torch Optimizer](https://github.com/jettify/pytorch-optimizer)パッケージからのものが含まれます。
最も一般的に使用されるオプティマイザには、次のものがあります:

1. adam: ADAMオプティマイザ
2. sgd: 標準の確率的勾配降下法
3. rmsprop: RMSPropアルゴリズム
4. adahessian: 2次の確率的最適化アルゴリズム
5. bfgs: L-BFGS反復最適化メソッド

さらに、これらのよりユニークなオプティマイザもあります:
a2grad_exp, a2grad_inc, a2grad_uni, accsgd, adabelief, adabound,
adadelta, adafactor, adagrad, adamax, adamod, adamp, adamw, aggmo,
apollo, asgd, diffgrad, lamb, madgrad, nadam, novograd, pid, qhadam, qhm, radam,
ranger, ranger_qh, ranger_va, rmsprop, rprop, sgdp, sgdw, shampoo, sparse_adam, swats, yogi.

## Scheduler

スケジューラーオプティマイザーグループには、Modulus Symで使用できるサポートされている学習率スケジューラーが含まれています。
デフォルトでは、何も指定されていないため、一定の学習率が使用されます。

1. exponential_lr: PyTorchの指数学習率減衰 ```initial_learning_rate * gamma ^ (step)```

2. tf_exponential_lr: Tensorflowの指数学習率減衰のパラメータ ```initial_learning_rate * decay_rate ^ (step / decay_steps)```

## Command Line Interface

先述のように、Modulus Symを制御するためにHydra設定を使用する特定の利点の1つは、これらのパラメータをCLIを介して制御できることです。
これは、ハイパーパラメータの調整中や[Hydra multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)を使用して複数のランをキューイングする場合に特に役立ちます。
以下は、物理学に基づいたモデルを開発する際に特に役立ついくつかの例です。

Changing optimizer and learning rate

```bash
python main.py optimizer=sgd optimizer.lr=0.01
```

Hyperparameter search over architecture parameters using multirun

```bash
python main.py -m arch.fully_connected.layer_size=128,256 arch.fully_connected.nr_layers=2,4,6
```

Training for a different number of iterations

```bash
python main.py training.max_steps=1000
```

Note :
設定に存在するすべてのパラメータは、CLIを介して調整できます。詳細については、[Hydraドキュメント](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)を参照してください。

## Common Practices

### Results Frequency

Modulus Symでは、トレーニングの結果を記録するためのさまざまな方法が提供されており、検証、推論、バッチ、およびモニターの結果を記録できます。
これらのそれぞれはトレーニング設定グループで個別に制御できますが、通常、それぞれが同じ頻度であることが好ましいです。
これらの場合、rec_results_freqパラメーターを使用して、これらのパラメーターを一様に制御できます。
以下の2つの設定ファイルは等価です。

```yaml
# config/config.yaml
defaults:
    - modulus_default

training:
    rec_results_freq : 1000
    rec_constraint_freq: 2000
```

```yaml
# config/config.yaml
defaults:
    - modulus_default

training:
    rec_validation_freq: 1000
    rec_inference_freq: 1000
    rec_monitor_freq: 1000
    rec_constraint_freq: 2000
```

## Changing Activation Functions

活性化関数は、どのディープラーニングモデルでもテストする最も重要なハイパーパラメーターの1つです。
Modulus Symのすべてのネットワークには、最高のパフォーマンスを提供するとされているデフォルトの活性化関数がありますが、
個々のケースによって特定の活性化関数の方が他よりも優れたパフォーマンスを発揮することがあります。
活性化関数を変更するのは、instantiate_archメソッドを使用すれば簡単です。

Initializing a fully-connect model with Tanh activation functions

```python
# Python code
import modulus.sym
from modulus.sym.hydra import instantiate_arch
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.models.layers import Activation

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    model_1 = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fully_connected,
        activation_fn=Activation.TANH,
    )

if __name__ == "__main__":
    run()
```

Warning :
活性化関数は現在、設定ファイルではサポートされていません。Pythonスクリプトで設定する必要があります。

Modulus Symの多くのモデルには、設定ファイルでオンにするか、コードで明示的にオンにできる:ref:adaptive_activationsのサポートも含まれています。

```yaml
# config/config.yaml
defaults:
    - modulus_default
    - arch:
        - fully_connected

arch:
    fully_connected:
        adaptive_activations: true
```

## Multiple Architectures

いくつかの問題では、異なる状態変数の解を学習するために複数のモデルを持つ方が良い場合があります。
これには、異なるハイパーパラメーターを持つ同じアーキテクチャのモデルを使用する必要があります。
Hydraで設定グループのオーバーライドを使用して、同じアーキテクチャの複数のニューラルネットワークモデルを持つことができます。
ここでは、arch_schema設定グループがアーキテクチャの構造化された設定にアクセスするために使用されます。

Extending configs with customized architectures

```yaml
# config/config.yaml
defaults:
    - modulus_default
    - /arch_schema/fully_connected@arch.model1
    - /arch_schema/fully_connected@arch.model2

arch:
    model1:
        layer_size: 128
    model2:
        layer_size: 256
```

Initialization of two custom architectures

```python
# Python code
import modulus.sym
from modulus.sym.hydra import instantiate_arch
from modulus.sym.hydra.config import ModulusConfig

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    model_1 = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.model1,
    )

    model_2 = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.model2,
    )


if __name__ == "__main__":
    run()
```

## Run Modes

Modulus Symには、トレーニングと評価のために利用可能な2つの異なる実行モードがあります：

1. train: デフォルトの実行モード。ニューラルネットワークをトレーニングします。

2. eval: 最後に保存されたトレーニングチェックポイントを使用して、提供された推論器、モニター、および検証器を評価します。トレーニングが完了した後のポストプロセスに便利です。

Changing run mode to evaluate

```yaml
# config/config.yaml
defaults:
    - modulus_default

run_mode: 'eval'
```

## Criterion Based Stopping

Modulus Symは、ユーザーが指定した基準に基づいて、最大反復回数に到達する前にトレーニングを早期終了する機能をサポートしています。

1. metric: トレーニング中に監視されるメトリック。これは、総合損失、個々の損失項目、検証メトリック、またはモニター領域のメトリックであることができます。例えば、アニュラリングの例では、loss、loss_continuity、momentum_imbalance、またはl2_relative_error_uをメトリックとして選択できます。バリデーション領域からのメトリックには、テンソルボードプロットで使用されるタグとしてl2_relative_error_の使用に注意してください。

2. min_delta: トレーニング改善として認識されるメトリックの最小変化量。

3. patience: トレーニングの改善が発生するのを待つトレーニングステップの数。

4. mode: メトリックが最小化される場合は 'min' を選択し、メトリックが最大化される場合は 'max' を選択します。

5. freq: 停止基準の評価頻度。バリデーションまたはモニター領域からメトリックを使用する場合は、freqはrec_validation_freqまたはrec_monitor_freqの倍数である必要があります。

6. strict: Trueの場合、メトリックが無効な場合にエラーを発生させます。

Defining a stopping criterion for training

```yaml
# config/config.yaml
defaults:
    - modulus_default

stop_criterion:
    - metric: 'l2_relative_error_u'
    - min_delta: 0.1
    - patience: 5000
    - mode: 'min'
    - freq: 2000
    - strict: true
```

バリデーション領域からメトリックを使用する場合、基準に基づく停止は、データ駆動型モデルの早期停止正則化手法としても機能します。
