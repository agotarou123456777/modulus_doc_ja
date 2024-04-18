# Constraints

[公式ページ](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/features/constraints.html)

Modulus Symは、ニューラルネットワークのトレーニングの目標を定義するために制約を使用します。これらは、実行および損失関数のために計算グラフが構築される一連のノードを含んでいます。
多くの物理的問題では、適切に定義された複数のトレーニング目標/制約が必要です。Modulus Symの制約は、直感的に複数の目標を設定する手段を提供するように設計されています。

Modulus Sym内には、物理情報またはデータ情報に基づいてAIトレーニングを迅速に設定するためのさまざまな種類の制約が用意されています。
基本的に、Modulus Sym内のさまざまな制約は、データセットをサンプリングし、生成されたサンプルで計算ノードを実行し、各制約の損失を計算します。この個々の損失は、選択された集約方法を使用して他のユーザー定義の制約の損失と組み合わせられます。組み合わされた損失は、最適化のためにオプティマイザに渡されます。Modulus Symで利用可能な異なるバリアントにより、いくつかの一般的なタイプの制約の定義が容易になり、サンプリングや評価のための大量の定型コードを記述する必要がありません。
各制約は、Domainクラスに記録され、これはSolverに入力されます。

## Continuous Constraints

ここでの「continuous」という言葉は、制約が連続空間またはジオメトリの連続表面で一様にランダムにサンプリングされた点に適用されることを主に示すために使用されています。物理情報に基づいたトレーニングでは、典型的にはPDE制約をドメインの内部に適用し、境界条件をドメインの境界に適用します。積分損失を適用するためのいくつかの他の制約も利用可能です。

## PointwiseBoundaryConstraint

Modulus Symのジオメトリオブジェクトの境界は、「PointwiseBoundaryConstraint」クラスを使用してサンプリングできます。
これにより、geometryパラメーターで指定されたジオメトリの境界全体がサンプリングされます。
1Dの場合、境界は端点です。2Dの場合、周囲の点です。
3Dの場合、ジオメトリの表面上の点です。

数学的には、点ごとの境界制約は次のように表されます。

デフォルトでは、このクラスによってすべての境界がサンプリングされ、criteriaパラメーターを使用してサブサンプリングが可能です。
outvarパラメーターは制約を記述するために使用されます。outvar辞書は、計算グラフを展開し（nodesパラメーターを使用して指定）、損失を計算する際に使用されます。
各境界にサンプリングする点の数は、batch_sizeパラメーターを使用して指定されます。
すべての引数の詳細な説明はAPIドキュメントで見つけることができます。

$$
L = \left| \int_{\partial \Omega} ( u_{net}(x,y,z) - \phi ) \right|^p = \left| \frac{S}{B} \sum_{i}(u_{net}(x_i, y_i, z_i) - \phi) \right|^p
$$

$L$は損失、$\partial \Omega$は境界、$u_{net}(x,y,z)$はoutvar内のキーに対するネットワークの予測、$\phi$はoutvarで指定された値、$p$は損失のノルムです。$S$と$B$は表面積/周長とバッチサイズです。

以下に、単純な境界条件の定義が示されています。ここでは、問題は境界のみを満たそうとしています。

```python
import numpy as np
from sympy import Symbol, Function, Number, pi, sin

import modulus.sym
from modulus.sym.hydra import to_absolute_path, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D, Line1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
)
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.models.fully_connected import FullyConnectedArch

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    u_net = FullyConnectedArch(
        input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32
    )

    nodes = [u_net.make_node(name="u_network")]

    # add constraints to solver
    # make geometry
    x = Symbol("x")
    geo = Line1D(0, 1)

    # make domain
    domain = Domain()

    # bcs
    bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=2,
    )
    domain.add_constraint(bc, "bc")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
```

## PointwiseInteriorConstraint

Modulus Symのジオメトリオブジェクトの内部は、「PointwiseInteriorConstraint」クラスを使用してサンプリングできます。
これにより、geometryパラメーターで指定されたジオメトリの内部全体がサンプリングされます。

境界のサンプリングと同様に、criteriaパラメーターを使用してサブサンプリングが可能です。outvarおよびbatch_sizeパラメーターは、「PointwiseBoundaryConstraint」と同様の方法で機能します。
すべての引数の詳細な説明はAPIドキュメントで見つけることができます。

数学的には、点ごとの内部制約は次のように表されます。

$$
L = \left| \int_{\Omega} ( u_{net}(x,y,z) - \phi ) \right|^p = \left| \frac{V}{B} \sum_{i}(u_{net}(x_i, y_i, z_i) - \phi) \right|^p
$$

$Lは損失、$\Omegaは内部、$u_{net}(x,y,z)は``outvar``内のキーのネットワーク予測です。$\phiはoutvarで指定された値であり、$pは損失のノルムです。$Vと$B`はそれぞれ体積/面積とバッチサイズです。

以下に、単純な内部制約の定義が示されています。

```python
import numpy as np
from sympy import Symbol, Function, Number, pi, sin

import modulus.sym
from modulus.sym.hydra import to_absolute_path, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D, Line1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.eq.pde import PDE

class CustomPDE(PDE):
    def __init__(self, f=1.0):
        # coordinates
        x = Symbol("x")

        # make input variables
        input_variables = {"x": x}

        # make u function
        u = Function("u")(*input_variables)

        # source term
        if type(f) is str:
            f = Function(f)(*input_variables)
        elif type(f) in [float, int]:
            f = Number(f)

        # set equations
        self.equations = {}
        self.equations["custom_pde"] = (
            u.diff(x, 2) - f
        )  # "custom_pde" key name will be used in constraints


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    eq = CustomPDE(f=1.0)
    u_net = FullyConnectedArch(
        input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32
    )

    nodes = eq.make_nodes() + [u_net.make_node(name="u_network")]

    # add constraints to solver
    # make geometry
    x = Symbol("x")
    geo = Line1D(0, 1)

    # make domain
    domain = Domain()

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"custom_pde": 0},
        batch_size=100,
        bounds={x: (0, 1)},
    )
    domain.add_constraint(interior, "interior")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
```

## IntegralBoundaryConstraint

この制約は、PointwiseBoundaryConstraintと同様にジオメトリオブジェクトの境界上の点をサンプリングしますが、ポイントごとの損失を計算する代わりに、指定された変数のモンテカルロ積分を計算し、それに指定された値を割り当てて損失を計算します。数学的には以下のように示すことができます：

$$
L = \left| \int_{\partial \Omega} u_{net}(x,y,z) - \phi \right|^p = \left| \left(\frac{S}{B} \sum_{i}u_{net}(x_i, y_i, z_i)\right) - \phi \right|^p
$$

$Lは損失、$\partial \Omegaは境界、$u_{net}(x,y,z)は``outvar``内のキーのネットワーク予測です。$\phiはoutvarで指定された値であり、$pは損失のノルムです。$Sと$B`はそれぞれ体積/面積とバッチサイズです。

ここで注意すべきなのは、「batch_size」がここではやや異なる意味を持つということです。「batch_size」パラメータは、適用する積分のインスタンス数を定義するために使用されますが、「integral_batch_size」は境界上でサンプリングされる実際の点です。

以下に、単純な積分制約の定義が示されています。

```python
import numpy as np
from sympy import Symbol, Function, Number, pi, sin

import modulus.sym
from modulus.sym.hydra import to_absolute_path, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D, Line1D
from modulus.sym.domain.constraint import (
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.eq.pde import PDE


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    u_net = FullyConnectedArch(
        input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32
    )

    nodes = [u_net.make_node(name="u_network")]

    # add constraints to solver
    # make geometry
    x = Symbol("x")
    geo = Line1D(0, 1)

    # make domain
    domain = Domain()

    # integral
    integral = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=1,
        integral_batch_size=100,
    )
    domain.add_constraint(integral, "integral")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
```

## Discrete Constraints

離散制約では、制約は空間の離散化された表現から取られた固定点の構造に適用されます。これの最も単純な例は一様格子です。

## SupervisedGridConstraint

この制約は、グリッドデータ上で標準の教師ありトレーニングを行います。この制約は、遅延読み込みを使用する場合に特に重要な複数のワーカーの使用もサポートしています。この制約は、主にFourier Neural Operatorsのようなグリッドベースのモデルで使用されます。これらの制約で計算される損失は、上記の境界と内部の制約と同様にポイントごとです。

以下に、単純な教師ありグリッド制約の定義が示されています。

```python
import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.key import Key

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.dataset import HDF5GridDataset

from modulus.sym.utils.io.plotter import GridValidatorPlotter

from utilities import download_FNO_dataset


@modulus.main(config_path="conf", config_name="config_FNO")
def run(cfg: ModulusConfig) -> None:

    # load training/ test data
    input_keys = [Key("coeff", scale=(7.48360e00, 4.49996e00))]
    output_keys = [Key("sol", scale=(5.74634e-03, 3.88433e-03))]

    download_FNO_dataset("Darcy_241", outdir="datasets/")
    train_path = to_absolute_path(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5"
    )
    test_path = to_absolute_path(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5"
    )

    # make datasets
    train_dataset = HDF5GridDataset(
        train_path, invar_keys=["coeff"], outvar_keys=["sol"], n_examples=1000
    )
    test_dataset = HDF5GridDataset(
        test_path, invar_keys=["coeff"], outvar_keys=["sol"], n_examples=100
    )

    # make list of nodes to unroll graph on
    model = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.fno,
    )
    nodes = model.make_nodes(name="FNO", jit=cfg.jit)

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
        num_workers=4,  # number of parallel data loaders
    )
    domain.add_constraint(supervised, "supervised")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
```

## Defining a custom constraint

ユーザー定義のカスタム制約は、「modulus/domain/constraint/constraint.py」で定義されている「Constraint」クラスを継承して実装することができます。
制約を使用するために指定する必要がある3つのメソッドがあります。「load_data」メソッドは、内部のデータローダーからデータのミニバッチを読み込むために使用されます。「loss」メソッドは、トレーニング時に使用される損失を計算します。
最後に、「save_batch」メソッドは、デバッグや後処理のためにバッチを保存する方法を指定します。
この構造は一般的なものであり、変分法で使用されるような多くの複雑な制約を形成することができます。
これらのメソッドの実装に関する参照資料は、上記のベース制約のいずれかを参照してください。
