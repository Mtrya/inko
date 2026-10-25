# Diffusion RL：用扩散模型能高效计算的量优化奖励

## 奖励落在终点，网络预测发生在途中

向一个已经会画图的模型输入“两只狗在草地上奔跑”。模型生成了几张图片，有的画对了两只狗，有的只画了一只。如果有一个程序能检查数量，就可以给每张图片打分。现在希望改变模型，让它以后更经常画对。评分程序已经告诉我们哪些结果好，却没有告诉我们去噪网络的参数该怎样改。

把提示词记为 $c$，把模型参数记为 $\theta$。模型从标准高斯噪声 $Z\sim\mathcal N(0,I)$ 出发，经过完整的采样过程 $\mathcal S_\theta$，得到样本 $X=\mathcal S_\theta(Z,c)$。$I$ 是单位阵，表示噪声各坐标独立、方差为 $1$。$X$ 是模型操作的连续表示；如果模型在压缩后的图像表示中工作，就把解码成图片和打分合在同一个函数 $R(X,c)$ 里。$R$ 越大，表示生成结果越符合评价标准。

同一个提示词换一份噪声，生成结果就会变化。把这些结果的分布记为 $\pi_\theta(\cdot\mid c)$，用 $\pi_\theta(x\mid c)$ 表示其概率密度。固定一个提示词，训练目标是提高平均奖励：

$$
J(\theta\mid c)
=\mathbb E_{Z\sim\mathcal N(0,I)}[R(\mathcal S_\theta(Z,c),c)]
=\mathbb E_{X\sim\pi_\theta(\cdot\mid c)}[R(X,c)].
$$

训练多个提示词时，再对提示词取平均。固定 $c$ 可以把生成本身的问题单独拿出来：我们能采到 $X$，也能算出 $R(X,c)$，但还需要一个能更新 $\theta$ 的方向。

直接对第一个期望用链式法则，会得到

$$
\nabla_\theta J
=\mathbb E_Z\left[
\left(\frac{\partial \mathcal S_\theta(Z,c)}{\partial\theta}\right)^\top
\nabla_X R(X,c)
\right].
$$

这个写法要求评分函数能对生成表示 $X$ 求导，还要把导数穿过整个采样过程。数量检查器只返回“对”或“不对”，不会返回 $\nabla_XR$。人类比较两张图片、调用外部评分服务，也可能只得到一个分数或胜负结果。能够评价图片，并不自动意味着能够沿生成表示对奖励反向传播。

也可以对第二个期望求导。假定奖励函数不依赖 $\theta$，把梯度移入积分，并用 $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$，得到

$$
\begin{aligned}
\nabla_\theta J
&=\int R(x,c)\nabla_\theta\pi_\theta(x\mid c)\,dx\\
&=\mathbb E_{X\sim\pi_\theta}
\left[R(X,c)\nabla_\theta\log\pi_\theta(X\mid c)\right].
\end{aligned}
$$

这里的 $\nabla_\theta\log\pi_\theta$ 叫 score function，是对模型参数求导。扩散模型中的 score 则是带噪样本对数密度对样本位置的梯度，二者求导的变量不同。这个恒等式不需要对奖励求导，但要求我们能算生成样本的对数密度及其参数梯度。

扩散模型通常不能像自回归模型那样，沿生成过程直接加出终点的对数密度。以速度预测为例，网络每次给出的是当前位置该怎样移动。让时间 $t$ 随生成推进，从 $t=0$ 的噪声走到 $t=1$ 的样本，一条采样轨迹 $Y_t$ 满足

$$
\frac{dY_t}{dt}=v_\theta(Y_t,t,c),\qquad Y_0=Z,\qquad X=Y_1.
$$

$v_\theta$ 是网络预测的速度场。只要会反复求它，就能数值求解这个常微分方程（ODE），得到样本。终点密度还依赖整个流动怎样压缩和拉伸体积；求解轨迹本身不会顺带给出终点密度的参数梯度。

一种办法是给每一步加入随机性，把相邻状态间的转移写成高斯分布。记离散时刻为 $t_0,\ldots,t_N$，一步转移密度为 $T_\theta(Y_{t_{k+1}}\mid Y_{t_k},c)$。整条轨迹的概率密度就能分解成初始噪声密度与这些转移密度的乘积。初始噪声不依赖参数，因此轨迹对数密度的梯度是

$$
\sum_{k=0}^{N-1}
\nabla_\theta\log T_\theta(Y_{t_{k+1}}\mid Y_{t_k},c).
$$

终点奖励乘上这个和，可以构造轨迹上的 policy gradient。Flow-GRPO 走的就是这条路：使转移密度可计算，再用奖励调整转移。训练需要保存或重建用于计算梯度的轨迹状态与转移信息，并与随机转移的具体构造相连。[DiffusionNFT 的背景部分](https://arxiv.org/html/2509.16117v2#S2.SS2)给出了对应的扩散形式。

扩散还有一种已经很熟悉的计算：拿一张干净图片，重新加噪，回归一个已知的去噪或速度目标。这个计算只需要一次局部网络预测，不需要终点密度，也不需要沿生成轨迹反向传播。如果奖励能通过这种回归改变生成分布，就能利用扩散模型原有的训练方式来做策略改进。

## 给高奖励样本更大的回归权重

把当前用于生成数据的模型暂时固定，记为 $v_{\mathrm{old}}$，它产生的终点分布记为 $\pi_{\mathrm{old}}$。生成一张样本 $X$ 后，重新抽一份与 $X$ 独立的高斯噪声 $\epsilon$，构造

$$
X_t=(1-t)\epsilon+tX,\qquad U=X-\epsilon.
$$

$X_t$ 是把干净样本重新加噪后得到的训练输入，$U$ 是这条线性插值的速度。它们与刚才的采样轨迹 $Y_t$ 分开记：$Y_t$ 是模型实际走过的状态，$X_t$ 是拿终点和新噪声构造的状态。新抽的 $\epsilon$ 也与原来生成 $X$ 时用的 $Z$ 分开记。

给定一个状态 $s=(x,t,c)$，网络只能看到位置、时刻和提示词。同一状态可能由不同的 $X$ 与 $\epsilon$ 配对产生。平方误差回归会把这些配对的速度取条件平均：

$$
u_{\mathrm{old}}(s)
:=\mathbb E[U\mid X_t=x,c].
$$

这个平均为什么是最优解，可以直接展开。平方误差 $\|v-U\|^2$ 是各坐标误差平方的和。对固定状态上的任意预测向量 $v$，有

$$
\mathbb E[\|v-U\|^2\mid s]
=\|v-u_{\mathrm{old}}(s)\|^2
+\mathbb E[\|U-u_{\mathrm{old}}(s)\|^2\mid s].
$$

交叉项消失，是因为 $\mathbb E[U-u_{\mathrm{old}}\mid s]=0$。右边第二项不依赖 $v$，所以回归最优值就是 $u_{\mathrm{old}}$。这是 flow matching 用单个样本的条件速度学习平均速度的原因。

现在给每张样本一个权重 $w(R)$。权重非负，并且奖励越高，权重不减。记 $W=\mathbb E_{\pi_{\mathrm{old}}}[w(R)]$，要求 $0<W<\infty$。用这些权重重新分配终点的概率，得到一个定义明确的分布

$$
q_w(X\mid c)
=\frac{w(R(X,c))}{W}\pi_{\mathrm{old}}(X\mid c).
$$

它的平均奖励与原分布之差是

$$
\begin{aligned}
\mathbb E_{q_w}[R]-\mathbb E_{\pi_{\mathrm{old}}}[R]
&=\frac{\mathbb E_{\pi_{\mathrm{old}}}[w(R)R]}{W}
-\mathbb E_{\pi_{\mathrm{old}}}[R]\\
&=\frac{\operatorname{Cov}_{\pi_{\mathrm{old}}}(R,w(R))}{W}.
\end{aligned}
$$

协方差 $\operatorname{Cov}(A,B)$ 的定义是 $\mathbb E[AB]-\mathbb E[A]\mathbb E[B]$。对这里的两个标量 $R$ 与 $w(R)$，协方差非负。取奖励的两个独立副本 $R,R'$，展开乘积可得

$$
\operatorname{Cov}(R,w(R))
=\frac12\mathbb E[(R-R')(w(R)-w(R'))]\geq0.
$$

最后一个不等号来自单调性：两个差总是同号。于是 $q_w$ 的平均奖励至少与原分布一样高；当奖励差异确实对应权重差异时，平均奖励会提高。

这个分布可以直接用加权回归来拟合。把 $t$ 从 $[0,1]$ 均匀抽取，$X$ 从 $\pi_{\mathrm{old}}$ 抽取，$\epsilon$ 独立抽取，最小化

$$
\mathcal L_w(\theta)
=\mathbb E_{t,X,\epsilon}
\left[w(R(X,c))\|v_\theta(X_t,t,c)-U\|^2\right].
$$

由于权重只作用于 $X$，这个期望等于

$$
W\,\mathbb E_{t,\,X\sim q_w,\,\epsilon}
\left[\|v_\theta(X_t,t,c)-U\|^2\right].
$$

常数 $W$ 不改变最优点。因此，加权回归就是在 $q_w$ 提供的端点上做 flow matching。也可以直接在状态 $s$ 上求导：

$$
\nabla_v\mathbb E[w(R)\|v-U\|^2\mid s]
=2\mathbb E[w(R)\mid s]v-2\mathbb E[w(R)U\mid s].
$$

在 $\mathbb E[w(R)\mid s]>0$ 的位置，令导数为零，得到

$$
u_w(s)
=\frac{\mathbb E[w(R)U\mid s]}{\mathbb E[w(R)\mid s]}.
$$

贝叶斯公式把这个比值解释为 $q_w$ 下的条件平均速度。它对应的密度路径就是从独立噪声到 $q_w$ 样本的线性插值。取任意只在有限区域内非零的光滑函数 $\varphi$，有

$$
\frac{d}{dt}\mathbb E_{q_w}[\varphi(X_t)]
=\mathbb E_{q_w}[\nabla\varphi(X_t)\cdot U]
=\mathbb E_{q_w}[\nabla\varphi(X_t)\cdot u_w(X_t,t,c)].
$$

把插值得到的 $X_t$ 的密度记为 $q_{w,t}$。将刚才等式两边的期望写成积分，右边分部积分，便有

$$
\int\varphi(x)\,\partial_tq_{w,t}(x)\,dx
=-\int\varphi(x)\,\nabla\cdot(q_{w,t}(x)u_w(x,t,c))\,dx.
$$

这个等式对任意光滑测试函数 $\varphi$ 成立，因此

$$
\partial_tq_{w,t}+\nabla\cdot(q_{w,t}u_w)=0.
$$

这叫连续性方程：某处概率密度的增加，来自速度场带来的净流入。现在另做一次实际采样，从高斯噪声出发，沿 $\dot Y_t=u_w(Y_t,t,c)$ 积分，把采样状态的密度记为 $p_t$。对这条轨迹做同样的链式法则计算，会得到同一个方程 $\partial_t p_t+\nabla\cdot(p_tu_w)=0$。

于是，插值定义的密度 $q_{w,t}$ 与 ODE 产生的密度 $p_t$ 使用同一个速度场，也从同一个标准高斯分布出发。在速度场足够光滑、输运方程的解唯一时，两者就是同一条密度路径。插值在 $t=1$ 到达 $q_w$，ODE 也就到达 $q_w$。回归确定速度场，密度演化的唯一性把它接到生成分布上。[flow matching 笔记](flow_matching.md)进一步展开了这个连接。

到这里，奖励已经通过一个可计算的损失进入了扩散训练。证明的前提是把相应的速度场拟合出来，并沿它采样。实际训练每轮只有有限样本，也只更新有限次；降低一些加权回归误差，还没有等同于生成分布已经变成 $q_w$。对已有模型做一次怎样的调整，成了接下来真正需要处理的问题。

## 正负样本给出同一个改进方向

DiffusionNFT 把奖励转换成一个落在 $[0,1]$ 内的权重 $r(X,c)$。$r$ 越接近 $1$，样本越受偏好。原始评分 $R$ 与这个权重分开记，因为评分可以有任意量纲，而 $r$ 还要用于构造概率。

论文的做法是在同一个提示词下生成 $K$ 个样本，用组内平均奖励 $\bar R_c$ 居中，再用正数 $\zeta_c$ 缩放：

$$
r(X,c)=\frac12+\frac12\operatorname{clip}
\left(\frac{R(X,c)-\bar R_c}{\zeta_c},-1,1\right).
$$

$\operatorname{clip}(a,-1,1)$ 把 $a$ 截到 $[-1,1]$。高于平均奖励的样本得到 $r>1/2$，低于平均奖励的样本得到 $r<1/2$。为看清分布关系，先把 $r$ 当作一个固定函数；组内估计是训练时构造这个函数的办法。

想象给每张样本做一次随机标记：以概率 $r(X,c)$ 标为正，以概率 $1-r(X,c)$ 标为负。记总体正标记概率为 $\mu=\mathbb E_{\pi_{\mathrm{old}}}[r]$，并取 $0<\mu<1$，让两种标记都有样本。贝叶斯公式给出

$$
\pi_+(X\mid c)=\frac{r(X,c)}{\mu}\pi_{\mathrm{old}}(X\mid c),\qquad
\pi_-(X\mid c)=\frac{1-r(X,c)}{1-\mu}\pi_{\mathrm{old}}(X\mid c).
$$

这两个分布来自同一批生成结果。训练时可以直接使用 $r$ 和 $1-r$ 作为权重，不必真的随机分组。把两式分别乘回分母再相加，还能恢复原分布：

$$
\pi_{\mathrm{old}}=\mu\pi_++(1-\mu)\pi_-.
$$

前面的协方差计算同时说明了三个分布的奖励关系：

$$
\mathbb E_{\pi_+}[R]-\mathbb E_{\pi_{\mathrm{old}}}[R]
=\frac{\operatorname{Cov}(R,r)}{\mu}\geq0,
\qquad
\mathbb E_{\pi_{\mathrm{old}}}[R]-\mathbb E_{\pi_-}[R]
=\frac{\operatorname{Cov}(R,r)}{1-\mu}\geq0.
$$

正分布偏向高奖励，负分布偏向低奖励。现在把两者都接上同一条加噪路径，分别得到条件平均速度 $u_+(s)$ 与 $u_-(s)$。要把端点之间的关系搬到中间状态，混合权重也必须随观察到的状态更新。

具体地，定义

$$
m(s):=\mathbb E[r(X,c)\mid X_t=x,c].
$$

$m(s)$ 是已经看到带噪状态后，样本会被标为正的概率。它与总体比例 $\mu$ 不同：总体各占一半，也可能在某个状态附近几乎全是正样本。按同样的正负标记对条件分布分组，条件期望的全概率公式给出

$$
u_{\mathrm{old}}(s)=m(s)u_+(s)+(1-m(s))u_-(s).
$$

这个等式也能从加权回归的比值直接看出。在 $0<m(s)<1$ 的状态，

$$
u_+(s)=\frac{\mathbb E[rU\mid s]}{m(s)},\qquad
u_-(s)=\frac{\mathbb E[(1-r)U\mid s]}{1-m(s)}.
$$

两式乘回各自分母后相加，就是 $\mathbb E[U\mid s]$。移项则得到

$$
m(s)(u_+-u_{\mathrm{old}})
=(1-m(s))(u_{\mathrm{old}}-u_-).
$$

这正是 NFT 的正负方向共线关系：朝正场移动，与远离负场移动，给出同一个方向，只差一个正的倍数。两边还可以写成一个更直接的量：

$$
\begin{aligned}
m(s)(u_+-u_{\mathrm{old}})
&=\mathbb E[rU\mid s]-\mathbb E[r\mid s]\mathbb E[U\mid s]\\
&=\operatorname{Cov}(r,U\mid s)
\;=:g(s).
\end{aligned}
$$

$r$ 是标量，$U$ 是向量，所以这里的协方差也是向量，每个分量都在衡量：较高的奖励权重是否伴随该方向上的较大速度。$g(s)$ 给出了条件平均速度随奖励偏好改变的方向。DiffusionNFT 用后验混合关系推导这条方向；后续工作 DiffusionART 将它进一步写成条件协方差。两种写法描述的是同一个量。[NFT，附录 A](https://arxiv.org/html/2509.16117v2#A1)；[ART，附录 B.1](https://arxiv.org/html/2610.12572v1#A2.SS1)。

## 把双峰里的概率往右移

用一个能完整算出的例子看这些量。令当前终点分布是一维双峰：

$$
\pi_{\mathrm{old}}(X)
=\frac12\mathcal N(-1,\sigma_{\mathrm{data}}^2)
+\frac12\mathcal N(1,\sigma_{\mathrm{data}}^2),\qquad
\sigma_{\mathrm{data}}^2=0.1.
$$

每个峰各占一半。用一个取值为 $-1$ 或 $+1$ 的变量 $M$ 表示生成 $X$ 的峰，$X\mid M$ 服从 $\mathcal N(M,\sigma_{\mathrm{data}}^2)$。奖励偏好右峰，但只看得到 $X$。令奖励等于从 $X$ 判断其来自右峰的后验概率；这个奖励本来就在 $[0,1]$ 内，直接取 $r=R$：

$$
r(X)=R(X)
=\Pr(M=+1\mid X)
=\frac{1}{1+\exp(-2X/\sigma_{\mathrm{data}}^2)}.
$$

最后一式来自两边高斯密度之比：$\exp(-(X-1)^2/(2\sigma_{\mathrm{data}}^2))/\exp(-(X+1)^2/(2\sigma_{\mathrm{data}}^2))=\exp(2X/\sigma_{\mathrm{data}}^2)$。因此，给定一个数 $X$ 就能打分，不需要知道生成时实际选了哪个峰。

对称性给出 $\mu=1/2$。根据后验概率的定义，$r(X)\pi_{\mathrm{old}}(X)=\tfrac12\mathcal N(X;1,\sigma_{\mathrm{data}}^2)$，所以

$$
\pi_+=\mathcal N(1,\sigma_{\mathrm{data}}^2),\qquad
\pi_-=\mathcal N(-1,\sigma_{\mathrm{data}}^2).
$$

原分布的平均奖励是 $0.5$；在右峰分布下积分同一个奖励函数，得到约 $0.998794$。重新分配概率提高了奖励，而且这个变化可以直接从两个分布算出来。

给定峰 $M$，加噪状态 $X_t=(1-t)\epsilon+tX$ 仍是高斯，均值为 $tM$，方差为

$$
V_t=(1-t)^2+t^2\sigma_{\mathrm{data}}^2.
$$

对联合高斯变量，条件均值等于原均值加上“协方差除以观测方差，再乘观测偏差”。这里 $\operatorname{Cov}(X,X_t\mid M)=t\sigma_{\mathrm{data}}^2$，$\operatorname{Cov}(\epsilon,X_t\mid M)=1-t$，因此

$$
\mathbb E[X\mid X_t=x,M]
=M+\frac{t\sigma_{\mathrm{data}}^2}{V_t}(x-tM),
\qquad
\mathbb E[\epsilon\mid X_t=x,M]
=\frac{1-t}{V_t}(x-tM).
$$

两式相减，就得到单个峰的条件平均速度。为缩短式子，记

$$
b_t=\frac{t\sigma_{\mathrm{data}}^2-(1-t)}{V_t}.
$$

于是正负两个场分别为

$$
u_+(x,t)=1+b_t(x-t),\qquad
u_-(x,t)=-1+b_t(x+t).
$$

两个带噪高斯的均值现在位于 $\pm t$。再次用密度比求后验，得到

$$
m(x,t)=\frac{1}{1+\exp(-2tx/V_t)}.
$$

由于 $X_t$ 在给定 $X$ 后不再包含关于峰 $M$ 的额外信息，这个后验也等于 $\mathbb E[r(X)\mid X_t=x]$。原场与奖励方向便是

$$
u_{\mathrm{old}}=m u_++(1-m)u_-,\qquad
g=m(1-m)(u_+-u_-).
$$

在 $t=0.5$ 时，代入几个位置：

| $x$ | $m(x,t)$ | $u_+$ | $u_-$ | $u_{\mathrm{old}}$ | $g$ |
|---:|---:|---:|---:|---:|---:|
| $-0.5$ | $0.13965$ | $2.63636$ | $-1.00000$ | $-0.49217$ | $0.43691$ |
| $0$ | $0.50000$ | $1.81818$ | $-1.81818$ | $0$ | $0.90909$ |
| $0.5$ | $0.86035$ | $1.00000$ | $-2.63636$ | $0.49217$ | $0.43691$ |

三个位置的 $g$ 都指向右边，但大小不同。在中间，状态还不能分辨来自哪个峰，改变两边的后验权重会显著改变平均速度；靠近某个峰时，原后验已经有所偏向，速度变化较小。在 $X\in[-6,6]$ 上用间隔 $2.5\times10^{-4}$ 的网格对条件密度做梯形积分，得到的 $m$、$u_{\mathrm{old}}$ 和 $g$ 与表中数值舍入前的闭式值之差小于 $10^{-12}$。

这个例子还把总体与局部的区别摆出来了。总体正负比例各为 $1/2$，但在 $x=0.5,t=0.5$，正标记概率已经达到 $0.86035$。对一组奖励居中，不能使每个带噪状态里的正负权重都相等。

## 用一个模型实现正负两个方向

理想的 $u_+$、$u_-$ 说明了怎样改变速度，但直接求它们需要对未知的条件分布取平均。实际训练手里只有生成样本、奖励、重新抽取的噪声，以及旧网络的预测。NFT 用这些量构造一个平方误差，使同一个可训练模型同时接收正负反馈。

在状态 $s$ 上，把当前网络相对冻结旧网络的变化记为

$$
d(s)=v_\theta(s)-v_{\mathrm{old}}(s).
$$

选一个正数 $\beta$，用同一个变化 $d$ 构造两种预测：

$$
v_\theta^+(s)=v_{\mathrm{old}}(s)+\beta d(s),\qquad
v_\theta^-(s)=v_{\mathrm{old}}(s)-\beta d(s).
$$

模型向某个方向改变时，正分支沿这个方向改变，负分支沿相反方向改变。两者都与样本的条件速度 $U$ 做回归，只是权重不同：

$$
\ell_{\mathrm{NFT}}
=r\|v_\theta^+-U\|^2+(1-r)\|v_\theta^--U\|^2,
\qquad
\mathcal L_{\mathrm{NFT}}=\mathbb E_{t,X,\epsilon}[\ell_{\mathrm{NFT}}].
$$

$r$ 接近 $1$ 时，损失更强调让正分支预测 $v_\theta^+$ 靠近 $U$；$r$ 接近 $0$ 时，更强调让负分支预测 $v_\theta^-$ 靠近 $U$。从 $v_\theta=v_{\mathrm{old}}$ 出发时，前者让模型朝样本速度移动，后者由于 $d$ 带负号而让模型向相反方向移动。两个平方误差的权重始终非负。

把样本速度与旧预测之差记为 $h=U-v_{\mathrm{old}}(s)$，损失便变成

$$
\begin{aligned}
\ell_{\mathrm{NFT}}
&=r\|\beta d-h\|^2+(1-r)\|-\beta d-h\|^2\\
&=\beta^2\|d\|^2-2\beta(2r-1)\langle d,h\rangle+\|h\|^2\\
&=\beta^2\left\|d-\frac{2r-1}{\beta}h\right\|^2
+4r(1-r)\|h\|^2.
\end{aligned}
$$

第二个等号把两份平方展开；第三个等号配方，并用了 $1-(2r-1)^2=4r(1-r)$。$\langle d,h\rangle$ 表示向量内积。旧预测、样本和奖励在更新时固定，所以最后一项与可训练参数无关。

因此，NFT 的基本目标等价于回归一个样本目标

$$
v_{\mathrm{tgt}}^{\mathrm{NFT}}
=v_{\mathrm{old}}+\frac{2r-1}{\beta}(U-v_{\mathrm{old}}).
$$

这个目标完全可算。$r=1/2$ 时，它就是旧预测；$r>1/2$ 时，它沿旧预测到样本速度的方向移动；$r<1/2$ 时，它沿相反方向移动。$1/\beta$ 控制目标移动的幅度。只需要保留终点和奖励，就能重新加噪构造这些回归样本。

样本目标包含噪声，网络最终回归的是它在给定状态下的平均。暂时假设旧网络恰好等于端点诱导的条件平均速度，即 $v_{\mathrm{old}}=u_{\mathrm{old}}$。此时

$$
\begin{aligned}
\mathbb E[(2r-1)(U-u_{\mathrm{old}})\mid s]
&=2\mathbb E[rU\mid s]-2m(s)u_{\mathrm{old}}\\
&=2g(s).
\end{aligned}
$$

这里用到了 $\mathbb E[U-u_{\mathrm{old}}\mid s]=0$。在函数空间里逐状态优化，NFT 的最优场于是为

$$
v^*(s)=v_{\mathrm{old}}(s)+\frac{2}{\beta}g(s).
$$

这就把正负分布定义出的方向接到了一个可训练的损失上。这个结论对应 [NFT 的策略优化定理及附录推导](https://arxiv.org/html/2509.16117v2#A1)；把损失配成单个回归目标的写法也见于 [ART 附录 B.2.1](https://arxiv.org/html/2610.12572v1#A2.SS2.SSS1)。

它与直接拟合正分布仍有一个具体区别。由 $g=m(u_+-u_{\mathrm{old}})$，得到

$$
v^*=u_{\mathrm{old}}+\frac{2m(s)}{\beta}(u_+-u_{\mathrm{old}}).
$$

只有系数恰好等于 $1$，这个场才等于 $u_+$。在双峰例子里取 $\beta=1$，$x=0,t=0.5$ 的系数为 $1$；$x=0.5,t=0.5$ 的系数约为 $1.72070$，目标速度约为 $1.36599$，已经超过该处的 $u_+=1$。正负方向共线确定了局部方向，固定 $\beta$ 还选择了沿这个方向走多远。因此，正分布的奖励较高，并不能直接替代对这个新场生成结果的检查。

而且，推导刚才还用了 $v_{\mathrm{old}}=u_{\mathrm{old}}$。这两个对象的来源不同：一个是采样器使用的网络，另一个是重新加噪后的条件平均。检查这个等号，会发现 NFT 更新中的另一项。

## 同一个终点分布，可以来自不同的速度场

生成 $X$ 时，旧网络沿自己的轨迹 $Y_t$ 走到终点。训练时，$X_t=(1-t)\epsilon+tX$ 把这个终点与独立噪声重新配对。两种过程拥有同一个干净端点分布，却未必拥有同一条中间密度路径。

双峰例子可以把这个差异单独造出来。假设 $u_{\mathrm{old}}(x,t)$ 被精确算出，并能精确积分。让实际采样走同一条轨迹，但把时间进度改成 $t^2$：在新时刻 $t$，到达原轨迹时刻 $t^2$ 的位置。链式法则给出的实际采样速度是

$$
v_{\mathrm{old}}(x,t)=2t\,u_{\mathrm{old}}(x,t^2).
$$

$t=0$ 与 $t=1$ 的位置保持不变，所以终点分布仍然是原来的双峰。中途走得快慢不同，速度场就不同。即使没有拟合误差，也有理由出现 $v_{\mathrm{old}}\ne u_{\mathrm{old}}$。

把二者在同一状态的差记为

$$
e(s)=u_{\mathrm{old}}(s)-v_{\mathrm{old}}(s).
$$

要看这个差怎样影响训练，对 NFT 展开后的损失求输出梯度：

$$
-\nabla_{v_\theta}\ell_{\mathrm{NFT}}
=2\beta(2r-1)h-2\beta^2d.
$$

先看当前模型与采样模型相同的更新起点，$v_\theta=v_{\mathrm{old}}$，因而 $d=0$。把负梯度除以正常数 $2\beta$，再对状态取条件平均，得到输出空间里的更新方向

$$
F_{\mathrm{NFT}}(s)
=\mathbb E[(2r-1)(U-v_{\mathrm{old}})\mid s].
$$

在括号中加减 $u_{\mathrm{old}}$，原来的理想方向与场的差异便分开了：

$$
\begin{aligned}
F_{\mathrm{NFT}}
&=\mathbb E[(2r-1)(U-u_{\mathrm{old}})\mid s]
+\mathbb E[2r-1\mid s](u_{\mathrm{old}}-v_{\mathrm{old}})\\
&=2g(s)+(2m(s)-1)e(s).
\end{aligned}
$$

第一项是前面推导出的奖励方向。第二项的方向来自两个过程的速度差，强度由当前状态下的正负不平衡决定。在 $m>1/2$ 的位置，它把模型拉向重新加噪所定义的平均场；在 $m<1/2$ 的位置，它把模型推离这个场。这就是 ART 分析中的 consistency force。[ART，命题 1 与附录 B.2.2](https://arxiv.org/html/2610.12572v1#A2.SS2.SSS2)。

沿时间重排后的双峰采样器，在 $x=0.5,t=0.5$ 处，有 $u_{\mathrm{old}}\approx0.49217$、$v_{\mathrm{old}}\approx-0.35212$，所以 $e\approx0.84429$。该处的理想奖励项 $2g\approx0.87381$，附加项 $(2m-1)e\approx0.60848$，实际方向则约为 $1.48229$。把 $(2r-1)(U-v_{\mathrm{old}})$ 按条件密度直接积分，也得到同一个值。这个附加项来自路径变化，终点分布并没有变。

正负分支各自的方向也可以算出。在同一个更新起点，沿用除以 $2\beta$ 的归一化，正分支给出

$$
F_+(s)=\mathbb E[r(U-v_{\mathrm{old}})\mid s]=g(s)+m(s)e(s),
$$

负分支给出

$$
F_-(s)=-\mathbb E[(1-r)(U-v_{\mathrm{old}})\mid s]
=g(s)-(1-m(s))e(s).
$$

两个分支在 $g$ 上同向，在 $e$ 上反向。把它们相加，奖励方向增强，而速度差只剩下 $(2m-1)e$。这解释了正负联合训练比任一分支单独训练多做了什么。ART 的消融也观察到：在每轮都由当前模型生成训练数据的 on-policy 设置下，单独保留任一分支都会崩溃，联合训练能明显改善学习过程。

不过，总体奖励居中只控制 $\mu$，不能把每个状态的 $m(s)$ 都变成 $1/2$。双峰表格已经给出了一个例子。即使 $\mathbb E_s[2m(s)-1]=0$，也不能推出 $\mathbb E_s[(2m(s)-1)e(s)]=0$，因为 $e(s)$ 同样随状态变化。

还有一层平均发生在网络参数上。真正的参数梯度是

$$
\nabla_\theta\mathcal L
=\mathbb E\left[
\left(\frac{\partial v_\theta(s)}{\partial\theta}\right)^\top
\nabla_{v_\theta}\ell
\right].
$$

括号中的矩阵描述每个参数怎样影响该状态的预测。不同状态使用同一组参数，却通过不同的矩阵贡献梯度。只看输出方向的平均大小，还不能判断最终参数更新中哪一项占主导。

## 冻结的旧预测还提供了一股恢复力

当模型已经相对旧预测移动了一段距离，$d=v_\theta-v_{\mathrm{old}}$ 不再为零。NFT 的负输出梯度中还保留着

$$
-2\beta^2d.
$$

这项总指向旧预测，因为它是 $\beta^2\|d\|^2$ 的负梯度。当前位置离旧预测越远，恢复作用越强。它与奖励方向一起决定当前更新，而不需要额外写一个恢复损失。

NFT 用参数的指数移动平均（EMA）更新采样模型：

$$
\theta_{\mathrm{old}}\leftarrow
\rho\theta_{\mathrm{old}}+(1-\rho)\theta,\qquad 0\leq\rho<1.
$$

$\rho$ 越接近 $1$，旧参数保留得越多，采样模型跟随训练模型的速度越慢。下一轮训练开始时，当前模型与这个滞后的参照之间通常已有距离，恢复项因而能立即起作用。

这个机制可以用消融区分。ART 保留 EMA 模型产生的样本，却在训练梯度里去掉旧预测的参照，结果仍然崩溃。另一个实验直接用当前模型采样，把同一轮数据分成两批，固定采样时的参照，连续更新两次。第一次更新从 $d=0$ 出发，第二次就出现了恢复力；训练能维持稳定，但奖励提升明显变慢。[ART，§3.2](https://arxiv.org/html/2610.12572v1#S3.SS2)。

因此，稳定性与旧预测在损失中的位置有关。来自另一份模型参数的数据称为 off-policy 数据。on-policy 与 off-policy 描述采样来源，恢复项则由实际训练目标决定。

到这一步，NFT 的平方误差已经分出了三个作用：奖励相关的 $g$、两个场不一致带来的 $e$，以及偏离冻结预测后出现的恢复力。ART 保留回归和恢复的做法，重新构造其中的奖励方向，使构造过程直接使用终点奖励与采样预测。

## 从一组终点直接算出共同方向

同一个提示词下，已经生成了若干样本，也知道各自的奖励。即使不知道任一带噪状态的后验分布，仍然可以计算这些终点的均值、奖励的均值，以及两者怎样一起变化。ART 从这组可以直接估计的量出发构造方向。

让高奖励样本的权重随一个参数 $\tau\geq0$ 增长，定义

$$
\pi_\tau(X\mid c)
=\frac{\pi_{\mathrm{old}}(X\mid c)\exp(\tau R(X,c))}
{\mathbb E_{\pi_{\mathrm{old}}}[\exp(\tau R)]}.
$$

$\tau$ 控制偏好强度，与采样时间 $t$ 是不同的量。$\tau=0$ 时恢复原分布，增大 $\tau$ 会提高高奖励终点的相对权重。这种指数重加权叫 reward tilting。

要知道分布的均值怎样改变，不必把整个新分布拟合出来。均值是两个期望的比值，对 $\tau$ 求导即可。以有界奖励和有限的一阶样本矩为例，求导可以移入期望；在 $\tau=0$ 处，分母等于 $1$，于是

$$
\begin{aligned}
\left.\frac{d}{d\tau}\mathbb E_{\pi_\tau}[X\mid c]\right|_{\tau=0}
&=\left.\frac{d}{d\tau}
\frac{\mathbb E_{\pi_{\mathrm{old}}}[X e^{\tau R}\mid c]}
{\mathbb E_{\pi_{\mathrm{old}}}[e^{\tau R}\mid c]}
\right|_{\tau=0}\\
&=\mathbb E[XR\mid c]-\mathbb E[X\mid c]\mathbb E[R\mid c]\\
&=\operatorname{Cov}_{\pi_{\mathrm{old}}}(R,X\mid c)
\;=:C(c).
\end{aligned}
$$

这就是 ART 使用的端点协方差。它与 NFT 的 $g(s)$ 都是标量奖励与向量之间的协方差，但条件不同：$g(s)$ 固定了一个带噪状态，$C(c)$ 只固定提示词，对整组端点汇总。[ART，§4.1 与附录 B.3](https://arxiv.org/html/2610.12572v1#A2.SS3)。

对同一个比值，把分子里的 $X$ 换成 $R$，还能得到

$$
\frac{d}{d\tau}\mathbb E_{\pi_\tau}[R\mid c]
=\mathbb E_{\pi_\tau}[R^2\mid c]
-\mathbb E_{\pi_\tau}[R\mid c]^2
=\operatorname{Var}_{\pi_\tau}(R\mid c)\geq0.
$$

因此，指数重加权提高奖励，而 $C(c)$ 描述它在当前分布处怎样移动样本均值。两个结论来自同一次商法则计算。

在双峰例子里，$R=r$，$\mathbb E[X]=0$，且 $r(X)\pi_{\mathrm{old}}(X)$ 是右峰密度的一半，所以

$$
C=\mathbb E[RX]=\frac12\mathbb E_{\mathcal N(1,0.1)}[X]=\frac12.
$$

正的协方差把均值推向右边。用 $\tau=\pm10^{-4}$ 的重加权分布做中心差分，均值导数约为 $0.4999999996$；平均奖励导数约为 $0.2493971711$，与原奖励方差 $0.2493971713$ 一致。

均值只保留分布的一部分信息。同一个对称双峰若改用奖励 $\widetilde R(X)=\mathbf1\{|X|>1\}$，其中 $\mathbf1\{\cdot\}$ 在条件成立时取 $1$、否则取 $0$，就表示偏好绝对值较大的样本。$\widetilde R$ 是偶函数，$X$ 是奇函数，因此 $\operatorname{Cov}(\widetilde R,X)=0$。然而奖励有 $0$ 和 $1$ 两种结果，方差为正，指数重加权仍会提高平均奖励。这个变化增加了两侧的概率质量，均值保持不动。所以，把均值变化方向用作速度引导，是 ART 采用的一种可计算的构造；它没有保留完整分布变化的所有信息。

端点协方差也能与条件速度联系起来。因为训练路径上的 $U=X-\epsilon$，并且重新抽取的 $\epsilon$ 与 $X$ 独立，

$$
\operatorname{Cov}(R,U\mid c)
=\operatorname{Cov}(R,X\mid c)-\operatorname{Cov}(R,\epsilon\mid c)
=C(c).
$$

这里必须使用独立的新噪声 $\epsilon$。实际生成时的初始噪声 $Z$ 会决定终点 $X$，不能用同一个独立性论证。在纯噪声端 $t=0$，$X_t=\epsilon$ 没有透露端点信息。若 $r$ 没有触发截断，它与 $R$ 是线性关系，此时 NFT 的 $g$ 就等于 $C(c)/(2\zeta_c)$。进入中间时刻后，$g$ 会随状态改变，而 $C(c)$ 仍然只属于整个提示词。

实际计算时，对每个提示词生成 $K$ 个终点。记奖励均值与终点均值为 $\bar R_c$ 和 $\bar X_c$，经验协方差是

$$
\widehat C(c)
=\frac1K\sum_{i=1}^K
(R_i-\bar R_c)(X_i-\bar X_c).
$$

较高奖励的样本以正系数贡献它相对均值的偏移，较低奖励的样本以负系数贡献偏移。所有样本的贡献先在同一坐标空间相加，形成一个共同向量。

ART 再把这个向量的均方根归一化。设输出表示共有 $D$ 个坐标，对向量 $a$ 定义 $\operatorname{RMS}(a)=\sqrt{D^{-1}\sum_{j=1}^D a_j^2}$，然后取

$$
G(c)=\frac{\widehat C(c)}{\operatorname{RMS}(\widehat C(c))}.
$$

在协方差为零时取 $G(c)=0$。非零时，归一化使方向的均方根幅度为 $1$，消除了奖励量纲对整体幅度的影响。在一维右峰例子里，$C>0$，归一化后的方向就是 $G=+1$。

如果评价器只会比较两张图片，也可以做同样的计算。给胜者 $X_{\mathrm{win}}$ 的标签 $1$，败者 $X_{\mathrm{lose}}$ 的标签 $0$，标签均值为 $1/2$，端点均值为 $\bar X=(X_{\mathrm{win}}+X_{\mathrm{lose}})/2$。这两个样本的经验协方差为

$$
\begin{aligned}
\widehat C_{\mathrm{pair}}
&=\frac12\left[
\frac12(X_{\mathrm{win}}-\bar X)
-\frac12(X_{\mathrm{lose}}-\bar X)
\right]\\
&=\frac14(X_{\mathrm{win}}-X_{\mathrm{lose}}).
\end{aligned}
$$

归一化后，方向从败者指向胜者。若两张平局，就都给 $1/2$，贡献为零。同一个提示词有多对比较时，每一对的标签均值都为 $1/2$，可以先汇总各对的协方差，再统一归一化。这是 ART-Pairwise 与标量奖励版本共用同一计算的原因。[ART，附录 C.3](https://arxiv.org/html/2610.12572v1#A3.SS3)。

## 把方向加到实际采样预测上

NFT 的训练状态由终点重新加噪得到。ART 保存实际采样轨迹中每一步出发时的状态 $Y_t$，以及旧模型当时的预测。这些状态满足 $0\leq t<1$；$t=1$ 的终点用于评分，不再发起下一步，也不参与后面的参考转移计算。对同一个提示词的各条轨迹，都把共同方向 $G(c)$ 加到采样预测上，形成冻结目标

$$
v_{\mathrm{tgt}}^{\mathrm{ART}}(s)
=v_{\mathrm{old}}(s)+G(c),\qquad s=(Y_t,t,c).
$$

这个目标直接建立在实际使用的速度上，没有把旧网络替换成独立加噪下的条件平均。代价也随之改变：ART 的训练需要缓存采样状态和预测，NFT 则能仅凭终点与奖励重新构造训练状态。

在单个状态上，用一个非负权重 $\omega(s)$ 做均方误差回归：

$$
\ell_{\mathrm{ART}}(s)
=\frac{\omega(s)}{D}
\|v_\theta(s)-v_{\mathrm{old}}(s)-G(c)\|^2
=\frac{\omega(s)}{D}\|d(s)-G(c)\|^2.
$$

$D$ 仍是输出坐标数，除以它就得到每个坐标的平均平方误差。$\omega$ 用于选择时刻和控制更新，计算梯度时把它固定。负输出梯度为

$$
-\nabla_{v_\theta}\ell_{\mathrm{ART}}
=\frac{2\omega(s)}{D}\bigl(G(c)-d(s)\bigr).
$$

从 $v_\theta=v_{\mathrm{old}}$ 开始时，$d=0$，每个状态都收到同方向的 $G(c)$。构造中没有出现 $u_{\mathrm{old}}-v_{\mathrm{old}}$，因此也没有 NFT 的 consistency 项。同一提示词内的输出目标先共享一个方向，随后再通过各状态的网络导数转成参数梯度。

共同方向没有使用每个状态的后验信息；同一次参数更新也会让不同状态移动不同的距离。固定目标使这些移动可以被下一次回归观察到：当前预测已经移动了 $d(s)$，剩下的误差就变成 $G(c)-d(s)$。原始引导 $G(c)$ 继续存在，同时出现指向旧预测的 $-d(s)$。

ART 把一轮采样数据分成两个互不重叠的批次，各做一次参数更新。两次都使用采样时冻结的目标。第一批更新前没有位移；第一批更新之后，模型参数已经改变，第二批状态的预测也随之改变。于是第二次更新同时包含引导与恢复，而不需要维护一份跨轮次滞后的 EMA 模型。[ART，§4.2](https://arxiv.org/html/2610.12572v1#S4.SS2)。

在单个输出坐标上，取 $G=1$ 就能看清恢复的作用。如果第一次更新后位移是 $d=0.3$，剩下的方向为 $0.7$，仍然向前；如果位移是 $d=1.4$，剩下的方向变成 $-0.4$，会向回拉。恢复力度由当前预测偏离采样预测的距离直接决定。

## 只屏蔽越界后继续向外的更新

恢复项让平方误差的方向随位移改变，但神经网络的参数由许多状态共同更新。有些状态仍可能移动过多。ART 用一个 trust-region mask 检查当前位置是否已经偏离旧预测太远，以及当前输出方向是否还在把它推得更远。

衡量两份转移分布的变化，可以使用 KL 散度，即新分布下对数密度比的期望。确定性 ODE 的一步转移集中在一个点上，两个不同的点质量之间没有有限的 KL 散度可比。ART 借用随机转移的形式作为度量：在同一状态 $s$，给新旧预测各配一个高斯参考转移，并让二者使用相同的噪声方差，

$$
\mathcal T_\theta(\cdot\mid s)
=\mathcal N(a_\theta(s),\sigma_{\mathrm{ref}}^2(t)I),\qquad
\mathcal T_{\mathrm{old}}(\cdot\mid s)
=\mathcal N(a_{\mathrm{old}}(s),\sigma_{\mathrm{ref}}^2(t)I).
$$

$a_\theta$ 与 $a_{\mathrm{old}}$ 是参考下一状态均值，$\sigma_{\mathrm{ref}}^2$ 是共同的参考方差。论文沿用 Flow-GRPO 的参考转移构造。把一步的时间增量记为 $\Delta t>0$，参考噪声系数记为 $\nu>0$，在 $0<t<1$ 的状态 $s=(x,t,c)$ 上，这个构造写成

$$
a_\theta(s)
=x+\Delta t\left[
\left(1+\frac{\nu^2}{2}\right)v_\theta(s)
-\frac{\nu^2}{2t}x
\right],\qquad
\sigma_{\mathrm{ref}}^2(t)
=\nu^2\frac{1-t}{t}\Delta t.
$$

旧参考均值使用相同的 $x,t,\Delta t,\nu$，只把速度替换成 $v_{\mathrm{old}}$。所以，两份均值相减时共同项抵消：

$$
a_\theta(s)-a_{\mathrm{old}}(s)
=\Delta t\left(1+\frac{\nu^2}{2}\right)d(s).
$$

参考均值差正好是速度位移的正数倍。在噪声端 $t=0$，论文另取 $a_\theta=x+\Delta t\,v_\theta$、$\sigma_{\mathrm{ref}}^2=\nu^2$；均值差仍是速度位移的正数倍。实际采样可以继续沿 ODE 进行，这两个高斯用来量化预测变化。

因为两个高斯方差相同，计算对数密度比时，归一化常数抵消，只剩两份平方距离之差。把参考转移中的下一状态随机变量记为 $\Xi$，在新高斯下取期望时，随机波动也抵消，得到

$$
\begin{aligned}
\kappa(s)
&:=\frac1D D_{\mathrm{KL}}
(\mathcal T_\theta(\cdot\mid s)\|\mathcal T_{\mathrm{old}}(\cdot\mid s))\\
&=\frac{\mathbb E_{\Xi\sim\mathcal T_\theta}
[\|\Xi-a_{\mathrm{old}}\|^2-\|\Xi-a_\theta\|^2]}
{2D\sigma_{\mathrm{ref}}^2(t)}\\
&=\frac{\|a_\theta(s)-a_{\mathrm{old}}(s)\|^2}
{2D\sigma_{\mathrm{ref}}^2(t)}.
\end{aligned}
$$

把均值差的表达式代进去，在同一内点状态上就有

$$
\kappa(s)
=\frac{(\Delta t)^2(1+\nu^2/2)^2}
{2D\sigma_{\mathrm{ref}}^2(t)}\|d(s)\|^2.
$$

系数与当前网络参数无关，而且为正。因此，增大该状态的速度位移平方，也就在增大参考转移的 KL。这个 $\kappa$ 度量的是局部参考转移；它没有计算整条轨迹或最终图片分布的 KL。[ART，§4.3 与附录 C.2](https://arxiv.org/html/2610.12572v1#A3.SS2)。

给定阈值 $\delta>0$，$\kappa\geq\delta$ 表示局部变化已超过阈值。还需要判断剩余回归方向是在向外还是向内。先把一个状态的输出看成可独立移动的向量，沿 $G-d$ 走一小步 $\lambda>0$，位移平方的变化为

$$
\|d+\lambda(G-d)\|^2-\|d\|^2
=2\lambda\langle d,G-d\rangle
+\lambda^2\|G-d\|^2.
$$

内积 $\langle d,G-d\rangle>0$ 时，一阶变化朝外；小于零时，一阶变化朝内。这个输出空间的几何判断用来构造权重。ART 只在“已经越界”与“还在向外”同时成立时屏蔽该状态：

$$
\omega(s)
=(1-t)\left[
1-\mathbf1\{\kappa(s)\geq\delta\}
\mathbf1\{\langle d(s),G(c)-d(s)\rangle>0\}
\right].
$$

越界但正在向内的修正仍被保留。前面的 $1-t$ 是论文采用的高噪声时刻加权；时间在这里从噪声端的 $0$ 走向样本端的 $1$，所以越靠近噪声端，权重越大。每次更新前重新计算掩码，反向传播时把它当常数。

## 更新之后，再用生成结果回答奖励问题

一轮 NFT 训练用旧模型生成终点并评分，重新加噪做正负回归，再更新采样模型。一轮 ART 训练还缓存轨迹状态和预测，按提示词汇总奖励方向，固定回归目标，做两次带恢复与掩码的更新。两种方法都把难以直接求导的终点奖励接到了可计算的局部网络回归上。

局部回归误差可以告诉我们预测离目标还有多远。原始目标 $J(\theta\mid c)$ 则要通过新模型生成的样本来估计：固定评价用的提示词集合，重新生成并计算奖励，比较更新前后的均值。前面的推导分别说明了重加权分布的奖励、回归最优场的方向，以及实际更新里包含的项；这些项通过共享参数和采样器作用到终点，效果落在新生成的样本上。

计算成本也要回到同一个奖励目标上比较。ART 在 HunyuanVideo-13B 的文生视频实验中，对视觉质量与运动质量的原始评分取平均，报告达到 NFT 后期平均训练奖励所需的 GPU 小时数约为 NFT 的 $1/12.02$。这个数衡量达到相同奖励水平的成本。[ART，附录 A.5 与 Figure 1](https://arxiv.org/html/2610.12572v1#A1.SS5)。

每轮结束后，模型会产生一批新的终点，奖励也会重新分布。回归目标随这批新经验重建，采样、评分和更新便形成了在线学习过程。它持续追求的仍是开头那个量：从当前生成分布中抽到一张样本时，平均能得到多少奖励。

## 出处

- Zheng et al., *DiffusionNFT: Online Diffusion Reinforcement with Forward Process*, ICLR 2026. [arXiv:2509.16117v2](https://arxiv.org/abs/2509.16117v2)。正负分布与目标见 §3，后验混合与策略优化推导见附录 A。
- Wan et al., *DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models*, 2026. [arXiv:2610.12572v1](https://arxiv.org/abs/2610.12572v1)。更新分解见 §3 与附录 B.1–B.2，端点协方差见附录 B.3，参考转移与成对比较见附录 C.2–C.3。
