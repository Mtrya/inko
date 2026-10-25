# Flow Matching：速度场从哪里来

## 从加噪分布构造生成过程

一张干净的照片，加上一点噪声，会变得模糊；再加一点，细节继续消失；加到足够多次，它就成了一片雪花。扩散模型做的事情，是把这个过程反过来：从一片雪花出发，一点一点把结构找回来。

为了让生成沿着时间增加的方向进行，把纯噪声记在 $t=0$，把干净图片记在 $t=1$。加噪从 $1$ 走向 $0$，生成则从 $0$ 走向 $1$。对时间求导得到的速度，也沿着这个生成方向定义。

具体写法是 $x_t=\alpha_t x_1+\sigma_t x_0$。$x_1$ 从数据分布 $p_{\text{data}}$ 抽取，$x_0\sim\mathcal N(0,I)$ 是与它独立的标准高斯噪声；$I$ 是单位阵。$\alpha_t$ 和 $\sigma_t$ 控制两者的比例。$t=1$ 时 $\alpha_1=1$、$\sigma_1=0$，只剩图片；$t=0$ 时 $\alpha_0=0$、$\sigma_0=1$，只剩噪声。中间的时刻图片和噪声混在一起。

常见的选法是 $\alpha_t^2+\sigma_t^2=1$。独立性给出 $\operatorname{Cov}(x_t)=\alpha_t^2\operatorname{Cov}(x_1)+\sigma_t^2I$，所以当数据各坐标已经标准化到单位方差时，每个坐标的方差沿途保持为 $1$。$\alpha_t$ 负责保留信号，$\sigma_t$ 负责注入噪声。

取可微的系数，并让 $\alpha_t,\sigma_t$ 在 $0<t<1$ 内为正。给定 $x_1$，$x_t$ 的条件密度可直接计算：$p_t(x\mid x_1)=\mathcal N(\alpha_t x_1,\sigma_t^2 I)$。边缘密度 $p_t(x)=\int p_t(x\mid x_1)p_{\text{data}}(x_1)\,dx_1$ 则混合了所有可能的干净图片，一般无法直接算出这个积分。能够抽取带噪样本，不等于知道它的密度值。

## 用 score 表示生成速度

绕开的办法是学 score。score 是密度的对数对位置的梯度：$s_t(x)=\nabla_x\log p_t(x)$。它指向密度上升最快的方向。对一维高斯 $\mathcal N(\mu,\tau^2)$，score 是 $-(x-\mu)/\tau^2$，其中 $\mu$ 和 $\tau^2$ 分别是均值与方差。点在均值左边，score 为正，指向右边；点在均值右边，score 为负，指向左边。离均值同样远时，方差越小，score 的模长越大。以下的 $\nabla$ 都对位置 $x$ 求导。

在扩散里，$p_t$ 是加噪到时刻 $t$ 的密度，所以 score 指向当前噪声层级下更常见的样本。它包含了局部密度变化的信息，还需要配上随时间变化的系数，才能成为生成速度。

采样可以用一个确定性的常微分方程（ODE）来描述。把采样轨迹记为 $y_t$，让它满足 $dy_t/dt=u_t(y_t)$，其中 $u_t(x)$ 表示时刻 $t$、位置 $x$ 的速度。从 $y_0\sim\mathcal N(0,I)$ 出发，希望每个时刻的 $y_t$ 都具有密度 $p_t$，这样终点就服从数据分布。实现这一点的 ODE 称为概率流 ODE。质量沿着速度场搬运，密度增加来自净流入，因此必须满足连续性方程

$$
\partial_t p_t+\nabla\cdot(p_t u_t)=0.
$$

这里 $\nabla\cdot(p_tu_t)$ 是概率流量的散度，表示局部的净流出。在多维空间里，给定密度路径通常不足以唯一确定速度：若另一个场 $w_t$ 满足 $\nabla\cdot(p_tw_t)=0$ 和相应的边界条件，把它加到 $u_t$ 上，密度演化仍然相同。仿射加噪的形式提供了一种具体构造。

给定一对端点，插值曲线的速度是 $\dot x_t=\dot\alpha_t x_1+\dot\sigma_t x_0$，上面的点表示对 $t$ 求导。不同配对可能经过同一位置，把它们的速度按后验权重平均，定义 $u_t(x)=\mathbb E[\dot x_t\mid x_t=x]$。要算这个平均，可以先对边缘密度的积分表达式求梯度：

$$
\nabla\log p_t(x)
=\frac{\int -\frac{x-\alpha_t x_1}{\sigma_t^2}\,
p_t(x\mid x_1)p_{\text{data}}(x_1)\,dx_1}{p_t(x)}
=-\frac1{\sigma_t}\mathbb E[x_0\mid x_t=x].
$$

因此，在 $0<t<1$ 内有

$$
\mathbb E[x_0\mid x_t=x]=-\sigma_t\nabla\log p_t(x),
\qquad
\mathbb E[x_1\mid x_t=x]=\frac{x+\sigma_t^2\nabla\log p_t(x)}{\alpha_t}.
$$

第二个等式由 $x_t=\alpha_t x_1+\sigma_t x_0$ 取条件期望，再代入第一个等式得到。把这两个等式代入平均速度，整理得

$$
u_t(x)=a(t)x+b(t)\nabla\log p_t(x),
$$

其中 $a(t)=\dot\alpha_t/\alpha_t$，$b(t)=\dot\alpha_t\sigma_t^2/\alpha_t-\dot\sigma_t\sigma_t$。这两个系数只由加噪路径决定。两端有系数为零，这个含除法的表示需要取极限，不能直接代入。

还要验证平均速度确实推动了 $p_t$。取任意光滑且只在有限区域内非零的测试函数 $\varphi$，利用链式法则与条件期望，

$$
\frac{d}{dt}\mathbb E[\varphi(x_t)]
=\mathbb E[\nabla\varphi(x_t)\cdot\dot x_t]
=\mathbb E[\nabla\varphi(x_t)\cdot u_t(x_t)]
=-\int\varphi(x)\nabla\cdot(p_tu_t)(x)\,dx.
$$

最后一步把期望写成积分，再分部积分；$\varphi$ 的支撑使边界项为零。左边也等于 $\int\varphi(x)\partial_t p_t(x)\,dx$。等式对所有这样的 $\varphi$ 成立，就得到连续性方程。

ODE 轨迹的密度也满足这条方程，因为 $dy_t/dt=u_t(y_t)$，对 $\varphi(y_t)$ 做同样计算即可。两条密度路径从同一个 $p_0=\mathcal N(0,I)$ 出发；当速度场足够光滑，使输运方程的解唯一时，ODE 密度就等于插值密度 $p_t$。因此，$y_1$ 服从数据分布。生成轨迹不必逐条重现端点插值，但它们在每个时刻具有同一分布。

系数来自加噪路径，score 是这个速度表达式里唯一需要学的量。把 score 学出来，就能从噪声出发，沿着 ODE 走到数据。

## 用条件 score 训练

剩下的一块是 score。$p_t$ 无法直接计算，$\nabla\log p_t$ 也不能直接拿来当训练标签。但给定一张干净图片，条件 score 可以计算，能够用来构造回归目标。

给定 $x_1$，条件分布是高斯的，条件 score 有闭式：

$$
\nabla\log p_t(x\mid x_1)=-\frac{x-\alpha_t x_1}{\sigma_t^2}.
$$

这个式子只依赖 $x$、$x_1$ 和系数，完全可以算。把它对所有可能的 $x_1$ 按后验权重平均，就得到边缘 score：

$$
\mathbb E\big[\nabla\log p_t(x_t\mid x_1)\mid x_t=x\big]=\nabla\log p_t(x).
$$

由贝叶斯公式，$p_t(x\mid x_1)p_{\text{data}}(x_1)=p_t(x)p(x_1\mid x_t=x)$。两边取对数再对 $x$ 求梯度，左边是 $\nabla\log p_t(x\mid x_1)$，右边第一项是 $\nabla\log p_t(x)$，第二项是 $\nabla\log p(x_1\mid x_t=x)$。对 $x_1$ 取后验期望时，第二项的期望是

$$
\int p(x_1\mid x)\nabla\log p(x_1\mid x)\,dx_1
=\int\nabla p(x_1\mid x)\,dx_1=0.
$$

这里用 $p(x_1\mid x)$ 简写 $p(x_1\mid x_t=x)$。积分变量是 $x_1$，求导变量是条件 $x$。条件密度对 $x_1$ 的积分恒为 $1$，因此其对 $x$ 的梯度为零。

固定一个 $0<t<1$，用参数为 $\theta$ 的网络 $s_\theta(x_t,t)$ 预测 score，训练目标可以写成

$$
\mathcal L_{\mathrm{score}}(\theta;t)
=\mathbb E_{x_1\sim p_{\text{data}},\ x_0\sim\mathcal N(0,I)}
\big\|s_\theta(x_t,t)-\nabla\log p_t(x_t\mid x_1)\big\|^2.
$$

网络只看到带噪图片与时间，训练标签却还依赖生成它的干净图片。平方误差会把同一输入对应的不同标签连接到一个平均预测。把条件 score 标签暂记为 $q$，已知 $\mathbb E[q\mid x_t]=s_t(x_t)$。在预测误差中加减 $s_t$，再展开平方：

$$
\mathbb E\|s_\theta-q\|^2
=\mathbb E\|s_\theta-s_t\|^2+\mathbb E\|s_t-q\|^2.
$$

交叉项为零，因为给定网络输入 $x_t$ 后，$s_t-q$ 的条件均值为零。右边第二项不依赖网络。因此，在可以自由选择预测函数、二阶矩有限时，最优预测就是边缘 score。每个条件标签指向 $\alpha_t x_1$；网络学到的是这些向量按后验加权的平均。

因为 $x_t=\alpha_t x_1+\sigma_t x_0$，条件 score 等于 $-x_0/\sigma_t$。把网络写成 $s_\theta=-\epsilon_\theta/\sigma_t$，损失就等于 $\mathbb E\|\epsilon_\theta-x_0\|^2/\sigma_t^2$。这说明 score 预测与噪声预测可以互换参数化，但损失权重也要跟着换。

训练覆盖多个时刻时，可以最小化 $\mathbb E_t[\lambda(t)\mathcal L_{\mathrm{score}}(\theta;t)]$，其中 $\lambda(t)>0$ 是时间权重。若直接用均匀时间和单位权重，$1/\sigma_t^2$ 可能使零噪声端附近的积分发散。取 $\lambda(t)=\sigma_t^2$ 则得到普通的噪声预测均方误差。正权重不改变每个固定时刻的理想最优预测，却会改变各时刻在训练中的相对分量。

训练完之后，用 $s_\theta$ 替代真实 score，得到近似速度 $v_\theta(x,t)=a(t)x+b(t)s_\theta(x,t)$。从高斯噪声出发，数值求解这份速度定义的 ODE。Euler 方法每步求一次速度，高阶方法可能求多次。减小步长是在提高对已学 ODE 的数值精度；网络对真实速度的拟合误差仍然存在。

生成成本来自反复调用网络。沿一条采样轨迹，速度的方向或大小都可能变化，数值求解器需要用足够小的步长跟上这些变化。选择怎样的中间路径，会影响生成时需要多少次网络求值。

高斯加噪给了可计算的条件 score，也给了把 score 转成速度的系数。但前面的连续性方程证明只使用了 $\dot x_t$ 及其条件期望。flow matching 从这个构造继续：选择可微的样本插值，直接用插值速度做回归，让生成沿着它诱导的密度路径演化。

## 把路径和速度分开

扩散里的 $u_t$ 是由 score 拼出来的。现在把 score 拿掉，只留下两个对象：一条密度路径 $p_t$，一个速度场 $u_t$。给定 $p_t$，什么样的 $u_t$ 才算把它从噪声搬到数据？

一条密度路径，是一族随 $t$ 变化的密度 $p_t$。$t=0$ 是噪声，$p_0=\mathcal N(0,I)$；$t=1$ 是数据，$p_1$。生成就是把质量从 $p_0$ 搬到 $p_1$。

一个速度场，是一个随位置和时刻变化的向量 $u_t(x)$。它说：在时刻 $t$，位于 $x$ 的质量该往哪走、走多快。

两者要匹配，速度必须让密度按 $p_t$ 演化，也就是满足连续性方程

$$
\partial_t p_t+\nabla\cdot(p_t u_t)=0.
$$

它说密度的变化正好被净流出抵消。在多维空间里，同一密度路径通常允许不止一个速度场。选择样本怎样插值，能够同时给出一条密度路径和一个与它匹配的速度。

最直接的选择发生在样本层面。取 $x_0\sim p_0$ 和 $x_1\sim p_1$，先让它们独立。让点沿一条确定的曲线从 $x_0$ 走到 $x_1$，曲线取仿射形式

$$
x_t=\alpha_t x_1+\sigma_t x_0,\qquad
\alpha_0=0,\ \sigma_0=1,\ \alpha_1=1,\ \sigma_1=0.
$$

这条给定端点的曲线叫条件路径。它诱导出每个时刻的密度 $p_t=\operatorname{Law}(x_t)$，其中 $\operatorname{Law}$ 表示随机变量的分布；中间密度由插值决定。沿曲线走，$x_t$ 的变化率是

$$
\dot x_t=\dot\alpha_t x_1+\dot\sigma_t x_0.
$$

这个量叫条件速度，它以端点为条件。玩具例子用线性插值 $\alpha_t=t$、$\sigma_t=1-t$，此时 $\dot x_t=x_1-x_0$。

不同的配对会经过同一个时空点。把经过同一点的所有条件速度平均，得到一个只依赖 $(x,t)$ 的速度：

$$
u_t(x)=\mathbb E[\dot x_t\mid x_t=x].
$$

这个平均叫边缘速度。在时刻 $t$、位置 $x$，它汇总所有经过这里的配对及其速度。直接计算这个平均，需要知道给定 $x_t=x$ 后各个端点配对的后验权重。

这个定义仍能使用连续性方程的证明。取任意光滑且具有紧支撑的函数 $\varphi$，即只在有限区域内非零，看它的期望怎样随时间变化：

$$
\frac{d}{dt}\mathbb E[\varphi(x_t)]
=\mathbb E[\nabla\varphi(x_t)\cdot\dot x_t]
=\mathbb E[\nabla\varphi(x_t)\cdot u_t(x_t)]
=\int p_t\,\nabla\varphi\cdot u_t\,dx
=-\int \varphi\,\nabla\cdot(p_tu_t)\,dx.
$$

第一个等号是链式法则；第二个等号把条件速度换成它对 $x_t$ 的条件期望；第三个等号把期望写成对 $p_t$ 的积分；第四个等号是分部积分。左边同时等于 $\int\varphi\,\partial_t p_t\,dx$，所以对任意 $\varphi$ 都有 $\int\varphi(\partial_t p_t+\nabla\cdot(p_tu_t))\,dx=0$，连续性方程成立。

从 $p_0$ 出发、沿 $u_t$ 生成的密度满足同一方程和初值。在输运方程解唯一的条件下，它与插值密度 $p_t$ 相同，因而到达 $p_1$。这套论证没有用到端点服从高斯，也没有用到 score；只要插值可微、相关期望存在，就能先从配对样本定义速度，再检查相应的正则性。

先用一个玩具例子把这件事具体化。一维双峰：$p_1$ 是两个方差为 $s^2=0.1$、均值分别为 $-1$ 和 $+1$ 的高斯各取一半，$p_0=\mathcal N(0,1)$。用线性插值。给定峰 $\mu\in\{-1,+1\}$，$x_t=(1-t)x_0+t x_1$ 是两个独立高斯的线性组合，所以它也是高斯，均值为 $t\mu$，方差为

$$
V_t=(1-t)^2+t^2s^2.
$$

两个峰各给一半权重，于是

$$
p_t=\tfrac12\mathcal N(t,V_t)+\tfrac12\mathcal N(-t,V_t).
$$

$t=0$ 时两个峰重合在 $0$，方差为 $1$；随着 $t$ 增大，两个峰分别向 $\pm1$ 移动。一个团裂成两个。后面会反复用这个例子，因为它的所有量都能手算。

## 条件速度就够了

现在有一个定义清楚、但通常无法直接计算的速度：$u_t(x)$ 是端点配对的条件平均。训练时我们能抽配对样本，却不知道给定 $x_t=x$ 后的完整后验。可算的是每对样本的条件速度 $\dot x_t$。score 回归中使用过的平方分解，在这里也能把条件目标接到边缘目标上。

固定 $t$，假定相关二阶矩有限。对任意预测函数 $v(x,t)$，令 $U=\dot x_t$，$\bar U(x)=\mathbb E[U\mid x_t=x]$。把 $v(x_t,t)-U$ 拆成 $(v-\bar U)+(\bar U-U)$，展开平方：

$$
\mathbb E\|v-U\|^2
=\mathbb E\|v-\bar U\|^2
+2\,\mathbb E\langle v-\bar U,\ \bar U-U\rangle
+\mathbb E\|\bar U-U\|^2.
$$

交叉项对 $x_t$ 取条件期望后为零，因为内层是 $\langle v-\bar U,\ \bar U-\mathbb E[U\mid x_t]\rangle$，而 $\bar U(x_t)=\mathbb E[U\mid x_t]$。所以

$$
\mathbb E\|v-U\|^2
=\mathbb E\|v-\bar U\|^2+\mathbb E\|\bar U-U\|^2.
$$

右边第二项与 $v$ 无关。因此，最小化左边等价于最小化第一项；可以自由选择预测函数时，最小值点在 $v=\bar U$ 处取到。以条件速度为目标做回归，理想最优预测就是边缘速度。

损失可以写成

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t\sim\operatorname{Unif}[0,1]}\ \mathbb E_{x_0,x_1}
\big\|v_\theta(x_t,t)-\dot x_t\big\|^2.
$$

$t$ 在 $[0,1]$ 上均匀取，$x_0$ 和 $x_1$ 独立地从 $p_0$ 和 $p_1$ 抽取。计算插值就直接得到了 $p_t$ 的样本，目标速度也由这对端点直接给出。训练不用为取得这些状态而求解 ODE，也不需要知道 score。这就是“免模拟”（simulation-free）的含义。

这套做法就是 flow matching：不经过 score，直接用配对样本定义并回归速度场。路径可以是任意仿射形式；训练目标始终是条件速度。

拿前面的双峰例子算一遍，会看到这个恒等式具体长什么样。用线性插值，直接算 $\mathbb E[x_1-x_0\mid x_t=x]$。给定峰 $\mu$，$x_0$ 和 $x_t$ 的联合分布是高斯。对联合高斯的两个分量 $a$ 和 $b$，有

$$
\mathbb E[a\mid b=y]=\mathbb E[a]+\frac{\operatorname{Cov}(a,b)}{\operatorname{Var}(b)}(y-\mathbb E[b]).
$$

代入 $a=x_0$、$b=x_t$：$\operatorname{Cov}(x_0,x_t)=1-t$，$\operatorname{Var}(x_t)=V_t$，$\mathbb E[x_t]=t\mu$，所以

$$
\mathbb E[x_0\mid x_t=x,\mu]=\frac{(1-t)(x-t\mu)}{V_t}.
$$

同理，$\operatorname{Cov}(x_1,x_t)=ts^2$，所以

$$
\mathbb E[x_1\mid x_t=x,\mu]=\mu+\frac{ts^2(x-t\mu)}{V_t}.
$$

两式相减，得到给定峰时的条件速度：

$$
\mathbb E[x_1-x_0\mid x_t=x,\mu]
=\mu+c_t(x-t\mu),\qquad
c_t=\frac{ts^2-(1-t)}{V_t}.
$$

还剩两个峰之间的加权。权重是后验概率 $r_\pm(x)$，由贝叶斯公式，

$$
\frac{r_+(x)}{r_-(x)}
=\frac{\exp\!\big(-(x-t)^2/(2V_t)\big)}
{\exp\!\big(-(x+t)^2/(2V_t)\big)}
=\exp\!\big(2tx/V_t\big).
$$

归一化后 $r_+-r_-=\tanh(tx/V_t)$。把两个峰的条件速度按 $r_\pm$ 加权，得到

$$
u_t(x)=c_t x+(1-c_t t)\tanh\!\Big(\frac{tx}{V_t}\Big).
$$

两个端点给出 $u_0(x)=-x$、$u_1(x)=x$。$t=0$ 时，独立性给出 $\mathbb E[x_1\mid x_0=x]=0$，所以平均位移是 $0-x$，指向原点。$t=1$ 时，$\mathbb E[x_0\mid x_1=x]=0$，所以平均位移是 $x-0$，背离原点。这是插值分布到达终点时的速度；到达 $t=1$ 后采样就结束。

取 $t=0.5$，分别在 $x=-0.5,0,0.5$ 处代入公式，得到速度约为 $-0.492174,0,0.492174$。用端点后验 $p(x_1\mid x_t=x)\propto p_1(x_1)\mathcal N(x;tx_1,(1-t)^2)$ 直接积分条件速度，在 $x_1\in[-6,6]$ 上取间距 $0.00025$ 的网格，三个位置的结果与闭式之差均小于 $10^{-12}$。

平方分解说明回归的理想目标是边缘速度 $u_t$，连续性方程则说明精确的 $u_t$ 会把 $p_0$ 送到 $p_1$。实际生成使用网络近似 $v_\theta$ 和数值求解器，还包含函数拟合误差与积分误差。回归目标与生成分布之间的连接依靠前面两个独立的论证。

## 扩散路径对应哪一种速度回归

现在回到开头的方差保持加噪路径。它与线性插值使用同一个仿射形式，只是换了一组系数：$\alpha_t$ 从 $0$ 升到 $1$，$\sigma_t$ 从 $1$ 降到 $0$，并取 $\alpha_t^2+\sigma_t^2=1$。例如 $\alpha_t=\sin(\pi t/2)$、$\sigma_t=\cos(\pi t/2)$：$t=0$ 时得到纯噪声，$t=1$ 时得到数据。由于 $x_0$ 是独立高斯，这条路径与线性插值都具有高斯条件分布；区别在于均值和方差随时间怎样变化。

在这个路径上，条件速度不是 $x_1-x_0$，而是 $\dot x_t=\dot\alpha_t x_1+\dot\sigma_t x_0$。要把它写成 $x_t$ 和条件 score 的函数，前面已经做过这一步：给定 $x_1$，条件 score 是

$$
\nabla\log p_t(x\mid x_1)=-\frac{x-\alpha_t x_1}{\sigma_t^2}.
$$

由 $x_t$ 的定义可以解出 $x_0$ 和 $x_1$，把它们代入 $\dot x_t$，整理得

$$
\dot x_t=a(t)x_t+b(t)\nabla\log p_t(x_t\mid x_1),
$$

其中 $a(t)=\dot\alpha_t/\alpha_t$，$b(t)=\dot\alpha_t\sigma_t^2/\alpha_t-\dot\sigma_t\sigma_t$。再对 $x_1$ 取条件期望，条件 score 的期望是边缘 score，于是

$$
u_t(x)=a(t)x+b(t)\nabla\log p_t(x).
$$

这就是扩散采样时用的概率流速度。它不是新假设，它是前面的条件期望速度在这条高斯路径上的具体形式。这些式子对 $0<t<1$ 成立；两端 $\alpha_t$ 或 $\sigma_t$ 为零，需要取极限。

训练目标也要跟着变。这里把网络参数化成 score 的形式，$v_\theta=a(t)x+b(t)s_\theta(x,t)$，只是为了和扩散对上；flow matching 本身不需要这个参数化，$v_\theta$ 可以直接输出速度。两式相减，

$$
v_\theta-\dot x_t=b(t)\big(s_\theta-\nabla\log p_t(x_t\mid x_1)\big).
$$

所以前面的损失变成

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E\Big[b(t)^2\big\|s_\theta(x_t,t)-\nabla\log p_t(x_t\mid x_1)\big\|^2\Big].
$$

这正好是时间权重取 $b(t)^2$ 的条件 score 回归。再由平方分解，对每个固定的 $t$，它与同权重的边缘 score 目标只差一个与 $\theta$ 无关的常数。在 $b(t)\neq0$ 处，两者的理想最优预测相同。

因此，在这条路径上，以 score 参数化速度、并使用对应的时间权重，就能把 flow matching 损失写成去噪 score matching 损失。生成时，两种参数化给出同一个概率流 ODE。[Flow Matching 论文](https://arxiv.org/abs/2210.02747)把扩散路径纳入速度回归框架，靠的就是这样的连接。

高斯在这个连接中做了两件事：让条件 score 有闭式，并把速度写成 score 的线性函数。条件平均与连续性方程的论证不依赖这两件事。即使换成非高斯的端点分布，只要仍能抽样并计算插值速度，就还能构造相应的回归目标。

这样看，扩散把两个决定绑在了一起：走哪条路径，以及怎样估计速度。flow matching 把它们拆开。路径自由之后，估计器还是同一个条件期望恒等式；变化的是路径带来的成本。

## 路径自由之后

不同路径可以共用条件速度回归的原理，但具体目标 $\dot x_t$ 会随路径改变。$\alpha_t$ 和 $\sigma_t$ 因而成为可以选择的设计参数，它们同时影响训练目标与采样成本。

以 Euler 方法为例，一步更新是 $y_{t+h}=y_t+h\,v_\theta(y_t,t)$，$h$ 是时间步长。它用起点速度近似整个小区间的运动。如果沿轨迹的速度保持不变，一步就能精确走完整条匀速直线。速度的方向或大小变化越快，通常就需要更小的步长，才能达到同样的积分精度。

线性插值的条件路径是匀速直线，因为给定一对端点，速度 $x_1-x_0$ 不随 $t$ 变。沿边缘 ODE 从某个初值走出的曲线叫边缘轨迹。它不一定保持匀速直线运动，因为参与条件平均的配对及其权重会随位置和时刻变化。

双峰例子给出 $u_0(x)=-x$、$u_1(x)=x$。从 $y_0=1$ 出发，起点速度向左；场关于原点对称且足够光滑，$u_t(0)=0$，所以轨迹不会穿过恒为零的另一条 ODE 轨迹。终点仍为正，速度因而向右。它必须先向内，再折返向外，不能保持条件插值的匀速运动。

用步长 $1/4000$ 的四阶 Runge–Kutta 数值积分，轨迹在几个时刻的位置是：

| $t$ | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| $y_t$，从 $y_0=1$ 出发 | 1.000 | 0.796 | 0.761 | 0.914 | 1.150 |

轨迹先降后升，最小值约为 $0.7453$，出现在 $t=0.4135$。一维空间里所有位置本来就在同一条数轴上；这里变化的是位置随时间的关系，图像 $t\mapsto y_t$ 发生了转弯。直接用起点速度走一步 Euler，会得到 $1+u_0(1)=0$，而精细积分的终点约为 $1.1503$。条件路径是直线，并没有消除边缘速度在沿途的变化。

配对也是一个自由参数。前面默认 $x_0$ 和 $x_1$ 独立抽取。换一种配对规则，就换一条插值密度路径，也就换一个速度场。在两个端点分布固定的前提下，使 $\mathbb E\|x_1-x_0\|^2$ 最小的联合分布叫平方距离代价下的最优传输耦合。这个目标直接优化端点之间的配对距离。

也可以用已有生成轨迹的两端重新配对，这就是 reflow。它来自 rectified flow：线性插值加条件速度回归，是 rectified flow 的基本形式；reflow 是其中“用模型自己的输出重新配对、再训一次”的那一步。具体做法是：

1. 用独立配对训练一个速度场 $v_\theta$；
2. 从 $p_0$ 抽 $y_0$，解 ODE 得到 $y_1$，把 $(y_0,y_1)$ 作为新的端点配对；
3. 用新配对重训同一个损失；
4. 重复。

在精确的条件平均速度和精确积分下，新配对的两端仍然服从 $p_0,p_1$。它还有一个可直接证明的性质：平方传输代价不增加。因为 $y_1-y_0=\int_0^1u_t(y_t)\,dt$，

$$
\begin{aligned}
\mathbb E\|y_1-y_0\|^2
&\leq\int_0^1\mathbb E\|u_t(y_t)\|^2\,dt\\
&=\int_0^1\mathbb E\big\|\mathbb E[x_1-x_0\mid x_t]\big\|^2\,dt\\
&\leq\mathbb E\|x_1-x_0\|^2.
\end{aligned}
$$

两个不等号都来自平方范数的凸性：平均之后的平方，不超过平方之后的平均。中间的等号用到 $y_t$ 与 $x_t$ 同分布，以及边缘速度的定义。把平方范数换成其他凸函数，同样的推导仍成立。这是 [rectified flow 的传输代价定理](https://arxiv.org/html/2209.03003v1#S3.SS2)。

端点配对距离与沿途速度变化是不同的量，以上不等式直接保证的是前者。重新训练会让模型拟合新端点之间的匀速插值；[论文的图像实验](https://arxiv.org/html/2209.03003v1#S5.SS2)展示了 reflow 后用单步 Euler 生成的结果。实际模型存在拟合误差时，新配对的第二端点服从模型生成分布，不能继续把它当作精确的 $p_1$。

路径和配对决定了条件回归目标，条件平均决定了理想生成速度。它们怎样影响实际训练与采样成本，还要比较模型的拟合结果和数值积分误差。

## 出处

- Lipman et al., *Flow Matching for Generative Modeling*, 2022. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, 2022. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Albergo & Vanden-Eijnden, *Building Normalizing Flows with Stochastic Interpolants*, 2022. arXiv:2209.15571
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, 2021. arXiv:2011.13456
- Vincent, *A Connection Between Score Matching and Denoising Autoencoders*, 2011. *Neural Computation* 23(7):1661–1674
