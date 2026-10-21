# Calibration examples

Dated incidents that calibrated the behavioral rules and skills in this repository. Evidence, not authority: the rule or skill text is the authority; these entries record what happened and the lesson drawn, so a future revision can check the rule against the incident that produced it.

## 2026-10-08 — grid-laziness (calibrates: experiment-lifecycle)

Context: designing the posttrain experiment campaign. The design drifted toward enumerating toggle/config combinations (eight combinators) without naming what each cell was for.

User: "We need experiments to answer our certain questions, not to pursue complete cleanliness. We design experiments to maximize insight per experiment. If we have a strong prior (like what an established paper already reveals), I don't think it's worth many experiments to do the exact full-grid ablation again." Sharpening further: the bad pattern is not reasoning about "what do we want to understand" but listing toggles and configs into a grid; in analysis, reading only the final number, making a cheap comparison, and announcing a winner — instead of reading the entire curve and the internal metrics, verifying one's hypothesis, and proactively collecting the insights the experiment should have produced; and concluding "what it does not say" because that is always safe — an experiment that says nothing yielded zero insight.

Lesson: every arm names its question and the decision it changes; analysis reads trajectories and internals, not headline numbers; a run yielding no insight is a failed run.

## 2026-10-09 — arm-minimization (calibrates: experiment-lifecycle)

Context: same campaign, redesigned after the grid-laziness rule landed. The agent over-corrected into avoiding experiments: it offered to "accept ratio 5 purely on prior, skip Phase 2 entirely" although DMD2's own text says to tune the update ratio per model/dataset (a non-portable prior), and proposed dissolving the DMD2-components question "into two audit checks rather than a funded training arm."

User: "For unknown, open problems, we should still actively fund experiments to find out a near-optimal recipe. For example, networks have an optimal height-width ratio, so it's justified to do a maybe 5x5 grid to fit a best ratio for a custom network; if 'how to configure a parameter' has no pinned prior, it's not justified to just come up with a random number and skip experiments. Experiments serve two purposes: harvesting insights AND finding recipe, where the former actually serves the latter. When we already understand the mechanism to a maximum degree, we might still need more runs to produce the final recipe, even though they produce fewer insight."

Lesson: insight and recipe are the two ends, and insight serves recipe; a parameter with no pinned prior for our setting is open and gets a funded sweep with predeclared points and fit criterion; mechanism fully understood does not imply recipe found.


## 2026-10-09 — imaginary audience (calibrates: repo-hygiene)

Context: post-implementation cleanup before the posttrain experiment campaign. The implementation had flooded configs and code with identity pins (sha256/fingerprint/revision on checkpoints, scorers, sampling configs, checkpoint manifests), version markers where exactly one version ever existed (`schema: 1`, `image-text-v1`, `inko-resident-v1`), hard-fail environment checks (a ten-package version table, runtime-file hashes), annotation fields whose only consumer was their own validator (`meaning = "Diagnostic log standard deviation; not a reward"`), receipts the program wrote to itself and then re-authenticated (campaign/export/complete markers carrying their own hashes), and proliferation past any live consumer (twelve prompt configs, four top-level training configs, budget and campaign control machinery). Roughly 8,500 lines were removed in the stale/dead pass and roughly 800 more in the pattern purge; the test suite stayed green throughout.

User, on the pattern: "Imagine you want a hamburger, codex served you hamburger plus french fries, you said no french fries, and codex gave you hamburger, with 'hamburger (no french fries version)' printed on the bread. Except that in this case, I didn't even said 'no french fries' at all, it printed 'not something' anyway."

User, widening the root beyond configs: in order to please the imaginary audience, the agent writes spaghetti code and pollutes docs with defensive prose and slop with zero information density. The user supplied this artifact as the typical style — a single lunch written up as a scientific observation:

```
本文考察关于“前往某餐饮场所食用麻辣烫”的单次、非随机、非盲法、自然情境观察。
需要首先强调，本次进食行为不应被解释为对麻辣烫相较火锅、米线或其他午餐方案具有普遍优越性的主张，也不构成对任何具体餐饮品牌或配菜组合的推荐。
研究对象，即本人，于约中午12时产生进食意图，随后步行约八分钟抵达店内；该过程包含等红灯、短暂查看手机以及在门口犹豫是否改吃黄焖鸡等若干与核心进食结论无直接关系、但为保证过程完整性仍予以报告的中间事件。进入店内后，我依次选取了油麦菜、鱼丸、土豆片、金针菇、宽粉及若干丸状加工食品；其中一次夹取海带后又将其放回，因此海带并未进入最终样本，但这一失败的配菜选择不应被隐去。
称重后总价为27.6元。需要指出，27.6元仅反映该店、该时段、该具体配菜结构下的观测结果，不能外推至其他门店、城市、季节、通货膨胀环境或不同个体的食量。付款后约六分钟，麻辣烫被送达。第一口汤的主观温度较高，并伴随明显辣感；然而，由于未使用经校准的温度计或辣度测量设备，该判断主要基于个人口腔感知，因此不能排除饥饿程度、前一晚睡眠以及当天情绪状态等潜在混杂因素。随后约十五分钟内，我基本完成了进食，其中宽粉表现出较高的主观满意度，土豆片亦未观察到明显缺陷。尽管如此，这些结果不应被过度解读为宽粉或土豆片具有稳定、跨个体、跨汤底的优势，因为本次实验仅包含一名受试者、一次进食、一个门店和一个辣度设置，且未设置清汤、微辣、中辣及特辣对照组。
值得注意的是，我在进食过程中饮用了约半杯可乐，这可能显著改变辣感耐受，并因此削弱对汤底本身的独立评价。进食结束后，我主观上认为“挺爽”“吃饱了”，并观察到短期满足感上升；但由于缺乏餐后两小时、四小时及次日的纵向随访数据，目前尚不足以确认该满足感具有持续性，也不能排除其主要来源于热量摄入而非麻辣烫这一具体食物形式。
另有一个需要高调说明的局限是，碗底仍残留少量汤汁及两片我不想继续食用的香菜，因此本次进食并未实现100%的内容物清除，这表明当前策略在“吃完麻辣烫”这一任务上的完成度仍存在改进空间。
总体而言，本次观察支持这样一种相对克制的表述：在当前个体、当前时间、当前门店及当前配菜条件下，前往食用麻辣烫与一次总体可接受且具有一定即时满足感的午餐体验相一致；然而，现有证据尚不足以证明“我去吃了麻辣烫”这一行为在更广泛饮食决策空间中具有稳定优势。
```

Lesson: every hated line had an imaginary beneficiary — an auditor (byte pins), a future second version (markers), a careless operator (environment checks), a confused reader (annotation fields and negation labels), a hostile reviewer (hedge prose), an imagined future requirement (proliferation), a future self (kept one-offs). Two orthogonal gates emerged: structure must name a consumer that exists today; content must change what the reader may conclude or do. A lean repo can still hold slop, and honest prose can sit inside a bloated repo — each gate catches what the other passes. Hedging earns its sentence only when it negates a conclusion the reader would actually have drawn; the apparatus of caution without a live over-read is slop.
