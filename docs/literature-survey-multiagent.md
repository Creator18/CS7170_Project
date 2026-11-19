# Literature Survey: Multi-Agent Coordination and Intent Inference

**Written 2026-10-22** for the project in [project-summary.md](project-summary.md). Every entry below was checked against its paper page or abstract in this session; details beyond the abstract are marked where they came from secondary summaries.

**Our task, for reference:** predict a visible partner's **stated intent** from **muted first-person video of both people**, before they speak, using speech as free ground truth, and test which visible cues make it possible.

---

## 1. Computational theory of mind: inferring goals and beliefs from behaviour

| work | what it does | relation to us |
| --- | --- | --- |
| **Baker, Saxe, Tenenbaum. Action understanding as inverse planning.** *Cognition* 113(3), 2009 | Bayesian inverse planning: assume agents act roughly rationally toward goals given beliefs, and infer goals by inverting a planning model. Tested on animated agents in simple mazes | the founding formal account of intent inference. Our setting is the opposite end: real, unscripted video with no planner to invert |
| **Rabinowitz et al. Machine Theory of Mind (ToMnet).** ICML 2018 | a network meta-learns models of other agents from their behaviour alone; passes a Sally-Anne false-belief test in gridworlds | learned intent models from observation, as we propose; but gridworld agents, not people |
| **Jin, Shu et al. MMToM-QA.** ACL 2024 | 600 multiple-choice questions on goals and beliefs of one person searching for objects in 134 videos of a **simulated** household, with text descriptions; proposes BIP-ALM, inverse planning accelerated by language models | multiple-choice goal and belief inference from video, like our evaluation; but one simulated actor, no partner |
| **Li et al. EgoToM.** arXiv 2503.22152, 2025 (Meta) | multiple-choice video QA on Ego4D about the **camera wearer's** goals, beliefs and next actions, generated with a causal theory-of-mind model. Multimodal LLMs are near human level on goals, below it on beliefs and future actions | the closest egocentric benchmark; but it reasons about the wearer, not about a partner the wearer is working with |

## 2. Belief and intent in real human collaboration

| work | what it does | relation to us |
| --- | --- | --- |
| **Duan, Yu, Tan, Yi, Tan. BOSS.** arXiv 2206.10665, 2022 | two people collaborate on tasks **using only nonverbal communication**; the task is to predict both people's **beliefs for every frame**. 900 videos, 347,490 frames, 20 participants, egocentric and third-person, with pose, gaze, gesture and object signals | **the closest prior work**: dyadic, nonverbal, first-person, belief prediction. Differences: talking was forbidden, the tasks are short and object-centred, and it is small; we use natural unscripted collaboration, take labels from what people actually said, and test which cues carry intent |
| **Bortoletto et al. MToMnet.** 2024 (arXiv 2407.06762) | an explicit theory-of-mind network for belief prediction in nonverbal social interaction, evaluated on BOSS | a model baseline for nonverbal belief prediction |
| **Bara, CH-Wang, Chai. MindCraft.** EMNLP 2021, Outstanding Paper | human pairs in Minecraft's blocks world, with dialogue; participants periodically **self-report** their beliefs about the task and each other | situated theory of mind with real partners; but a virtual world, and beliefs are self-reported rather than read off speech in context |
| **Gavryushin, Zhang, Huang et al. CoMind.** arXiv 2607.06691, 2026 (ECCV 2026 per a team post) | paired people cooking together in 55 real kitchens; egocentric and exocentric video, audio, gaze, 3D scans; shared attention and social-cue labels; benchmarks for **joint attention, action anticipation and handover prediction** | our core dataset. It anticipates actions and handovers; we predict the **intent behind** an upcoming utterance, from muted video |
| **Wang et al. HoloAssist.** ICCV 2023 | a headset performer guided by a remote instructor, 166 h, 350 pairs; mistake and intervention labels; benchmark predicts the **intervention type** from the seconds before the instructor speaks | the same "predict before speech" design, but the instructor has no body in the scene |
| **Chavan et al. IndEgo.** NeurIPS 2025 D&B | industrial collaboration, both workers on Aria glasses, 197 h first-person total, mistakes and roles, collaborative task understanding benchmark | our transfer set; no cue labels |

## 3. Human-AI coordination and ad hoc teamwork

| work | what it does | relation to us |
| --- | --- | --- |
| **Carroll, Shah, Ho, Griffiths, Seshia, Abbeel, Dragan. On the Utility of Learning about Humans for Human-AI Coordination (Overcooked-AI).** NeurIPS 2019 | agents trained by self-play coordinate poorly with people; agents trained with a model of human play do better, in a simulated kitchen and a user study | motivates modelling real human behaviour rather than idealised partners; intent there is implicit in a learned human model, never read from video |
| **Hu, Lerer, Peysakhovich, Foerster. "Other-Play" for Zero-Shot Coordination.** ICML 2020 | self-play invents conventions that fail with new partners; other-play uses problem symmetries for strategies that work with unseen partners, tested in Hanabi and with humans | the coordination problem in its game-theoretic form: work with a partner you have not trained with |
| **Puig et al. Watch-And-Help.** ICLR 2021 | an agent watches one demonstration, infers the goal, then helps in a new environment (VirtualHome-Social), evaluated with simulated and real humans | the closest "infer intent, then help" formulation; goals are known by construction in simulation |
| **PARTNR** (Meta FAIR, with Habitat 3.0) | about 100,000 natural-language household tasks across 60 simulated houses for human-robot coordination; LLM planners solve about 30% against about 93% for people, struggling with coordination, progress tracking and error recovery (project-page figures) | shows coordination and progress tracking are open problems even in simulation |

## 4. Intent from nonverbal cues in human-robot collaboration

| work | what it does | relation to us |
| --- | --- | --- |
| **Huang, Mutlu. Anticipatory Robot Control for Efficient Human-Robot Collaboration.** HRI 2016 | a robot infers which item a person will request from their **gaze** and acts before being asked; it finished tasks about 2.5 s faster on average than a reactive robot (abstract and secondary summaries) | direct evidence that visible cues reveal intent before speech; one cue (gaze), a constrained task, a robot partner |

---

## 5. Where our project sits

Three lines of work meet here, and none covers the combination:

1. **Theory-of-mind benchmarks** test goal and belief inference, but for one actor (MMToM-QA, EgoToM), in simulation or gridworlds (MMToM-QA, ToMnet), or about the camera wearer rather than a partner (EgoToM).
2. **Human-AI coordination** studies agents working with partners, but in simulators where intent is known by construction (Overcooked-AI, Watch-And-Help, PARTNR).
3. **Real collaboration datasets** come closest. BOSS predicts beliefs in nonverbal dyads, but with speech forbidden, short tasks and small scale; CoMind anticipates actions and handovers, not the intent behind them.

**Our contribution:** predict a **visible partner's naturally stated intent** from **muted real first-person video of both people**, using **speech as free, large-scale ground truth** (about 1,800 stated intents in CoMind), and measure **which visible cues** (gesture, gaze, world state) make that possible, by comparing gesture-cued, speech-cued and uncued moments.

## Sources

- [Baker, Saxe, Tenenbaum 2009 (MIT DSpace)](https://dspace.mit.edu/handle/1721.1/60852)
- [Machine Theory of Mind (arXiv 1802.07740)](https://arxiv.org/abs/1802.07740v2)
- [MMToM-QA (ACL 2024)](https://preview.aclanthology.org/add_acl24_videos/2024.acl-long.851)
- [EgoToM (arXiv 2503.22152)](https://arxiv.org/abs/2503.22152v1)
- [BOSS (arXiv 2206.10665)](https://arxiv.org/pdf/2206.10665)
- [BOSS project page](https://sites.google.com/view/bossbelief)
- [MToMnet (arXiv 2407.06762)](https://arxiv.org/html/2407.06762v2)
- [MindCraft (EMNLP 2021)](https://www.aclanthology.org/2021.emnlp-main.85/)
- [CoMind (arXiv 2607.06691)](https://arxiv.org/pdf/2607.06691)
- [HoloAssist (arXiv 2309.17024)](https://arxiv.org/abs/2309.17024)
- [IndEgo (NeurIPS 2025 D&B)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8e5dc5969a6174fcaaececd890c7f59b-Abstract-Datasets_and_Benchmarks_Track.html)
- [On the Utility of Learning about Humans for Human-AI Coordination (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html)
- ["Other-Play" for Zero-Shot Coordination (ICML 2020)](https://mlanthology.org/icml/2020/hu2020icml-otherplay)
- [Watch-And-Help (ICLR 2021)](https://iclr.cc/virtual/2021/spotlight/3491)
- [PARTNR project page](https://aihabitat.org/partnr/)
- [Anticipatory Robot Control (HRI 2016)](https://peopleandrobots.wisc.edu/publications/Anticipatory-Robot-Control-for-Efficient-Human-Robot-Collaboration-inproceedings/)
