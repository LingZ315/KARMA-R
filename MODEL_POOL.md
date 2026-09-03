# Panel-C model pool

The exact machine-readable pool is `configs/panel_c_model_pool.json`.

- Incumbent: SmolVLM-Instruct @ `81cd9a775a4d644f2faf4e7becff4559b46b14c7`.
- Candidates: Granite 4.0 3B Vision @ `bf108f36960fb4df79bf035e506c592f4ee3c2d3`; Ovis2.5-9B @ `d73b2283ae2a930b7762f8d7b8b8a3f0f3b5c3bd`; Phi-4 Multimodal @ `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`; Qwen3-VL-4B-Instruct @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`; InternVL3.5-4B-HF @ `6bd4487402110ef9889ba50eb7aefeb302526fed`.
- Independent semantic classifier: Qwen3-VL-32B-Instruct @ `0cfaf48183f594c314753d30a4c4974bc75f3ccb`.

The semantic classifier is frozen to BF16 without quantization on exactly four visible RTX 5090 GPUs, `device_map="auto"`, SDPA, batch size one, and a 30-GiB per-device ceiling. The synthetic-only execution receipt is PASS. Its runtime settings do not change the model pool or any scientific parameter.

The pool is frozen without Panel-C predictions or outcomes. It will not be expanded to increase count. Any change requires a new preregistration version, archive, and external timestamp.
