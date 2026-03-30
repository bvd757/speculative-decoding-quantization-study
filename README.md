# Quantized Draft Models for Speculative Decoding

Проект для курса **Эффективные системы ML**.

## Идея

В проекте исследуется, насколько эффективно использовать квантованные модели в роли **draft-моделей** в классическом speculative decoding.

Сравниваются:
- `bf16` draft
- `int8` draft
- `int4` draft

Основной вопрос:
ускоряет ли квантизация draft-модели генерацию достаточно сильно, чтобы это перекрыло возможное падение `acceptance rate`.

## Что измеряется

- **End-to-End Latency** — итоговое время генерации
- **Latency per Token** — среднее время на токен
- **Draft Generation Time** — время генерации черновика
- **Target Verification Time** — время верификации target-моделью
- **Acceptance Rate** — доля принятых draft-токенов
- **Acceptance Length** — средняя длина принятого префикса
- **VRAM Usage** — потребление видеопамяти

## Структура репозитория

```text
specdec-quant-draft/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── qwen25_3b_05b.yaml
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── figures/
│   └── notes.md
├── results/
│   ├── runs/
│   ├── tables/
│   └── profiles/
├── scripts/
│   ├── run_baseline.py
│   ├── run_speculative.py
│   ├── profile_run.py
│   └── plot_results.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── speculative.py
│   ├── metrics.py
│   ├── profiler.py
│   └── utils.py
└── tests/
    └── test_speculative.py