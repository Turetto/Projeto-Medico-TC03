# Projeto-Medico-TC03 — Triagem Automática de Laudos Médicos

Projeto desenvolvido para o **Tech Challenge — Fase 3 (PosTech)**: deploy de um modelo de NLP em produção, cobrindo decisão arquitetural, CI/CD, orquestração de pipeline de treino, monitoramento e otimização de latência.

## Sumário

1. [Visão Geral](#visão-geral)
2. [Decisão Arquitetural](#decisão-arquitetural)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Setup do Ambiente](#setup-do-ambiente)
5. [Modelo de Classificação](#modelo-de-classificação)
6. [API de Inferência](#api-de-inferência)
7. [CI/CD](#cicd)
8. [Pipeline de Treino (Airflow)](#pipeline-de-treino-airflow)
9. [Monitoramento e Observabilidade](#monitoramento-e-observabilidade)
10. [Otimização de Latência](#otimização-de-latência)
11. [Como Rodar Tudo](#como-rodar-tudo)
12. [Limitações Conhecidas](#limitações-conhecidas)
13. [Trabalhos Futuros](#trabalhos-futuros)

---

## Visão Geral

O projeto implementa um serviço de classificação automática de laudos/resumos médicos em **5 categorias de condição**, servido via API REST, com pipeline completo de MLOps: treino orquestrado, CI/CD automatizado, monitoramento em tempo real e uma análise crítica sobre otimização de latência.

**Dataset**: [Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/chaitanyakck/medical-text) (Kaggle) — 14.438 resumos médicos em inglês, rotulados em 5 classes:

| Label | Condição |
|---|---|
| 1 | Neoplasms (neoplasias/tumores) |
| 2 | Digestive system diseases |
| 3 | Nervous system diseases |
| 4 | Cardiovascular diseases |
| 5 | General pathological conditions |

## Decisão Arquitetural

**Deploy em tempo real**, via API REST síncrona, containerizada. Provedor de referência: **Azure** (ex.: Azure Container Apps ou Azure App Service).

A escolha de tempo real considera o caso de uso: triagem de laudos exige resposta imediata para priorização clínica. Descartamos:
- **Batch**: introduziria atraso inaceitável para casos que exigem triagem imediata.
- **Serverless puro**: cold start incompatível com SLA de resposta clínica.

**Escopo de cloud**: a arquitetura foi projetada para portabilidade (containerizada, sem dependências específicas de provedor), com Azure como referência documentada. **O deploy real em ambiente de nuvem foi conscientemente deixado fora do escopo desta entrega**, priorizando profundidade nas etapas de CI/CD, orquestração, observabilidade e otimização de latência — a portabilidade do container já demonstra viabilidade de deploy em qualquer provedor sem alteração de código.

## Estrutura do Projeto

```
Projeto-Medico-TC03/
├── .github/workflows/ci.yml         # Pipeline de CI (lint + testes)
├── airflow/
│   ├── dags/train_pipeline_dag.py   # DAG de treino/retreino
│   ├── Dockerfile                   # Imagem Airflow + deps de ML
│   ├── requirements-airflow.txt
│   └── docker-compose.yaml          # Orquestração do Airflow (oficial + build customizado)
├── app/
│   ├── main.py                      # Endpoints FastAPI
│   ├── schemas.py                   # Contratos Pydantic
│   ├── inference.py                 # Carregamento/predição/reload do modelo
│   └── metrics.py                   # Instrumentação Prometheus
├── model/
│   ├── pipeline_utils.py            # Lógica compartilhada (dados, treino, quality gate)
│   ├── train.py                     # Treino manual (CLI)
│   ├── export_onnx.py               # Conversão para ONNX + quantização
│   └── artifacts/                   # Modelos treinados (versionados no Git)
├── monitoring/
│   ├── prometheus.yml
│   ├── docker-compose.yml           # API + Prometheus + Grafana
│   └── grafana/provisioning/        # Datasource e dashboard como código
├── scripts/
│   ├── benchmark_latency.py         # Benchmark via HTTP (ponta a ponta)
│   ├── benchmark_model_only.py      # Benchmark in-process (só o modelo)
│   ├── benchmark_onnx.py            # Comparativo sklearn vs ONNX FP32 vs INT8
│   └── validate_onnx_parity.py      # Confere paridade de predições
├── tests/
│   ├── test_api.py                  # Testes funcionais da API
│   └── test_data_contracts.py       # Testes de validação de entrada
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup do Ambiente

- **Gerenciador de projeto**: [uv](https://docs.astral.sh/uv/)
- **Versão do Python**: 3.14
- **Linter/formatter**: [Ruff](https://docs.astral.sh/ruff/)

```bash
uv init Projeto-Medico-TC03 --python 3.14
cd Projeto-Medico-TC03
uv add fastapi "uvicorn[standard]" prometheus-client kagglehub joblib pandas scikit-learn imbalanced-learn
uv add --dev ruff pytest httpx requests skl2onnx onnx onnxruntime
```

Lint e formatação (rodar antes de todo commit):
```bash
uv run ruff check . --fix
uv run ruff format .
```

## Modelo de Classificação

### Evolução do modelo

| Versão | Vetorização | Classificador | Balanceamento | Macro F1 |
|---|---|---|---|---|
| v1 (baseline) | TF-IDF (20k features) | Random Forest | `class_weight=balanced` | 0.48 |
| v2 | TF-IDF (tunado: `min_df`, `sublinear_tf`) | LinearSVC | RandomOverSampler | 0.50 |
| v3 (produção) | TF-IDF (tunado) | **LogisticRegression** | RandomOverSampler | **0.578** |

A migração para `LogisticRegression` na v3 ocorreu durante a Etapa 4, motivada originalmente por uma limitação de compatibilidade do `LinearSVC` na conversão para ONNX (ausência de `predict_proba` nativo). A troca trouxe, como benefício adicional, um ganho real de Macro F1 (+0.078 sobre a v2).

### Quality Gate

O treino (manual ou via Airflow) só **promove** um modelo candidato a produção se `macro_f1 >= 0.45` (`MIN_MACRO_F1` em `model/pipeline_utils.py`). Isso evita que um retreino com dados ruins substitua um modelo funcional.

### Limitação conhecida: classe "general_pathological_conditions"

Essa classe é, por natureza, um "catch-all" heterogêneo (não representa uma condição médica específica, mas um agrupamento residual). Ela consistentemente apresenta o pior desempenho entre as 5 classes — comportamento documentado na literatura sobre esse dataset, não uma falha do pipeline.

## API de Inferência

Construída com **FastAPI**. Endpoints:

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Health check (usado por orquestradores/monitoramento) |
| POST | `/classificar` | Classifica um texto em uma das 5 condições |
| GET | `/metrics` | Métricas no formato Prometheus |
| POST | `/admin/reload-model` | Recarrega o modelo do disco sem reiniciar o container (protegido por token) |

**Exemplo de request** (`POST /classificar`):
```json
{"texto": "Patient presented with chest pain and elevated troponin levels"}
```
**Response**:
```json
{"condicao": "cardiovascular_diseases", "label_id": 4}
```

O modelo é carregado em memória com `@lru_cache` para evitar releitura do disco a cada requisição.

### Endpoint de reload — por que existe

O treino (Airflow) e a API são sistemas **desacoplados**, conectados apenas pelo arquivo `model/artifacts/baseline_model.joblib` em um volume compartilhado. Como a API mantém o modelo em cache de memória, ela não percebe automaticamente quando o Airflow promove um modelo novo. O endpoint `POST /admin/reload-model` (protegido por header `X-Admin-Token`, configurado via variável de ambiente `ADMIN_RELOAD_TOKEN`) resolve isso sem exigir reinício do container.

## CI/CD

**GitHub Actions** (`.github/workflows/ci.yml`), com dois jobs sequenciais:

1. **`lint`**: `ruff check` + `ruff format --check`
2. **`test`** (roda só se `lint` passar): `pytest` sobre os testes funcionais e de contrato de dados da API

Ambiente reproduzível via `uv sync --frozen`, usando o `uv.lock` versionado.

## Pipeline de Treino (Airflow)

DAG `treino_classificador_laudos` (`airflow/dags/train_pipeline_dag.py`), com 3 tasks via TaskFlow API:

```
ingestao → treino → validacao_e_promocao
```

- **`ingestao`**: baixa o dataset do Kaggle (com retries automáticos para falhas transitórias de rede)
- **`treino`**: treina o modelo e retorna só as métricas via XCom (o artefato do modelo fica em disco, fora do banco de metadados do Airflow)
- **`validacao_e_promocao`**: aplica o quality gate (`MIN_MACRO_F1 = 0.45`); só promove o candidato a produção se aprovado

O Airflow roda em ambiente Docker próprio (imagem oficial `apache/airflow:3.3.1` estendida com as dependências de ML), separado do ambiente `uv` da API — os dois só se comunicam através do artefato de modelo em um volume compartilhado.

Disparo atual: **manual** (`schedule=None`). Retreino agendado automaticamente é uma extensão natural (`schedule="@weekly"`), fora do escopo desta entrega.

## Monitoramento e Observabilidade

**Prometheus + Grafana**, orquestrados via `monitoring/docker-compose.yml`, com dashboard e datasource provisionados como código (sem configuração manual).

### Métricas expostas (`app/metrics.py`)
- `api_requests_total{status}` — contador de requisições por status (sucesso/erro)
- `api_predictions_by_class_total{condicao}` — contador de predições por condição médica prevista (permite observar deriva de dados)
- `api_inference_latency_seconds` — histograma de latência de inferência (usado para calcular p50/p95/p99)

### Dashboard (3 painéis)
1. **Taxa de Requisições** (por status) — `rate(api_requests_total[1m])`
2. **Latência de Inferência (p50/p95/p99)** — `histogram_quantile()` sobre o histograma de latência
3. **Predições por Condição Médica** — distribuição das classes previstas ao longo do tempo

> *Inserir aqui os prints do dashboard Grafana.*

### Observação sobre outliers de latência

Chamadas logo após um reinício do container ou um `/admin/reload-model` incluem o tempo de recarregar o modelo do disco, gerando picos pontuais no p95/p99. Esse comportamento foi observado e confirmado nos dados reais do dashboard, e é esperado — reloads são eventos raros, não representativos da latência normal de operação.

## Otimização de Latência

### Benchmark ponta a ponta (via HTTP, Subetapa 1.4)
| Métrica | Valor |
|---|---|
| Mediana (p50) | ~3 ms |
| p95 | ~25 ms |
| p99 | ~29 ms |

### Benchmark do modelo isolado (in-process, sem overhead de rede)

| Versão | p50 (ms) | Tamanho (KB) |
|---|---|---|
| sklearn (produção) | 0.57 | 1744 |
| ONNX FP32 | 0.16 | 1079 |
| ONNX INT8 (quantizado) | 0.14 | 1079 |

Paridade de predições validada: **5/5** casos de teste concordam entre sklearn, ONNX FP32 e ONNX INT8.

### Decisão: manter o modelo sklearn original em produção

O ganho de latência do ONNX (~0.4 ms) é irrelevante frente à latência real de ponta a ponta da API (~3 ms, dominada por rede e overhead HTTP), e não justifica o custo de manutenção de adicionar `onnxruntime` como dependência de produção e sincronizar um passo de exportação a cada retreino. Os artefatos ONNX foram mantidos no repositório como evidência do experimento, mas não são usados em runtime.

**Descoberta técnica**: a maior parte do ganho de latência do ONNX vem do *runtime* de execução (grafo compilado vs. interpretação Python), não da quantização em si — os pesos do classificador são pequenos; o tamanho do artefato é dominado pelo vocabulário do TF-IDF (20.000 termos), que não é afetado por quantização de pesos.

## Como Rodar Tudo

### 1. API isolada (Docker)
```bash
docker build -t triagem-laudos-api:latest .
docker run -d -p 8000:8000 --name triagem-api -e ADMIN_RELOAD_TOKEN=segredo-temporario-123 triagem-laudos-api:latest
```

### 2. Stack completa de monitoramento (API + Prometheus + Grafana)
```bash
docker compose -f monitoring/docker-compose.yml up -d --build
```
- API: http://127.0.0.1:8000
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 (login: `admin` / `admin`)

### 3. Airflow (pipeline de treino)
```bash
cd airflow
docker compose up airflow-init
docker compose up -d
```
- Interface: http://localhost:8080 (login padrão: `airflow` / `airflow`)

### 4. Testes e lint (ambiente local via uv)
```bash
uv run ruff check .
uv run pytest -v
```

## Limitações Conhecidas

- **Idioma**: o modelo foi treinado exclusivamente com textos em inglês. Entradas em outros idiomas não são rejeitadas pela API, mas produzem classificações não confiáveis, pois o vocabulário do TF-IDF não reconhece os termos.
- **Classe "general_pathological_conditions"**: desempenho consistentemente inferior às demais classes, por ser uma categoria heterogênea por definição do próprio dataset.
- **Latência pós-reload**: requisições imediatamente após um `/admin/reload-model` (ou reinício do container) pagam o custo de recarregar o modelo do disco, gerando outliers pontuais nas métricas de latência.
- **Sem deploy real em nuvem**: a arquitetura foi projetada para portabilidade, mas não foi provisionada em um ambiente Azure real nesta entrega (ver seção "Decisão Arquitetural").
- **Monitoramento e orquestração não estão na nuvem**: Prometheus, Grafana e Airflow rodam localmente via Docker; não foram implantados em serviços gerenciados equivalentes.

## Trabalhos Futuros

- Deploy real da API em Azure (Container Apps ou App Service), validando a portabilidade documentada.
- Suporte multi-idioma (detecção de idioma + tradução, ou re-treino com corpus multilíngue).
- Embeddings semânticos (ex. `sentence-transformers`) como alternativa ao TF-IDF, com nova avaliação de latência.
- Retreino agendado automaticamente (`schedule="@weekly"` na DAG), com avaliação periódica de deriva de dados via o painel de "Predições por Condição Médica".
- Validação cruzada K-Fold para uma estimativa mais robusta das métricas do modelo.