from __future__ import annotations

import sys

import pendulum

sys.path.insert(0, "/opt/airflow")

from airflow.sdk import dag, task

from model.pipeline_utils import (
    MIN_MACRO_F1,
    download_dataset,
    load_train_data,
    promote_candidate_to_production,
    save_candidate,
    train_and_evaluate,
)


@dag(
    dag_id="treino_classificador_laudos",
    schedule=None,  # disparo manual
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["tech-challenge-fase3", "treino"],
)
def treino_classificador_laudos():

    @task(retries=2, retry_delay=pendulum.duration(minutes=1))
    def ingestao() -> str:
        """
        Baixa o dataset do Kaggle
        """
        return download_dataset()

    @task
    def treino(dataset_path: str) -> dict:
        """
        Treina o modelo. Retorna só as métricas via XCom
        """
        df = load_train_data(dataset_path)
        pipeline, metrics = train_and_evaluate(df)
        save_candidate(pipeline)
        return {"macro_f1": metrics["macro_f1"]}

    @task
    def validacao_e_promocao(metrics: dict) -> None:
        """
        Quality gate: só promove o candidato a produção se atingir o
        macro F1 mínimo. Se falhar, a task (e a DAG) falha de propósito
        """
        macro_f1 = metrics["macro_f1"]
        if macro_f1 < MIN_MACRO_F1:
            raise ValueError(
                f"Macro F1 ({macro_f1:.3f}) abaixo do limiar mínimo ({MIN_MACRO_F1}). Modelo NÃO promovido."
            )
        promote_candidate_to_production()

    dataset_path = ingestao()
    metrics = treino(dataset_path)
    validacao_e_promocao(metrics)


treino_classificador_laudos()
