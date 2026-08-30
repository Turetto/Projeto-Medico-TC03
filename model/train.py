from model.pipeline_utils import (
    MIN_MACRO_F1,
    download_dataset,
    load_train_data,
    promote_candidate_to_production,
    save_candidate,
    train_and_evaluate,
)


def main():
    print("Baixando dataset...")
    dataset_path = download_dataset()

    print("Carregando dados...")
    df = load_train_data(dataset_path)
    print(f"Total de registros: {len(df)}")

    print("\nTreinando modelo...")
    pipeline, metrics = train_and_evaluate(df)
    print(f"\nMacro F1: {metrics['macro_f1']:.3f}")

    save_candidate(pipeline)

    if metrics["macro_f1"] >= MIN_MACRO_F1:
        promote_candidate_to_production()
        print(f"Modelo promovido a produção (Macro F1 >= {MIN_MACRO_F1}).")
    else:
        print(f"Macro F1 abaixo do limiar ({MIN_MACRO_F1}) - modelo NÃO promovido.")


if __name__ == "__main__":
    main()
