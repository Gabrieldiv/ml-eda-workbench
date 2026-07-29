import numpy as np
import pandas as pd


# importa o pandas pois iremos precisar
def pearson_correlation(df: pd.DataFrame, config: dict, limit: float) -> str:
    # declara a função
    id_column = config.get("id_column")
    # pega o nome da coluna ID
    target_column = config.get("target_column")
    # pega o nome da coluna target
    lines = ["Pearson correlation for every column:"]
    # cria uma lista que conterá as colunas redundantes

    exclude = {col for col in [id_column, target_column] if col is not None}
    # cria um conjunto com cada coluna na lista que não for none, a coluna tem que existir
    df_feat = df.drop(columns=list(exclude), errors="ignore")
    # cria um dataset novo, sem a colunas id e target, se der algum erro, ignore

    numeric_features = set(df_feat.select_dtypes(include="number").columns)
    # pega só as colunas numéricas, para evitar tentar achar a correlação entre textos
    skipped_features = set(df_feat.select_dtypes(exclude="number").columns)
    # pega só as colunas não numéricas, para dizer no relatório quais são

    if skipped_features:
        # se existir colunas não numéricas
        lines.append("Skipped non-numeric columns:")
        lines.append("\n".join(sorted(skipped_features)))
        # coloca no relatório quais colunas são as não númericas, uma abaixo da outra

    if numeric_features:
        # se existir colunas numéricas
        df_numeric = df_feat[sorted(numeric_features)]
        # cria um dataset auxiliar somente com as colunas de dados númericos
        correlation_matrix = df_numeric.corr()
        # cria uma matriz com a correlação de todas as colunas entre si do dataset auxiliar
        mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        # pega somente o triângulo superior, correlações não repetidas e não auto-comparadas
        correlation_pairs = correlation_matrix.where(mask).stack().dropna()  # noqa: PD013
        # monta uma lista apenas com as correlações não repetidas e não auto-comparadas
        redundant_pairs = correlation_pairs[correlation_pairs.abs() >= limit]
        # cria uma lista apenas com as colunas de dados redundantes

        if redundant_pairs.empty:
            # se não houverem pares reduntantes.
            lines.append(f"All columns correlations below the defined limit: {limit} ")
            # diz que todas as colunas tem redundância abaixo do limite estabelecido.

        else:
            n_features = len(correlation_matrix)
            # pega a quantidade de colunas
            mean_abs_corr = (correlation_matrix.abs().sum() - 1) / (n_features - 1)
            # pega a correlação média de cada coluna com todas as outras

            for (feature_a, feature_b), corr_value in redundant_pairs.items():
                if mean_abs_corr[feature_a] >= mean_abs_corr[feature_b]:
                    suggested_drop = feature_a
                else:
                    suggested_drop = feature_b

                lines.append(
                    f"{feature_a} <-> {feature_b}: {corr_value:.3f} "
                    f"(suggest dropping: {suggested_drop})"
                )
                # para cada par redundante, sugere eliminar a que mais se correlaciona com todas as outras
    else:
        lines.append("Numeric columns not found.")

    return "\n".join(lines)
