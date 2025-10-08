import argparse
import base64
import io
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EDA_REPORT")


# Вспомогательные исключения


class ReportError(Exception):
    """Базовое исключение отчёта."""


class DatasetLoadError(ReportError):
    """Ошибка загрузки датасета."""


class PlotError(ReportError):
    """Ошибка построения графика."""


class StatsError(ReportError):
    """Ошибка статистического теста."""


# Загрузка датасетов


def load_dataset(name: str = "iris") -> Tuple[pd.DataFrame, Optional[pd.Series], str]:
    """
    Загружает датасет как DataFrame и (опционально) Series-таргет.
    Возвращает (df, target, target_name).
    """
    try:
        name = name.strip().lower()
        if name not in {"iris", "boston"}:
            raise ValueError("dataset must be 'iris' or 'boston'")

        if name == "iris":
            from sklearn.datasets import load_iris

            data = load_iris(as_frame=True)
            df = data.frame.copy()
            target_name = data.target.name if hasattr(data, "target") else "target"
            target = df[target_name]
            # Переименуем признаки для удобства
            df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
            return df, target, target_name

        if name == "boston":
            try:
                from sklearn.datasets import load_boston
            except Exception as e:
                raise DatasetLoadError(
                    "load_boston недоступен в вашей версии scikit-learn. "
                    "Используйте --dataset iris."
                ) from e
            data = load_boston()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            target = pd.Series(data.target, name="MEDV")
            df["MEDV"] = target
            return df, target, "MEDV"
    except Exception as e:
        logger.exception("Не удалось загрузить датасет.")
        raise DatasetLoadError(str(e)) from e


def create_features(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
    Создаёт ≥5 новых признаков.
    Универсально для iris/boston (разные ветки в зависимости от доступных колонок).
    Возвращает новый DF с добавленными колонками.
    """
    try:
        new_df = df.copy()
        cols_num = [
            c
            for c in new_df.columns
            if c != target_name and pd.api.types.is_numeric_dtype(new_df[c])
        ]

        # Если это iris — ожидаем классические 4 фичи: sepal/petal длины/ширины
        iris_like = set([c.lower() for c in new_df.columns])
        if any("petal" in c for c in iris_like) or any("sepal" in c for c in iris_like):
            # Нормированные признаки
            for c in cols_num:
                new_df[f"{c}_z"] = (new_df[c] - new_df[c].mean()) / (
                    new_df[c].std(ddof=0) + 1e-9
                )
            # Отношения и взаимодействия
            if "petal_length" in new_df.columns and "sepal_length" in new_df.columns:
                new_df["petal_sepal_len_ratio"] = (new_df["petal_length"] + 1e-9) / (
                    new_df["sepal_length"] + 1e-9
                )
            if "petal_width" in new_df.columns and "sepal_width" in new_df.columns:
                new_df["petal_sepal_wid_ratio"] = (new_df["petal_width"] + 1e-9) / (
                    new_df["sepal_width"] + 1e-9
                )
            # Комбинации
            if set(["petal_length", "petal_width"]).issubset(new_df.columns):
                new_df["petal_area"] = new_df["petal_length"] * new_df["petal_width"]
            if set(["sepal_length", "sepal_width"]).issubset(new_df.columns):
                new_df["sepal_area"] = new_df["sepal_length"] * new_df["sepal_width"]
            # Полиномиальные
            for c in ["petal_length", "petal_width"]:
                if c in new_df.columns:
                    new_df[f"{c}_sq"] = new_df[c] ** 2

        else:
            # Ветка для boston/других регрессий — создаём взаимодействия, логи и т.п.
            for c in cols_num:
                new_df[f"{c}_log1p"] = np.log1p(new_df[c].clip(lower=0))
            # Примеры взаимодействий/top-k коррелирующих
            if len(cols_num) >= 2:
                new_df["feat_interaction_1"] = new_df[cols_num[0]] * new_df[cols_num[1]]
            if len(cols_num) >= 3:
                new_df["feat_interaction_2"] = new_df[cols_num[1]] * new_df[cols_num[2]]
            # Квадраты
            for c in cols_num[:3]:
                new_df[f"{c}_sq"] = new_df[c] ** 2
            # Суммарный индекс (нормированный)
            zsum = []
            for c in cols_num[:5]:
                z = (new_df[c] - new_df[c].mean()) / (new_df[c].std(ddof=0) + 1e-9)
                zsum.append(z)
            if zsum:
                new_df["z_index_sum"] = np.vstack(zsum).sum(axis=0)

        return new_df
    except Exception as e:
        logger.exception("Ошибка при создании признаков.")
        raise ReportError(f"Feature engineering failed: {e}") from e


# EDA: пропуски и выбросы


def eda_basic(df: pd.DataFrame, target_name: str) -> Dict:
    """
    Анализ пропусков, выбросов (IQR) и распределений (скошенность/эксцесс).
    Возвращает словарь метрик и таблиц.
    """
    try:
        result = {}
        # Пропуски
        missing = df.isna().sum().to_frame("missing_count")
        missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)

        # Скошенность и эксцесс
        desc = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                desc.append(
                    {
                        "feature": c,
                        "mean": float(np.nanmean(df[c])),
                        "std": float(np.nanstd(df[c], ddof=0)),
                        "min": float(np.nanmin(df[c])),
                        "max": float(np.nanmax(df[c])),
                        "skew": float(stats.skew(df[c], nan_policy="omit")),
                        "kurtosis": float(stats.kurtosis(df[c], nan_policy="omit")),
                    }
                )
        dist_df = pd.DataFrame(desc).sort_values("feature")

        # Выбросы по IQR
        outlier_info = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                q1, q3 = np.nanpercentile(df[c], [25, 75])
                iqr = q3 - q1
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                is_out = (df[c] < low) | (df[c] > high)
                outlier_info.append(
                    {
                        "feature": c,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "low_thr": low,
                        "high_thr": high,
                        "outliers_count": int(is_out.sum()),
                        "outliers_pct": round(float(is_out.mean() * 100), 2),
                    }
                )
        outliers_df = pd.DataFrame(outlier_info).sort_values(
            "outliers_pct", ascending=False
        )

        result["missing"] = missing
        result["dist"] = dist_df
        result["outliers"] = outliers_df
        return result
    except Exception as e:
        logger.exception("Ошибка EDA (базовый анализ).")
        raise ReportError(f"EDA basic failed: {e}") from e


# -Статистичекские тесты


def run_stat_tests(df: pd.DataFrame, target_name: str) -> Dict[str, Dict]:
    """
    Выполняет ≥3 статистических теста.
    Для iris (классификация):
        1) Shapiro–Wilk на нормальность (по фичам)
        2) One-way ANOVA (фича ~ класс)
        3) Kruskal–Wallis (робастная альтернатива)
    Для boston (регрессия):
        1) Shapiro–Wilk на нормальность таргета
        2) Пирсон корреляция feature ~ target (топ-3)
        3) Левен (равенство дисперсий между группами по бинингу таргета)
    Возвращает dict результатов.
    """
    try:
        results = {}
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        is_classification = (
            pd.api.types.is_integer_dtype(df[target_name])
            or df[target_name].nunique() <= 10
        )

        # 1) Shapiro–Wilk
        shapiro_res = {}
        for c in numeric_cols[:8]:  # ограничим количество для скорости
            try:
                w, p = stats.shapiro(
                    df[c].sample(min(len(df[c]), 500), random_state=42)
                )
                shapiro_res[c] = {"W": float(w), "p_value": float(p)}
            except Exception:
                # Shapiro требует 3..5000 наблюдений — пропустим, если не подходит
                continue
        results["shapiro"] = shapiro_res

        if is_classification:
            # Группы по таргету
            groups = [g[1] for g in df.groupby(target_name)]
            # 2) ANOVA для первой подходящей фичи и ещё пары
            anova = {}
            for c in numeric_cols[:5]:
                try:
                    samples = [grp[c].values for grp in groups]
                    f, p = stats.f_oneway(*samples)
                    anova[c] = {"F": float(f), "p_value": float(p)}
                except Exception:
                    continue
            results["anova"] = anova

            # 3) Kruskal–Wallis
            kruskal = {}
            for c in numeric_cols[:5]:
                try:
                    samples = [grp[c].values for grp in groups]
                    h, p = stats.kruskal(*samples)
                    kruskal[c] = {"H": float(h), "p_value": float(p)}
                except Exception:
                    continue
            results["kruskal"] = kruskal

        else:
            # Регрессия (boston)
            # 2) Корреляции Пирсона фич -> таргет
            pearson = {}
            for c in numeric_cols:
                if c == target_name:
                    continue
                try:
                    r, p = stats.pearsonr(df[c], df[target_name])
                    pearson[c] = {"r": float(r), "p_value": float(p)}
                except Exception:
                    continue
            # Отберём топ-5 по |r|
            pearson_top = dict(
                sorted(pearson.items(), key=lambda kv: abs(kv[1]["r"]), reverse=True)[
                    :5
                ]
            )
            results["pearson_top"] = pearson_top

            # 3) Левен: сравним дисперсии признака между 3 квантильными группами таргета
            try:
                bins = pd.qcut(df[target_name], q=3, labels=False, duplicates="drop")
                levene = {}
                for c in numeric_cols[:5]:
                    if c == target_name:
                        continue
                    g0 = df.loc[bins == 0, c]
                    g1 = df.loc[bins == 1, c]
                    g2 = df.loc[bins == 2, c]
                    stat, p = stats.levene(g0, g1, g2, center="median")
                    levene[c] = {"stat": float(stat), "p_value": float(p)}
                results["levene"] = levene
            except Exception as e:
                results["levene"] = {"error": str(e)}

        return results
    except Exception as e:
        logger.exception("Ошибка при выполнении статистических тестов.")
        raise StatsError(f"Stat tests failed: {e}") from e


# Визуализация


def _fig_to_base64_png(fig) -> str:
    """Преобразует matplotlib.figure в base64 PNG-строку."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def make_plots(df: pd.DataFrame, target_name: str) -> Dict[str, str]:
    """
    Строит минимум 5 графиков и возвращает dict {plot_name: base64_png}.
    """
    try:
        plots: Dict[str, str] = {}
        num_cols = [
            c
            for c in df.columns
            if c != target_name and pd.api.types.is_numeric_dtype(df[c])
        ][:6]
        if not num_cols:
            raise PlotError("Нет числовых признаков для визуализации.")

        # 1) Гистограммы (grid)
        fig, axes = plt.subplots(
            nrows=int(np.ceil(len(num_cols) / 2)), ncols=2, figsize=(10, 8)
        )
        axes = axes.flatten()
        for i, c in enumerate(num_cols):
            axes[i].hist(df[c].dropna(), bins=20, edgecolor="black")
            axes[i].set_title(f"Histogram: {c}")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plots["histograms"] = _fig_to_base64_png(fig)

        # 2) Boxplot по фичам
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([df[c].dropna() for c in num_cols], tick_labels=num_cols, vert=True)
        ax.set_title("Boxplots by feature")
        ax.tick_params(axis="x", rotation=45)
        plots["boxplots"] = _fig_to_base64_png(fig)

        # 3) Корреляционная матрица
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[[*num_cols, target_name]].corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation heatmap")
        plots["corr_heatmap"] = _fig_to_base64_png(fig)

        # 4) Scatter matrix (pairwise)
        from pandas.plotting import scatter_matrix

        fig = plt.figure(figsize=(10, 10))
        axs = scatter_matrix(df[num_cols], diagonal="hist", figsize=(10, 10))
        # Подправим заголовок на одной из осей
        plt.suptitle("Scatter matrix (subset)", y=1.02)
        plots["scatter_matrix"] = _fig_to_base64_png(plt.gcf())

        # 5) Violin/Box по классам (если классификация) или таргет и признак (регрессия)
        is_classification = (
            pd.api.types.is_integer_dtype(df[target_name])
            or df[target_name].nunique() <= 10
        )
        if is_classification:
            fig, ax = plt.subplots(figsize=(9, 5))
            c0 = num_cols[0]
            sns.violinplot(x=target_name, y=c0, data=df, ax=ax)
            ax.set_title(f"Violin: {c0} by {target_name}")
            plots["violin_by_class"] = _fig_to_base64_png(fig)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            c0 = num_cols[0]
            ax.scatter(df[c0], df[target_name])
            ax.set_xlabel(c0)
            ax.set_ylabel(target_name)
            ax.set_title(f"{c0} vs {target_name}")
            plots["scatter_vs_target"] = _fig_to_base64_png(fig)

        return plots
    except Exception as e:
        logger.exception("Ошибка при построении графиков.")
        raise PlotError(f"Plotting failed: {e}") from e


# HTML/PDF отчет


def df_to_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Безопасный рендер таблицы в HTML."""
    try:
        if len(df) > max_rows:
            df = df.head(max_rows)
        return df.to_html(
            classes="table table-sm table-striped",
            border=0,
            float_format=lambda x: f"{x:,.4f}",
        )
    except Exception as e:
        return f"<pre>Table render error: {e}</pre>"


def generate_html_report(
    dataset_name: str,
    df: pd.DataFrame,
    eda: Dict,
    tests: Dict,
    plots: Dict[str, str],
    features_info: List[str],
) -> str:
    """Генерирует HTML (как строку)."""
    try:
        style = """
        <style>
        body{font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin:24px; line-height:1.5;}
        h1,h2,h3{margin:0.6em 0;}
        .grid{display:grid; grid-template-columns:1fr 1fr; gap:18px;}
        .card{border:1px solid #ddd; border-radius:10px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,0.04);}
        .muted{color:#666;}
        img{max-width:100%; height:auto; border-radius:10px; border:1px solid #eee;}
        .table{width:100%; font-size:14px;}
        .foot{margin-top:24px; font-size:12px; color:#666;}
        code{background:#f6f8fa; padding:2px 6px; border-radius:6px;}
        </style>
        """
        # Блок тестов (коротко, json)
        tests_json = json.dumps(tests, ensure_ascii=False, indent=2)[:120000]

        # Секции
        header = f"""
        <h1>Комплексный отчёт EDA — {dataset_name}</h1>
        <p class="muted">Содержимое: данные, пропуски/выбросы/распределения, 5+ признаков, 3+ статистических теста, 5+ графиков.</p>
        """

        data_preview = f"""
        <div class="card">
          <h2>1) Датасет: предпросмотр</h2>
          <div>{df_to_html_table(df.head(10))}</div>
          <p class="muted">shape = {df.shape}</p>
        </div>
        """

        eda_section = f"""
        <div class="card">
          <h2>2) Пропуски, выбросы, распределения</h2>
          <h3>Пропуски</h3>
          {df_to_html_table(eda.get("missing", pd.DataFrame()))}
          <h3>Распределения (скошенность/эксцесс)</h3>
          {df_to_html_table(eda.get("dist", pd.DataFrame()))}
          <h3>Выбросы (IQR)</h3>
          {df_to_html_table(eda.get("outliers", pd.DataFrame()))}
        </div>
        """

        feats_list = "".join([f"<li><code>{f}</code></li>" for f in features_info])
        feats_section = f"""
        <div class="card">
          <h2>3) Созданные признаки (≥5)</h2>
          <ul>{feats_list}</ul>
        </div>
        """

        tests_section = f"""
        <div class="card">
          <h2>4) Статистические тесты (≥3)</h2>
          <p>Ключевые результаты (JSON):</p>
          <pre>{tests_json}</pre>
        </div>
        """

        # Графики
        plot_cards = []
        for name, b64 in plots.items():
            plot_cards.append(
                f"""
            <div class="card">
              <h3>{name}</h3>
              <img src="data:image/png;base64,{b64}" alt="{name}">
            </div>"""
            )
        plots_section = f"""
        <div class="card">
          <h2>5) Визуализации (≥5)</h2>
          <div class="grid">
            {''.join(plot_cards)}
          </div>
        </div>
        """

        conclusions = """
        <div class="card">
          <h2>Выводы (кратко)</h2>
          <ul>
            <li>Пропусков, как правило, в встроенных датасетах sklearn нет; проверка включена на случай обобщения.</li>
            <li>Выбросы оценены по правилу IQR; при необходимости можно применять винсоризацию/лог-преобразования.</li>
            <li>Новые признаки (нормировки, отношения, площади/взаимодействия, лог-преобразования) помогают выявлять нелинейности.</li>
            <li>Статистические тесты показали различия распределений между классами (iris) или значимые корреляции (boston).</li>
            <li>Визуализации подсветили структуру данных и потенциально информативные фичи.</li>
          </ul>
        </div>
        """

        foot = """
        <div class="foot">
          Сгенерировано автоматически: eda_report_generator.py • Версия 1.0<br>
          Советы: добавьте модель/кросс-валидацию, автоселект фич, и генерацию интерпретаций (SHAP) при необходимости.
        </div>
        """

        html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{style}</head><body>{header}{data_preview}{eda_section}{feats_section}{tests_section}{plots_section}{conclusions}{foot}</body></html>"
        return html
    except Exception as e:
        logger.exception("Ошибка при генерации HTML.")
        raise ReportError(f"HTML generation failed: {e}") from e


def save_report(html: str, out_path: str, save_pdf: bool = False) -> Optional[str]:
    """
    Сохраняет HTML. Если save_pdf=True и доступен pdfkit + wkhtmltopdf, сохраняет также PDF.
    Возвращает путь к PDF или None.
    """
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML отчёт сохранён: %s", out_path)

        pdf_path = None
        if save_pdf:
            try:
                import pdfkit  # требует установленный wkhtmltopdf

                pdf_path = os.path.splitext(out_path)[0] + ".pdf"
                pdfkit.from_string(html, pdf_path)
                logger.info("PDF отчёт сохранён: %s", pdf_path)
            except Exception as e:
                logger.warning(
                    "PDF не создан (%s). Установите pdfkit и wkhtmltopdf для экспорта в PDF.",
                    e,
                )
                pdf_path = None
        return pdf_path
    except Exception as e:
        logger.exception("Ошибка сохранения отчёта.")
        raise ReportError(f"Saving report failed: {e}") from e


# ОСНОВНОЙ КОНТУР


def build_report(dataset: str, out_path: str, want_pdf: bool = False) -> None:
    """
    Полный цикл: загрузка -> фичи -> EDA -> тесты -> графики -> HTML/PDF.
    """
    # 1) Данные
    df_raw, target, target_name = load_dataset(dataset)
    dataset_name = dataset.upper()
    logger.info(
        "Датасет: %s | shape=%s | target=%s", dataset_name, df_raw.shape, target_name
    )

    # 2) Фичи
    df_feat = create_features(df_raw, target_name=target_name)
    created_cols = [c for c in df_feat.columns if c not in df_raw.columns]
    features_info = created_cols[:25]  # перечислим до 25 в отчёте

    # 3) EDA
    eda = eda_basic(df_feat, target_name=target_name)

    # 4) Тесты
    tests = run_stat_tests(df_feat, target_name=target_name)

    # 5) Графики
    plots = make_plots(df_feat, target_name=target_name)

    # 6) HTML/PDF
    html = generate_html_report(dataset_name, df_feat, eda, tests, plots, features_info)
    save_report(html, out_path, save_pdf=want_pdf)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Комплексный генератор EDA-отчёта (данные, визуализации, выводы)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="iris",
        help="iris (по умолчанию) или boston (если доступен)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eda_report_iris.html",
        help="Путь к HTML файлу отчёта",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Также сохранить PDF (требуется pdfkit + wkhtmltopdf)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        build_report(dataset=args.dataset, out_path=args.out, want_pdf=args.pdf)
        logger.info("Готово.")
        return 0
    except DatasetLoadError as e:
        logger.error("Ошибка датасета: %s", e)
        return 2
    except StatsError as e:
        logger.error("Ошибка статистики: %s", e)
        return 3
    except PlotError as e:
        logger.error("Ошибка графиков: %s", e)
        return 4
    except ReportError as e:
        logger.error("Ошибка отчёта: %s", e)
        return 5
    except Exception as e:
        logger.exception("Непредвиденная ошибка: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
