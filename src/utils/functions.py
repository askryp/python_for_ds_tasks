"""Reusable plotting and analysis helpers for notebook workflows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def outlier_range(dataset, column):
    """Return the upper IQR outlier threshold for a numeric column."""
    q1 = dataset[column].quantile(0.25)
    q3 = dataset[column].quantile(0.75)
    iqr = q3 - q1
    return q3 + 1.5 * iqr


def draw_boxplot(df, categorical, continuous, max_continuous, title, hue_column, subplot_position):
    """Draw a boxplot for a numeric feature split by categorical dimensions."""
    plt.subplot(1, 2, subplot_position)
    plt.title(title)
    red_diamond = dict(markerfacecolor="r", marker="D")
    sns.boxplot(
        x=categorical,
        y=df[df[continuous] < max_continuous][continuous],
        data=df,
        flierprops=red_diamond,
        order=sorted(df[categorical].unique(), reverse=True),
        hue=hue_column,
        hue_order=sorted(df[hue_column].unique(), reverse=True),
    )
    plt.ticklabel_format(style="plain", axis="y")
    plt.xticks(rotation=90)


def bi_boxplot(df0, df1, categorical, continuous, max_continuous1, max_continuous0, hue_column):
    """Draw side-by-side boxplots for TARGET=1 and TARGET=0 groups."""
    plt.figure(figsize=(16, 10))

    draw_boxplot(df1, categorical, continuous, max_continuous1, "Payment Difficulties", hue_column, 1)
    draw_boxplot(df0, categorical, continuous, max_continuous0, "On-Time Payments", hue_column, 2)

    plt.tight_layout(pad=4)
    plt.show()


def numeric_vs_categorical_analysis(df0, df1, column_1, column_2, column_3):
    """Compare grouped descriptive stats and boxplots across TARGET subsets."""
    max_value1_column_1 = outlier_range(df1, column_1)
    max_value0_column_1 = outlier_range(df0, column_1)

    # Client group with payment difficulties
    print(df1.groupby(by=[column_2, column_3])[column_1].describe().head())

    # Client group with on-time payments
    print(df0.groupby(by=[column_2, column_3])[column_1].describe().head())

    bi_boxplot(df0, df1, column_2, column_1, max_value1_column_1, max_value0_column_1, column_3)

def bi_countplot_target(df0, df1, column, hue_column) :
    group_name = f'Нормалізований розподіл значень за категорією: {column}'
    print (group_name.upper())

    pltname = 'Клієнт зі складнощями щодо платності'
    unique_hue_values = df1[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14,4)

    proportions = df1.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
      by=unique_hue_values[0], ascending=False
      ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
      ax. bar_label(container, fmt='{:,.1f}%')

    pltname = 'Клієнти зі своєчасними платежами'
    unique_hue_values = df0[hue_column].unique()

    proportions = df0.groupby(hue_column)[column].value_counts (normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
    by=unique_hue_values[0], ascending=False
      ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
      ax.bar_label(container, fmt='{:,.1f}%')

    plt.show()

    #------------
    group_name = f'Кількість значень за категорією {column}'
    print(group_name.upper())

    pltname = 'Клієнт зі своєчасними платежами'
    unique_hue_values = df1[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14,4)
    counts = df1.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
      by=unique_hue_values [0], ascending=False
      ).plot.bar(ax=axes [0], title=pltname )

    for container in ax.containers:
      ax.bar_label(container)

    pltname = 'Клієнти зі своєчасними платежами'
    unique_hue_values = df0[hue_column].unique()
    counts = df0.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values (
      by=unique_hue_values[0], ascending=False
      ).plot.bar(ax=axes[1], title=pltname)

    for container in ax. containers:
      ax.bar_label(container)

    plt.show()


def bi_scatter_no_outliers(df0, df1, x_col, y_col, figsize=(14, 6)):
    """Plot side-by-side scatterplots for two groups after IQR upper clipping.

    Parameters
    ----------
    df0 : pandas.DataFrame
      Subset with TARGET=0 (on-time payments).
    df1 : pandas.DataFrame
      Subset with TARGET=1 (payment difficulties).
    x_col : str
      Feature to plot on x-axis.
    y_col : str
      Feature to plot on y-axis.
    figsize : tuple, default=(14, 6)
      Figure size for the combined chart.
    """
    max_x1 = outlier_range(df1, x_col)
    max_y1 = outlier_range(df1, y_col)
    max_x0 = outlier_range(df0, x_col)
    max_y0 = outlier_range(df0, y_col)

    filtered_df1 = df1[(df1[x_col] < max_x1) & (df1[y_col] < max_y1)]
    filtered_df0 = df0[(df0[x_col] < max_x0) & (df0[y_col] < max_y0)]

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.title("Payment Difficulties")
    sns.scatterplot(data=filtered_df1, x=x_col, y=y_col)
    plt.ticklabel_format(style="plain", axis="x")
    plt.ticklabel_format(style="plain", axis="y")

    plt.subplot(1, 2, 2)
    plt.title("On-Time Payments")
    sns.scatterplot(data=filtered_df0, x=x_col, y=y_col)
    plt.ticklabel_format(style="plain", axis="x")
    plt.ticklabel_format(style="plain", axis="y")

    plt.tight_layout(pad=4)
    plt.show()
