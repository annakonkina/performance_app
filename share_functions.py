def generate_grouped_df_metric_100(cart_buying, grouped_parameter, metric, base, label):
    grouped_df = (
        cart_buying
        .groupby([grouped_parameter])
        .agg({metric: 'sum'})
        )
    grouped_buyers_total = (
        cart_buying
        .groupby([grouped_parameter])
        .agg({'uid': 'nunique'})
        )
    grouped_df = (
        grouped_df
        .merge(grouped_buyers_total, how = 'left', on = grouped_parameter)
        .rename(columns = {'uid':'total_buyers',
                            metric:'total_metric'})
        )
    grouped_df = (
        grouped_df
        .assign(mean_metric = grouped_df.total_metric/grouped_df.total_buyers)
        )
    grouped_df = (
        grouped_df
        .assign(metric_100 = grouped_df.total_buyers/base * 100 * grouped_df.mean_metric)
        )
    grouped_df = grouped_df[['metric_100']]
    grouped_df.rename(columns = {'metric_100':f'{label}'}, inplace = True)
    grouped_df = grouped_df.sort_values(by=label, ascending=False)
    return grouped_df