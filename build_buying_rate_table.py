def build_buying_rate_table(act, cart, grouped_parameter, is_percent=True):
    import pandas as pd
    import numpy as np
    import string
    from stat_tests import stat_tests_proportions

    bases_df = act.groupby('experiment_name').uid.nunique()#cols: experiment_name, uid
    all_rows_options = act[act[grouped_parameter] != '-'][grouped_parameter].unique()
    dfs_list = []
    for opt in all_rows_options:
        all_buyers = act[(act.action == 'reached') & (act.page_type == 'checkout')].uid.unique()
        df_buyers = cart[(cart.uid.isin(all_buyers)) & (cart[grouped_parameter] == opt)].groupby('experiment_name').uid.nunique()
        dfs_list.append(pd.Series(df_buyers, name = opt))
    #grouped df
    grouped_df = pd.concat(dfs_list, axis = 1).fillna(0)
    
    if is_percent:
        grouped_df = grouped_df.div(bases_df, axis = 'rows').T
        grouped_df = grouped_df*100
    else:
        grouped_df = grouped_df.T
    
    grouped_df = grouped_df.sort_values(grouped_df.columns[0], ascending = False)
    #bases df
    base_row = bases_df.reset_index().T
    base_row.columns = base_row.loc['experiment_name']
    
    #stat tests
    stat_test_dict = {}
    for opt in grouped_df.index.values:
        stat_test_dict[opt] = {}
        for exp in grouped_df.columns:
            if not is_percent:
                nb = grouped_df.loc[opt, exp]
            else:
                nb = int(grouped_df.loc[opt, exp]*base_row.loc['uid', exp]/100)
                
            stat_test_dict[opt][exp] = {
                'number':nb,
                'base':base_row.loc['uid', exp],
                'stat':[],
            }
    stat_letters = dict(zip(grouped_df.columns, 
                    string.ascii_uppercase[:len(grouped_df.columns)]))
    stat_test_dict = stat_tests_proportions(stat_test_dict, stat_letters, no_stat_cols=[], alpha = 0.05)
    
    
    if is_percent:
        grouped_df = np.round(grouped_df, 2)
    #concatenation
    grouped_df = pd.concat([base_row, grouped_df])
    grouped_df = grouped_df.drop(index = ['experiment_name']).rename(index = {'uid':'Base'})

    for i, row in grouped_df.iterrows():
        if i != 'Base':
            for exp in grouped_df.columns:
                grouped_df.at[i, exp] = f"{row[exp]} {stat_test_dict[i][exp]['stat']}"
                
    if is_percent:
        grouped_df.rename(index = {i: i + ', %' for i in grouped_df.index.values if i != 'Base'}, inplace = True)

    if is_percent: 
        tested_index = [f'{i}, %' for i in act[act.is_tested_product == 1][grouped_parameter].unique()]
    else:
        tested_index = [i for i in act[act.is_tested_product == 1][grouped_parameter].unique()]
    
    # PUSHING THE TESTED ONES TO THE TOP
    grouped_df = grouped_df.loc[
                [i for i in grouped_df.index if
                        i in tested_index or i =='Base']
                +
                [i for i in grouped_df.index if
                        i not in tested_index and i != 'Base']
            ]
    n_letters = list(string.ascii_uppercase[:len(grouped_df.columns)])
    grouped_df.rename(columns = {i:f'{i}, {l}' for i,l in zip(grouped_df.columns, n_letters)}, inplace = True)

    #cutting to 15 top
    if grouped_parameter == 'product':
        grouped_df = grouped_df.iloc[:1+len(tested_index)+15, :]#base plus tested plus 15 competitors

    return grouped_df