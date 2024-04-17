
def stat_tests_proportions(vals_dict, stat_letters, no_stat_cols=[], alpha = 0.05):

    from statsmodels.stats.proportion import proportions_ztest
    import numpy as np
    import math
    import string

    def check_prop(p1ch, n1ch, p2ch, n2ch):
        if p1ch == '-':
            p1ch = 0
        if p2ch == '-':
            p2ch = 0
        if n1ch == '-':
            n1ch = 0
        if n2ch == '-':
            n2ch = 0

        stat, p = proportions_ztest(np.array([p1ch, p2ch]), np.array([n1ch, n2ch]),
                                    alternative="larger")
        if math.isnan(p):
            return True
        if not math.isnan(p) and p <= alpha:
            return False # Reject -> there is a difference
        else:
            return True # Accept -> there is no difference

    for prop in vals_dict.keys():
        all_columns = list(vals_dict[prop].keys()) #because table is a dictionary where keys are columns

        for t1 in all_columns:
            # we don't compare columns from no_stat_cols list, f.e. 'Total drivers'
            if t1 in no_stat_cols:
                continue
            tested_columns = list(vals_dict[prop].keys())
            tested_columns.remove(t1)

            for t2 in tested_columns:
                # we don't compare columns from no_stat_cols list, f.e. 'Total drivers'
                if t2 in no_stat_cols:
                    continue

                #DISCRETE DATA
                p1, n1, p2, n2 = vals_dict[prop][t1]['number'], vals_dict[prop][t1]['base'], \
                                vals_dict[prop][t2]['number'], vals_dict[prop][t2]['base']
                

                if p1 != '-' and p2 != '-' and n1 not in ['-', 0] and n2 not in ['-', 0]:
                    if p1 >= 5 and p2 >= 5:
                        try:
                            if not check_prop(p1, n1, p2, n2): #if not True >> Ha
                                vals_dict[prop][t1]['stat'].append(stat_letters[t2])
                        except:
                            pass

        for t in all_columns:
            vals_dict[prop][t]['stat'] = ", ".join(sorted([*set(vals_dict[prop][t]['stat'])])) if len(vals_dict[prop][t]['stat']) > 0 else "-"
            vals_dict[prop][t]['stat'] = vals_dict[prop][t]['stat'].strip(', ')
    return vals_dict