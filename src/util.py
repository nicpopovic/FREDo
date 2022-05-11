def get_f1(tp, fp, fn):


    if sum(tp.values())+sum(fp.values()) == 0:
        precision = 0
    else:
        precision = sum(tp.values())/(sum(tp.values())+sum(fp.values()))

    if sum(tp.values())+sum(fn.values()) == 0:
        recall = 0
    else:
        recall = sum(tp.values())/(sum(tp.values())+sum(fn.values()))

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return 100 * precision, 100 * recall, 100 * f1

def get_f1_macro(tp, fp, fn, prnt=False):
    if prnt:
        print('{:<10}  {:>10}  {:>10}  {:>10}    {:<10}'.format(f"type", f"precision", f"recall", f"f1", f"support"))

    p = []
    r = []
    f = []
    s = []
    for rtype in tp.keys():
        if tp[rtype]+fn[rtype] == 0:
            continue
        if tp[rtype]+fp[rtype] == 0:
            precision = 0
        else:
            precision = tp[rtype]/(tp[rtype]+fp[rtype])

        if tp[rtype]+fn[rtype] == 0:
            recall = 0
        else:
            recall = tp[rtype]/(tp[rtype]+fn[rtype])

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        support = tp[rtype]+fn[rtype]
        if prnt:
            print('{:<10}  {:>10}  {:>10}  {:>10}    {:<10}'.format(f"{rtype}", f"{100*precision:.2f}", f"{100*recall:.2f}", f"{100*f1:.2f}", f"{support}"))
        p.append(precision)
        r.append(recall)
        f.append(f1)
        s.append(support)

    if prnt:
        print('{:<10}  {:>10}  {:>10}  {:>10}    {:<10}'.format("-", "-", "-", "-", "-",))
        print('{:<10}  {:>10}  {:>10}  {:>10}'.format(f"macro", f"{100*sum(p)/len(p):.2f}", f"{100*sum(r)/len(r):.2f}", f"{100*sum(f)/len(f):.2f}"))
    
    return 100*sum(p)/len(p), 100*sum(r)/len(r), 100*sum(f)/len(f)
