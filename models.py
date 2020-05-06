"""
Date: 2020-05-06
Author: figalit (github.com/figalit)

Module to load necessary models for tAMPer training and analysis.
"""

def roc(clf, X, y):
    """For an already fitted model, visualize the ROC curves for a custom 5 fold cross validation"""
    cv = StratifiedKFold(n_splits=5)
    name = type(clf).__name__
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        predicted = clf.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], predicted[:,1]) #probas_[:,1]
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i+=1
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC, {}".format(name))
    ax.legend(loc="lower right")
    # plt.savefig('figs/ROC_{}.png'.format(name))
    figname = 'ROC_{}.png'.format(name)
    plt.savefig(figname)
    print("Saved ROC to", figname)
