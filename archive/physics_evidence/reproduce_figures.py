"""Recompute plotting data from archived experiments, verify, and render both figures.
Run: python reproduce_figures.py
No model training and no random data generation are performed.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'plot_data'; FIG=ROOT/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEEDS=list(range(42,47))
ARMS=['A0_none','B0_clean','B2_eta015','C_shuffle']
COLORS=['#2266A5','#D48128','#9361A6']
checks={}

def csv(df,name):
    df.to_csv(DATA/name,index=False,encoding='utf-8-sig',float_format='%.15g')

def validate_and_derive():
    manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        assert hashlib.sha256((ROOT/item['package_path']).read_bytes()).hexdigest()==item['sha256']
    checks['source_files_verified']=len(manifest)
    runs=[]
    supports={}
    for arm in ARMS:
        base=ROOT/'raw/teacher_gain'/arm
        df=pd.read_csv(base/'raw_runs.csv')
        assert len(df)==5 and set(df.seed)==set(SEEDS)
        assert (df.protocol=='sweep_group').all()
        for seed in SEEDS:
            row=df.loc[df.seed==seed].iloc[0]
            obj=json.loads((base/'runs'/f'raw5__residual32_gelu_relu_hybrid__sweep_group__{seed}.json').read_text(encoding='utf-8'))
            cm=np.asarray(obj['test']['confusion_matrix'],dtype=int)
            acc=np.trace(cm)/cm.sum()
            np.testing.assert_allclose(acc,row.test_accuracy,atol=1e-14,rtol=0)
            split=np.load(ROOT/'raw/reflection/splits'/f'sweep_group_seed{seed}'/'split.npz')
            assert len(split['test'])==cm.sum()
            if seed in supports: assert np.array_equal(cm.sum(axis=1),supports[seed])
            else: supports[seed]=cm.sum(axis=1)
            runs.append(dict(arm=arm,fold=seed-41,seed=seed,test_rows=int(cm.sum()),test_correct=int(np.trace(cm)),
                             student_accuracy_pct=acc*100,
                             teacher_validation_accuracy_pct=float(row.get('physics_teacher_validation_accuracy',np.nan))*100))
    run_df=pd.DataFrame(runs)
    csv(run_df,'figure1_accuracy_per_arm_fold.csv')
    wide=run_df.pivot(index='seed',columns='arm',values='student_accuracy_pct')
    paired=[]; summary=[]
    reference=pd.read_csv(ROOT/'reference_tables/T21_teacher_quality_vs_student_gain.csv').set_index('arm')
    for arm in ARMS[1:]:
        gains=wide[arm]-wide['A0_none']
        for seed in SEEDS:
            paired.append(dict(arm=arm,fold=seed-41,seed=seed,baseline_accuracy_pct=wide.loc[seed,'A0_none'],
                               student_accuracy_pct=wide.loc[seed,arm],gain_pp=gains.loc[seed]))
        mean=gains.mean(); sd=gains.std(ddof=1)
        np.testing.assert_allclose([mean,sd],[reference.loc[arm,'gain_pp'],reference.loc[arm,'gain_std_pp']],atol=1e-10,rtol=0)
        summary.append(dict(arm=arm,n_pairs=5,mean_gain_pp=mean,sample_sd_gain_pp=sd,
                            lower_mean_minus_sd_pp=mean-sd,upper_mean_plus_sd_pp=mean+sd,positive_folds=int((gains>0).sum())))
    paired=pd.DataFrame(paired);summary=pd.DataFrame(summary)
    csv(paired,'figure1_paired_gains.csv');csv(summary,'figure1_gain_summary.csv')
    checks['figure1_confusion_matrix_accuracy_recomputed']=True
    checks['figure1_fold_counts_and_class_support_match']=True
    checks['figure1_matches_archived_T21_mean_and_paired_sd']=True

    x=pd.read_csv(ROOT/'raw/dataset/x_train.csv',encoding='gbk').to_numpy(dtype=float)
    y=pd.read_csv(ROOT/'raw/dataset/y_train.csv',encoding='gbk').to_numpy().ravel().astype(int)
    frames=[]; max_diff=0.;max_replay_diff=0.
    for seed in SEEDS:
        run=f'sweep_group_seed{seed}'
        df=pd.read_csv(ROOT/'raw/reflection'/f'{run}_test_rows.csv.gz')
        ids=df.row_id.to_numpy(dtype=int);xx=x[ids]
        split=np.load(ROOT/'raw/reflection/splits'/run/'split.npz')
        assert np.array_equal(np.sort(ids),np.sort(split['test']))
        assert not set(split['test']) & set(split['train'])
        assert not set(split['test']) & set(split['valid'])
        assert np.array_equal(y[ids],df.label)
        assert np.array_equal(np.bincount(y[ids],minlength=15),supports[seed])
        measured=(xx[:,0]+1j*xx[:,1])/32768
        np.testing.assert_allclose(measured.real,df.gamma1_re,atol=1e-11,rtol=0)
        np.testing.assert_allclose(measured.imag,df.gamma1_im,atol=1e-11,rtol=0)
        with np.load(ROOT/'raw/reflection/models'/f'{run}_teacher.npz') as model:
            fi=np.searchsorted(model['frequencies'],xx[:,2])
            a,b,c=model['mobius_coefficients'][xx[:,3].astype(int),fi].T
            z=(xx[:,4]+1j*xx[:,5])/32768
            pred=(a*z+b)/(c*z+1)
        archived_pred=df.predicted_gamma1_re.to_numpy()+1j*df.predicted_gamma1_im.to_numpy()
        max_replay_diff=max(max_replay_diff,float(np.max(abs(pred-archived_pred))))
        np.testing.assert_allclose(pred,archived_pred,atol=2e-11,rtol=0)
        # Use archived per-record errors as plotting inputs, after independent replay validation.
        replay_error=abs(pred-measured)
        max_diff=max(max_diff,float(np.max(abs(replay_error-df.reflection_error))))
        np.testing.assert_allclose(replay_error,df.reflection_error,atol=2e-11,rtol=0)
        assert np.isfinite(df.reflection_error).all() and (df.reflection_error>=0).all()
        df['fold']=seed-41;df['seed']=seed
        df['band_MHz']=np.where(df.frequency_MHz<800,'703-748','824-849')
        frames.append(df)
    allrows=pd.concat(frames,ignore_index=True).sort_values('row_id')
    assert np.array_equal(allrows.row_id.to_numpy(),np.arange(96768))
    csv(allrows,'figure2_oof_records.csv')
    refs=pd.read_csv(ROOT/'reference_tables/C1_07_test_pooled_by_band.csv')
    refs=refs[refs.protocol=='sweep_group'].set_index('band')
    stats=[];cdf=[]
    for band in ['all','703-748','824-849']:
        sub=allrows if band=='all' else allrows[allrows.band_MHz==band]
        values=sub.reflection_error.to_numpy()
        row=dict(band_MHz=band,rows=len(values),mean=float(values.mean()),rmse=float(np.sqrt(np.mean(values**2))),
                 median=float(np.median(values)),p95=float(np.quantile(values,.95,method='linear')),maximum=float(values.max()))
        for key in ['mean','rmse','median','p95','maximum']:
            np.testing.assert_allclose(row[key],refs.loc[band,key],atol=1e-11,rtol=0)
        assert row['rows']==refs.loc[band,'rows']
        stats.append(row)
        if band!='all':
            vals,counts=np.unique(values,return_counts=True)
            cum=np.cumsum(counts)
            cdf.append(pd.DataFrame(dict(band_MHz=band,error=vals,count_at_error=counts,cumulative_count=cum,cdf=cum/len(values))))
    stats=pd.DataFrame(stats);cdf=pd.concat(cdf,ignore_index=True)
    csv(stats,'figure2_error_summary.csv');csv(cdf,'figure2_ecdf_coordinates.csv')
    checks.update(figure2_oof_rows=len(allrows),figure2_each_record_evaluated_once=True,
                  figure2_max_complex_prediction_replay_difference=max_replay_diff,
                  figure2_max_direct_error_replay_difference=max_diff,
                  figure2_matches_archived_C1_summary=True,
                  percentile_method='numpy quantile, method=linear',
                  figure1_uncertainty='sample SD of five paired accuracy differences, ddof=1; not a confidence interval',
                  figure1_pair_identity_note='Seeds align across original grouped protocol; test counts and class supports verified from confusion matrices against archived splits. Arm-specific row-level student predictions were not supplied in these run archives.',
                  runtime={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__})
    (ROOT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    return paired,summary,cdf,stats

def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'axes.linewidth':.8,
        'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})

def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=600 if ext=='png' else 300,bbox_inches='tight')
    plt.close(fig)

def plots(paired,summary,cdf,stats):
    style()
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    markers=['o','s','^','v','P']; offsets=np.linspace(-.16,.16,5)
    for i,arm in enumerate(ARMS[1:]):
        data=paired[paired.arm==arm].sort_values('seed')
        stat=summary[summary.arm==arm].iloc[0]
        for j,(_,row) in enumerate(data.iterrows()):
            ax.scatter(i+offsets[j],row.gain_pp,s=51,marker=markers[j],color=COLORS[i],edgecolor='white',linewidth=.6,zorder=4)
        ax.errorbar(i+.29,stat.mean_gain_pp,yerr=stat.sample_sd_gain_pp,fmt='D',color='#252525',
                    markersize=6,capsize=5,elinewidth=1.5,zorder=5)
        ax.text(i+.05,5.55,f'{stat.mean_gain_pp:+.3f} ± {stat.sample_sd_gain_pp:.3f}',ha='center',va='center',fontsize=10)
    ax.axhline(0,color='#333333',lw=1.1)
    ax.set_xticks(range(3),['Original\nteacher','Coefficient noise\n'+r'$\eta=0.15$','Frequency\nshuffle'])
    ax.set_ylabel('Student Top-1 accuracy gain\nover no-physics baseline (pp)')
    ax.set_ylim(-5.8,6.05);ax.set_xlim(-.5,2.6)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis='y',color='#dddddd',lw=.65,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([],[],color='#777777',marker=m,linestyle='none',markersize=6,label=f'Fold {i+1}') for i,m in enumerate(markers)]
    handles.append(Line2D([],[],color='#252525',marker='D',linestyle='none',markersize=6,label='Mean ± SD'))
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=9,handletextpad=.3,columnspacing=1.05,bbox_to_anchor=(.52,-.015))
    fig.subplots_adjust(bottom=.23,top=.95,left=.13,right=.99)
    save(fig,'Fig_Physical_Supervision_Paired_Gain')

    fig,axs=plt.subplots(1,2,figsize=(8.0,3.55))
    band_colors=['#2266A5','#D48128']; linestyles=['-','--']
    for band,col,ls in zip(['703-748','824-849'],band_colors,linestyles):
        curve=cdf[cdf.band_MHz==band]
        stat=stats[stats.band_MHz==band].iloc[0]
        xs=np.r_[0.,curve.error]; ys=np.r_[0.,curve.cdf]
        for ax in axs:
            ax.step(xs,ys,where='post',color=col,ls=ls,lw=1.6,label=f'{band} MHz (n = {int(stat.rows):,})')
        axs[0].plot([stat.p95,stat.p95],[0,.95],ls=':',color=col,lw=1)
        axs[0].scatter([stat.p95],[.95],s=27,color=col,zorder=4)
        axs[1].scatter([stat.maximum],[1],s=22,color=col,zorder=4)
    axs[0].axhline(.95,color='#777777',ls=':',lw=1)
    axs[0].text(.00025,.958,'95%',fontsize=9,color='#555555')
    axs[0].set(xlim=(0,.012),ylim=(0,1.02),title='(a) Central error distribution',ylabel='Empirical cumulative probability')
    axs[0].xaxis.set_major_locator(MultipleLocator(.004))
    axs[1].set(xlim=(.006,.047),ylim=(.945,1.002),title='(b) Upper tail')
    axs[1].yaxis.set_major_locator(MultipleLocator(.01))
    for ax in axs:
        ax.set_xlabel(r'Direct reflection error $|\widehat{\Gamma}_1-\Gamma_1|$')
        ax.grid(color='#e2e2e2',lw=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axs[0].text(.0115,.17,'P95\n703–748 MHz: 0.009813\n824–849 MHz: 0.006901',ha='right',va='bottom',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.92))
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=10,bbox_to_anchor=(.53,-.005))
    fig.subplots_adjust(left=.085,right=.985,bottom=.23,top=.9,wspace=.27)
    save(fig,'Fig_Surrogate_Heldout_Error_CDF')

if __name__=='__main__':
    p,s,c,t=validate_and_derive()
    plots(p,s,c,t)
    print('Verified all source hashes, reconstructed both figures, and saved full plotting data.')
    print(s.to_string(index=False));print(t.to_string(index=False))
