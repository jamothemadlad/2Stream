"""A reproducible one-step baseline study; whole runs are kept in separate splits."""

import json
import time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from .io import C


def load_runs(data_dir):
    runs = {}
    for path in sorted(Path(data_dir).glob('seed_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            run = {name:data[name] for name in data.files}
        seed = int(run['seed'])
        if seed in runs:raise ValueError(f'Duplicate seed {seed}')
        if run['phase_mass'].ndim != 4 or run['phase_mass'].shape[1] != 2:
            raise ValueError('Expected time, two streams, position, velocity')
        np.testing.assert_allclose(run['phase_mass'].sum(axis=(-2,-1)),1,rtol=2e-7)
        if not np.all(np.diff(run['times_s'])>0):raise ValueError('Invalid physical times')
        runs[seed]=run
    if not runs:raise FileNotFoundError(f'No processed data in {data_dir}')
    reference=next(iter(runs.values()))
    for run in runs.values():
        for key in ['z_edges_m','velocity_edges_over_c','times_s','field_z_m']:
            np.testing.assert_array_equal(run[key],reference[key])
    return runs


def check_split(runs, train, validation, test):
    groups=[set(train),set(validation),set(test)]
    if any(not group for group in groups):raise ValueError('Every split needs at least one whole seed run')
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('A seed cannot occur in more than one split')
    if not set.union(*groups).issubset(runs):raise ValueError('A requested seed is missing')


def window_data(run, pca, history):
    maps=run['phase_mass']
    coefficients=pca.transform(maps.reshape(len(maps),-1))
    nz=maps.shape[2]
    field=run['electric_field_V_m'][:,2]
    if field.shape[1]%nz:raise ValueError('Field grid must be divisible by the position-bin count')
    field=field.reshape(len(field),nz,-1).mean(axis=-1)
    features=np.column_stack([coefficients,field])
    indices=np.arange(history,len(maps))
    inputs=np.stack([features[i-history:i].ravel() for i in indices])
    return inputs,coefficients[indices],indices


def ballistic_transport(maps, z_edges, velocity_edges_over_c, delta_t):
    """Periodic transport in position; linear interpolation conserves each column's mass."""
    velocity=.5*(velocity_edges_over_c[:-1]+velocity_edges_over_c[1:])*C
    dz=z_edges[1]-z_edges[0]
    shifts=velocity*delta_t/dz
    output=np.zeros_like(maps)
    for j,shift in enumerate(shifts):
        whole=int(np.floor(shift));fraction=shift-whole
        output[...,j]=(1-fraction)*np.roll(maps[...,j],whole,axis=-1)+fraction*np.roll(maps[...,j],whole+1,axis=-1)
    return output


def metrics(truth, predicted):
    axes=tuple(range(1,truth.ndim))
    tiny=np.finfo(float).tiny
    l1=np.sum(abs(predicted-truth),axis=axes)/np.maximum(np.sum(abs(truth),axis=axes),tiny)
    l2=np.sqrt(np.sum((predicted-truth)**2,axis=axes)/np.maximum(np.sum(truth**2,axis=axes),tiny))
    density_true=truth.sum(axis=-1);density_pred=predicted.sum(axis=-1)
    density_l2=np.sqrt(np.sum((density_pred-density_true)**2,axis=(1,2))/np.maximum(np.sum(density_true**2,axis=(1,2)),tiny))
    negative=-np.minimum(predicted,0).sum(axis=axes)/np.sum(truth,axis=axes)
    number_error=abs(predicted.sum(axis=(-2,-1))-truth.sum(axis=(-2,-1)))
    return {'mean_relative_L1':float(l1.mean()),'mean_relative_L2':float(l2.mean()),
            'mean_density_relative_L2':float(density_l2.mean()),
            'mean_negative_mass_fraction':float(negative.mean()),
            'max_species_number_relative_error':float(number_error.max()),
            'samples':len(truth)}


def run_study(runs, train=(1,2,3), validation=(4,), test=(5,), components=32, history=3):
    check_split(runs,train,validation,test)
    if history<1:raise ValueError('History must be positive')
    training_maps=np.concatenate([runs[seed]['phase_mass'] for seed in train])
    if components>=min(len(training_maps),training_maps[0].size):
        raise ValueError('Too many PCA components for the training data')
    start=time.perf_counter()
    pca=PCA(n_components=components,svd_solver='randomized',random_state=0,iterated_power=4)
    pca.fit(training_maps.reshape(len(training_maps),-1))
    pca_seconds=time.perf_counter()-start
    windows={seed:window_data(run,pca,history) for seed,run in runs.items()}
    X=np.concatenate([windows[seed][0] for seed in train])
    y=np.concatenate([windows[seed][1] for seed in train])
    scaler=StandardScaler().fit(X)
    X=scaler.transform(X)

    def validation_error(model):
        scores=[]
        for seed in validation:
            x,_,indices=windows[seed]
            pred=pca.inverse_transform(model.predict(scaler.transform(x))).reshape(runs[seed]['phase_mass'][indices].shape)
            scores.append(metrics(runs[seed]['phase_mass'][indices],pred)['mean_relative_L1'])
        return float(np.mean(scores))

    start=time.perf_counter()
    linear_trials=[];selected_linear=None;linear_best=np.inf;linear_choice=None
    for alpha in [0.,.1,1.,10.]:
        model=LinearRegression() if alpha==0 else Ridge(alpha=alpha)
        model.fit(X,y)
        error=validation_error(model)
        linear_trials.append({'alpha':alpha,'validation_relative_L1':error})
        if error<linear_best:
            linear_best=error;selected_linear=model;linear_choice=alpha
    linear_seconds=time.perf_counter()-start
    start=time.perf_counter()
    forest_trials=[];selected_forest=None;forest_best=np.inf;forest_choice=None
    for depth,leaf in [(8,2),(None,2),(None,5)]:
        model=RandomForestRegressor(n_estimators=100,max_depth=depth,min_samples_leaf=leaf,
                                    max_features=1.0,random_state=0,n_jobs=2)
        model.fit(X,y)
        error=validation_error(model)
        choice={'max_depth':depth,'min_samples_leaf':leaf}
        forest_trials.append({**choice,'validation_relative_L1':error})
        if error<forest_best:
            forest_best=error;selected_forest=model;forest_choice=choice
    forest_seconds=time.perf_counter()-start
    models={'Linear regression':selected_linear,'Random forest':selected_forest}
    scores=[];predictions={}
    for seed in test:
        run=runs[seed];x,targets,indices=windows[seed]
        truth=run['phase_mass'][indices]
        last=run['phase_mass'][indices-1]
        delta=np.diff(run['times_s'])
        np.testing.assert_allclose(delta,delta[0],rtol=1e-10,atol=0)
        forecasts={'Persistence':last,
                   'Ballistic transport':ballistic_transport(last,run['z_edges_m'],run['velocity_edges_over_c'],float(delta[0]))}
        prediction_seconds={}
        for name,model in models.items():
            start=time.perf_counter()
            forecasts[name]=pca.inverse_transform(model.predict(scaler.transform(x))).reshape(truth.shape)
            prediction_seconds[name]=time.perf_counter()-start
        # Uses true future coefficients only as a compression diagnostic, never as model input.
        forecasts['PCA reconstruction diagnostic']=pca.inverse_transform(targets).reshape(truth.shape)
        for name,predicted in forecasts.items():
            if not np.isfinite(predicted).all():raise ValueError(f'Non-finite forecast: {name}')
            scores.append({'seed':seed,'method':name,**metrics(truth,predicted),
                           'batch_prediction_seconds':prediction_seconds.get(name)})
        predictions[seed]={'indices':indices,'truth':truth,**forecasts}
    report={
        'scope':'Exploratory one-step forecasting at one physical configuration; not a completed convergence or generalisation study',
        'split':{'train':list(train),'validation':list(validation),'test':list(test)},
        'pca_fit_seeds':list(train),'pca_fit_snapshots':len(training_maps),
        'pca_components':components,'history_snapshots':history,
        'pca_training_explained_variance_fraction':float(pca.explained_variance_ratio_.sum()),
        'forecast_inputs':'Past PCA coefficients and past spatial Ez profiles; no seed IDs, future states or future fields',
        'linear_selection':{'alpha':linear_choice,'trials':linear_trials},
        'random_forest_selection':{**forest_choice,'n_estimators':100,'trials':forest_trials},
        'pca_fit_seconds':pca_seconds,'linear_search_seconds':linear_seconds,
        'forest_search_seconds':forest_seconds,'test_scores':scores,
        'constraints':'No positivity clipping or number renormalisation applied to predictions',
        'limitations':['One held-out test seed in the default split; no confidence claim from many correlated snapshots.',
                       '32 PCA components and a history of 3 are initial choices; sensitivity studies remain.',
                       'Reference convergence, bin-resolution sensitivity, closed-loop forecasts and performance on other physical parameters are not established.',
                       'No acceleration relative to WarpX is claimed.'],
    }
    return {'report':report,'pca':pca,'scaler':scaler,'models':models,'predictions':predictions}


def plot_example(run,prediction,method='Random forest',sample=None):
    import matplotlib.pyplot as plt
    if sample is None:sample=len(prediction['indices'])//2
    truth=prediction['truth'][sample];predicted=prediction[method][sample]
    extent=[run['z_edges_m'][0],run['z_edges_m'][-1],run['velocity_edges_over_c'][0],run['velocity_edges_over_c'][-1]]
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for stream in [0,1]:
        vmax=max(float(truth[stream].max()),float(predicted[stream].max()))
        vmin=min(0.,float(predicted[stream].min()))
        difference=predicted[stream]-truth[stream]
        for col,array,title in [(0,truth[stream],'Reference'),(1,predicted[stream],method),(2,difference,'Prediction minus reference')]:
            kwargs={'vmin':vmin,'vmax':vmax,'cmap':'viridis'} if col<2 else {'cmap':'RdBu_r','vmin':-max(float(abs(difference).max()),1e-12),'vmax':max(float(abs(difference).max()),1e-12)}
            im=axes[stream,col].imshow(array.T,origin='lower',aspect='auto',extent=extent,**kwargs)
            axes[stream,col].set_title(f'{run["species"][stream]}: {title}')
            axes[stream,col].set_xlabel('Position z (m)')
            fig.colorbar(im,ax=axes[stream,col],shrink=.8,label='Normalised bin mass')
        axes[stream,0].set_ylabel('Physical velocity / c')
    step=int(run['iterations'][prediction['indices'][sample]])
    fig.suptitle(f'Held-out seed {int(run["seed"])} — one-step forecast at iteration {step}')
    fig.tight_layout()
    return fig
